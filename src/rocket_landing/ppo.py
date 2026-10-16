"""PPO from scratch: generalised advantage estimation and the clipped-objective update."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn

from rocket_landing.policy import ActorCritic


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 8
    norm_adv: bool = True
    target_kl: float | None = None
    # --- schedule options; the defaults reproduce the original constant-entropy / linear-to-0 LR ---
    # Entropy: if ``ent_coef_final`` is None the coefficient is constant. Otherwise it is held at
    # ``ent_coef`` until ``ent_decay_start`` (fraction of training), then decays linearly to
    # ``ent_coef_final`` at ``ent_decay_end`` and stays there.
    ent_coef_final: float | None = None
    ent_decay_start: float = 0.0
    ent_decay_end: float = 1.0
    # LR: with ``anneal_lr`` the factor goes from 1 to ``lr_final_frac`` over training following
    # ``lr_schedule`` ("linear" or "cosine"). lr_final_frac=0 + linear is the original schedule.
    lr_schedule: str = "linear"
    lr_final_frac: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PPOConfig:
        """Build a config from a dict, rejecting unknown keys."""
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown PPO config keys: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}; expected {LR_SCHEDULES}")
        if not 0.0 <= self.lr_final_frac <= 1.0:
            raise ValueError("lr_final_frac must be in [0, 1]")
        if not 0.0 <= self.ent_decay_start <= self.ent_decay_end <= 1.0:
            raise ValueError("need 0 <= ent_decay_start <= ent_decay_end <= 1")
        if self.ent_coef_final is not None and self.ent_coef_final < 0:
            raise ValueError("ent_coef_final must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LR_SCHEDULES = ("linear", "cosine")


def lr_at(cfg: PPOConfig, update: int, num_updates: int) -> float:
    """Learning rate for 1-based ``update``.

    Default (anneal_lr, linear, lr_final_frac=0) is exactly the original
    ``lr * (1 - (update - 1) / num_updates)``. ``anneal_lr=False`` is constant.
    """
    if not cfg.anneal_lr:
        return cfg.learning_rate
    progress = (update - 1) / num_updates
    if cfg.lr_schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        factor = 1.0 - progress
    return cfg.learning_rate * (cfg.lr_final_frac + (1.0 - cfg.lr_final_frac) * factor)


def ent_coef_at(cfg: PPOConfig, update: int, num_updates: int) -> float:
    """Entropy coefficient for 1-based ``update`` (constant unless ``ent_coef_final`` is set).

    Progress is (update - 1) / (num_updates - 1), so update 1 uses ``ent_coef`` and the last
    update uses exactly ``ent_coef_final`` (when ``ent_decay_end`` is 1).
    """
    if cfg.ent_coef_final is None:
        return cfg.ent_coef
    progress = (update - 1) / max(num_updates - 1, 1)
    if progress <= cfg.ent_decay_start:
        return cfg.ent_coef
    if progress >= cfg.ent_decay_end:
        return cfg.ent_coef_final
    frac = (progress - cfg.ent_decay_start) / (cfg.ent_decay_end - cfg.ent_decay_start)
    return cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(gamma, lam) over [T, N] rollouts; ``dones[t]`` cuts bootstrapping after step t.

    ``last_values`` ([N]) is V(s_T), the bootstrap value after the final step.
    Returns (advantages, returns) with returns = advantages + values.
    """
    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros_like(last_values)
    next_value = last_values
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_adv = delta + gamma * lam * not_done * next_adv
        advantages[t] = next_adv
        next_value = values[t]
    return advantages, advantages + values


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: PPOConfig,
    generator: torch.Generator,
    ent_coef: float | None = None,
) -> dict[str, float]:
    """Run ``cfg.update_epochs`` of minibatch PPO on a flat batch; return mean statistics.

    ``ent_coef`` overrides ``cfg.ent_coef`` for this update (used by the entropy schedule).
    """
    ent_coef = cfg.ent_coef if ent_coef is None else ent_coef
    b = batch["obs"].shape[0]
    var_returns = batch["returns"].var()
    explained_var = (
        float("nan")
        if var_returns == 0
        else float(1.0 - (batch["returns"] - batch["values"]).var() / var_returns)
    )
    sums = dict.fromkeys(
        ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm"], 0.0
    )
    n_steps = 0
    epochs_run = 0

    for _ in range(cfg.update_epochs):
        epochs_run += 1
        perm = torch.randperm(b, generator=generator)
        epoch_kl = 0.0
        epoch_mbs = 0
        for idx in torch.tensor_split(perm, cfg.num_minibatches):
            _, new_logp, entropy, new_value = model.get_action_and_value(
                batch["obs"][idx], batch["actions"][idx]
            )
            log_ratio = new_logp - batch["logprobs"][idx]
            ratio = log_ratio.exp()

            adv = batch["advantages"][idx]
            if cfg.norm_adv:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            pg_loss = torch.max(
                -adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
            ).mean()

            ret = batch["returns"][idx]
            v_loss = (new_value - ret) ** 2
            if cfg.clip_vloss:
                old_v = batch["values"][idx]
                v_clipped = old_v + (new_value - old_v).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss = torch.max(v_loss, (v_clipped - ret) ** 2)
            v_loss = 0.5 * v_loss.mean()

            ent = entropy.mean()
            loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = float(((ratio - 1) - log_ratio).mean())
                sums["clip_frac"] += float(((ratio - 1).abs() > cfg.clip_coef).float().mean())
            sums["policy_loss"] += float(pg_loss.detach())
            sums["value_loss"] += float(v_loss.detach())
            sums["entropy"] += float(ent.detach())
            sums["approx_kl"] += approx_kl
            sums["grad_norm"] += float(grad_norm)
            epoch_kl += approx_kl
            epoch_mbs += 1
            n_steps += 1

        if cfg.target_kl is not None and epoch_kl / epoch_mbs > cfg.target_kl:
            break

    stats = {k: v / n_steps for k, v in sums.items()}
    stats["explained_variance"] = explained_var
    stats["epochs_run"] = float(epochs_run)
    return stats
