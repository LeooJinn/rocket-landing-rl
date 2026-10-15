"""PPO training loop for the rocket-landing environment.

Usage:
    uv run python -m rocket_landing.train --config experiments/configs/ppo_v1.json \
        --seed 0 [--run-name NAME] [--out-dir experiments/results] \
        [--total-timesteps N] [--device cpu|mps|cuda]

Seeding. Python ``random``, numpy and torch are seeded with ``seed``; action
sampling and minibatch shuffling use two separate ``torch.Generator`` objects,
both seeded with ``seed``. Training env ``i`` is reset ONCE with
``1_000_000 + seed * 1000 + i`` and afterwards reset without a seed (it
continues its own RNG stream). Reset seeds 0..99999, which baseline and
evaluation use, are therefore never used to seed training.

Rewards are multiplied by ``reward_scale`` for learning only; every logged
reward is the raw environment reward. Timeouts are treated as terminal for GAE.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from rocket_landing.baseline import evaluate
from rocket_landing.environment import RocketLandingEnv
from rocket_landing.policy import (
    ActorCritic,
    PolicyController,
    make_obs_normalizer,
    save_checkpoint,
)
from rocket_landing.ppo import PPOConfig, compute_gae, ppo_update

OBS_DIM = 7
TRAIN_SEED_BASE = 1_000_000

DEFAULT_CONFIG: dict = {
    "total_timesteps": 2_000_000,
    "num_envs": 16,
    "num_steps": 256,
    "reward_version": "v1",
    "reward_scale": 0.01,
    "obs_norm": "running",  # "running" (RunningMeanStd, updated in training) or "fixed" scales
    "ppo": {
        "learning_rate": 3e-4,
        "anneal_lr": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 10,
        "num_minibatches": 8,
        "norm_adv": True,
        "target_kl": None,
    },
    "policy": {"hidden_sizes": [64, 64], "activation": "tanh"},
    "checkpoint_every": 50,
    "eval_every": 10,
    "eval_episodes": 50,
    "eval_seed_start": 20000,
    "torch_threads": 1,
}

_NESTED = ("ppo", "policy")


def resolve_config(overrides: dict | None) -> dict:
    """Merge ``overrides`` over DEFAULT_CONFIG, rejecting unknown keys."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (overrides or {}).items():
        if key not in DEFAULT_CONFIG:
            raise ValueError(f"unknown config key {key!r}")
        if key in _NESTED:
            unknown = set(value) - set(DEFAULT_CONFIG[key])
            if unknown:
                raise ValueError(f"unknown {key} config keys: {sorted(unknown)}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return resolve_config(json.load(f))


def training_env_seed(seed: int, i: int) -> int:
    return TRAIN_SEED_BASE + seed * 1000 + i


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _write_line(f, record: dict) -> None:
    f.write(json.dumps({k: _jsonable(v) for k, v in record.items()}) + "\n")
    f.flush()


def _frac(outcomes: list[str], name: str):
    return outcomes.count(name) / len(outcomes) if outcomes else None


def _mean_or_none(values: list[float]):
    return float(np.mean(values)) if values else None


def train(config: dict, seed: int, run_dir: str | Path, device: str = "cpu") -> dict:
    cfg = resolve_config(config)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg["torch_threads"]))
    act_gen = torch.Generator().manual_seed(seed)
    shuffle_gen = torch.Generator().manual_seed(seed)

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "config": cfg,
                "seed": seed,
                "device": device,
                "torch_version": torch.__version__,
            },
            f,
            indent=2,
        )

    n_envs, n_steps = int(cfg["num_envs"]), int(cfg["num_steps"])
    batch_size = n_envs * n_steps
    num_updates = max(1, int(cfg["total_timesteps"]) // batch_size)
    ppo_cfg = PPOConfig.from_dict(cfg["ppo"])
    scale = float(cfg["reward_scale"])
    version = cfg["reward_version"]

    dev = torch.device(device)
    model = ActorCritic(
        obs_dim=OBS_DIM,
        n_actions=4,
        hidden_sizes=tuple(cfg["policy"]["hidden_sizes"]),
        activation=cfg["policy"]["activation"],
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)
    obs_norm = make_obs_normalizer(cfg["obs_norm"], shape=(OBS_DIM,))

    envs = [RocketLandingEnv(reward_version=version) for _ in range(n_envs)]
    raw_obs = np.stack(
        [e.reset(seed=training_env_seed(seed, i))[0] for i, e in enumerate(envs)]
    ).astype(np.float32)
    ep_return = np.zeros(n_envs)
    ep_fuel = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=np.int64)

    buf_obs = torch.zeros((n_steps, n_envs, OBS_DIM), device=dev)
    buf_act = torch.zeros((n_steps, n_envs), dtype=torch.long, device=dev)
    buf_logp = torch.zeros((n_steps, n_envs), device=dev)
    buf_val = torch.zeros((n_steps, n_envs), device=dev)
    buf_rew = torch.zeros((n_steps, n_envs), device=dev)
    buf_done = torch.zeros((n_steps, n_envs), device=dev)

    eval_every = int(cfg["eval_every"])
    best = None  # (success_rate, mean_reward)
    best_update = None
    last_eval = None
    cum_episodes = 0
    global_step = 0
    t0 = time.time()
    metrics_f = open(run_dir / "metrics.jsonl", "w")
    eval_f = open(run_dir / "eval.jsonl", "w")

    def norm_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs_norm.normalize(x), dtype=torch.float32, device=dev)

    def make_ckpt(path: Path, update: int) -> None:
        save_checkpoint(
            path,
            model.cpu(),
            obs_norm,
            cfg,
            extra={"update": update, "global_step": global_step, "seed": seed},
        )
        model.to(dev)

    try:
        for update in range(1, num_updates + 1):
            lr = ppo_cfg.learning_rate
            if ppo_cfg.anneal_lr:
                lr = ppo_cfg.learning_rate * (1.0 - (update - 1) / num_updates)
            for g in optimizer.param_groups:
                g["lr"] = lr

            ep_out: list[str] = []
            ep_rets: list[float] = []
            ep_lens: list[int] = []
            ep_fuels: list[float] = []

            for t in range(n_steps):
                obs_norm.update(raw_obs.astype(np.float64))
                obs_t = norm_tensor(raw_obs)
                with torch.no_grad():
                    action, logp, _, value = model.get_action_and_value(
                        obs_t, generator=act_gen
                    )
                buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = (
                    obs_t,
                    action,
                    logp,
                    value,
                )
                actions = action.cpu().numpy()
                rewards = np.zeros(n_envs)
                dones = np.zeros(n_envs)
                next_raw = np.empty_like(raw_obs)
                for i, env in enumerate(envs):
                    o, r, term, trunc, info = env.step(int(actions[i]))
                    ep_return[i] += r
                    ep_fuel[i] += info["fuel_used"]
                    ep_len[i] += 1
                    rewards[i] = r
                    if term or trunc:
                        dones[i] = 1.0
                        ep_out.append(str(info["outcome"]))
                        ep_rets.append(float(ep_return[i]))
                        ep_lens.append(int(ep_len[i]))
                        ep_fuels.append(float(ep_fuel[i]))
                        ep_return[i] = ep_fuel[i] = 0.0
                        ep_len[i] = 0
                        o, _ = env.reset()
                    next_raw[i] = o
                raw_obs = next_raw
                buf_rew[t] = torch.as_tensor(rewards * scale, dtype=torch.float32, device=dev)
                buf_done[t] = torch.as_tensor(dones, dtype=torch.float32, device=dev)
                global_step += n_envs

            with torch.no_grad():
                _, last_val = model(norm_tensor(raw_obs))
            adv, ret = compute_gae(
                buf_rew, buf_val, buf_done, last_val, ppo_cfg.gamma, ppo_cfg.gae_lambda
            )
            batch = {
                "obs": buf_obs.reshape(-1, OBS_DIM),
                "actions": buf_act.reshape(-1),
                "logprobs": buf_logp.reshape(-1),
                "values": buf_val.reshape(-1),
                "advantages": adv.reshape(-1),
                "returns": ret.reshape(-1),
            }
            stats = ppo_update(model, optimizer, batch, ppo_cfg, shuffle_gen)

            cum_episodes += len(ep_out)
            record = {
                "update": update,
                "global_step": global_step,
                "episodes": len(ep_out),
                "cumulative_episodes": cum_episodes,
                "ep_reward_mean": _mean_or_none(ep_rets),
                "ep_length_mean": _mean_or_none(ep_lens),
                "fuel_used_mean": _mean_or_none(ep_fuels),
                "success_rate": _frac(ep_out, "landed"),
                "crash_rate": _frac(ep_out, "crashed"),
                "out_of_bounds_rate": _frac(ep_out, "out_of_bounds"),
                "timeout_rate": _frac(ep_out, "timeout"),
                "lr": lr,
                **{k: float(v) for k, v in stats.items()},
                "time_s": time.time() - t0,
            }
            _write_line(metrics_f, record)

            line = (
                f"upd {update}/{num_updates} step {global_step} "
                f"eps {len(ep_out)} "
                f"R {record['ep_reward_mean'] if record['ep_reward_mean'] is not None else float('nan'):.1f} "
                f"succ {record['success_rate'] if record['success_rate'] is not None else float('nan'):.2f} "
                f"ent {stats.get('entropy', float('nan')):.3f} "
                f"kl {stats.get('approx_kl', float('nan')):.4f} "
                f"sps {global_step / max(time.time() - t0, 1e-9):.0f}"
            )

            if eval_every and (update % eval_every == 0 or update == num_updates):
                controller = PolicyController(
                    model.cpu().eval(), copy.deepcopy(obs_norm), device="cpu", deterministic=True
                )
                s = evaluate(
                    controller,
                    int(cfg["eval_episodes"]),
                    seed=int(cfg["eval_seed_start"]),
                    reward_version=version,
                )
                model.to(dev).train()
                last_eval = {
                    "update": update,
                    "global_step": global_step,
                    "success_rate": s.success_rate,
                    "crash_rate": s.crash_rate,
                    "out_of_bounds_rate": s.out_of_bounds_rate,
                    "timeout_rate": s.timeout_rate,
                    "mean_reward": s.mean_reward,
                    "mean_fuel_used": s.mean_fuel_used,
                    "mean_steps": s.mean_steps,
                }
                _write_line(eval_f, last_eval)
                key = (s.success_rate, s.mean_reward)
                if best is None or key > best:
                    best, best_update = key, update
                    make_ckpt(run_dir / "best.pt", update)
                line += f" | eval succ {s.success_rate:.2f} R {s.mean_reward:.1f}"

            ck = int(cfg["checkpoint_every"])
            if ck and update % ck == 0:
                make_ckpt(run_dir / "checkpoints" / f"ckpt_{update:05d}.pt", update)
            print(line, flush=True)
    finally:
        metrics_f.close()
        eval_f.close()

    make_ckpt(run_dir / "final.pt", num_updates)
    summary = {
        "seed": seed,
        "num_updates": num_updates,
        "total_env_steps": global_step,
        "episodes": cum_episodes,
        "wallclock_s": time.time() - t0,
        "steps_per_s": global_step / max(time.time() - t0, 1e-9),
        "last_eval": last_eval,
        "best_eval": None
        if best is None
        else {"update": best_update, "success_rate": best[0], "mean_reward": best[1]},
        "run_dir": str(run_dir),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump({k: _jsonable(v) for k, v in summary.items()}, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name")
    ap.add_argument("--out-dir", default="experiments/results")
    ap.add_argument("--total-timesteps", type=int)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    name = args.run_name or f"{Path(args.config).stem}_s{args.seed}"
    summary = train(cfg, args.seed, Path(args.out_dir) / name, device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
