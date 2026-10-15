"""Actor-critic network, observation normaliser, checkpoints and a controller wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class RunningMeanStd:
    """Running mean/variance of observations (Chan parallel merge), with clipped output."""

    def __init__(self, shape: tuple[int, ...] = (7,), epsilon: float = 1e-4, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.epsilon = float(epsilon)
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        """Merge the statistics of a batch ``x`` of shape [N, *shape]."""
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        delta = x.mean(axis=0) - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m2 = self.var * self.count + x.var(axis=0) * n + delta**2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return clip((x - mean) / sqrt(var + 1e-8)) as float32; works for [7] and [N, 7]."""
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
            "clip": self.clip,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])
        self.clip = float(d["clip"])
        self.epsilon = float(d["epsilon"])

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> RunningMeanStd:
        obj = cls(shape=np.shape(d["mean"]), epsilon=d["epsilon"], clip=d["clip"])
        obj.load_state_dict(d)
        return obj


# Fixed physical observation scales, order [x, y, vx, vy, angle, angular_velocity, fuel].
# Chosen from the ranges in docs/environment.md: x is in [-100, 100] before going out of
# bounds (start +-40), y in [0, 200] (start 80-120), vx/vy are a few m/s to tens of m/s
# (start |v| <= 10, landing needs <= 1-2), tilt matters at ~0.2 rad (start +-0.2),
# angular velocity matters at ~0.5 rad/s, fuel is already a [0, 1] fraction.
FIXED_OBS_SCALES = (50.0, 100.0, 10.0, 10.0, 0.3, 0.5, 1.0)


class FixedScaleNormalizer:
    """Stateless normaliser: clip(x / scale, -clip, clip). ``update`` is a no-op."""

    kind = "fixed"

    def __init__(self, scales: tuple[float, ...] = FIXED_OBS_SCALES, clip: float = 10.0):
        self.scales = np.asarray(scales, dtype=np.float64)
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        """Fixed scales are never updated."""

    def normalize(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64) / self.scales
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"type": "fixed", "scales": self.scales.tolist(), "clip": self.clip}

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> FixedScaleNormalizer:
        return cls(tuple(d["scales"]), clip=d["clip"])


def make_obs_normalizer(kind: str, shape: tuple[int, ...] = (7,)):
    """Build a normaliser from the ``obs_norm`` config value (``running`` or ``fixed``)."""
    if kind == "running":
        return RunningMeanStd(shape=shape)
    if kind == "fixed":
        return FixedScaleNormalizer()
    raise ValueError(f"unknown obs_norm {kind!r}; expected 'running' or 'fixed'")


def obs_normalizer_from_state_dict(d: dict[str, Any]):
    """Rebuild a normaliser; dicts without a ``type`` key are legacy running stats."""
    if d.get("type") == "fixed":
        return FixedScaleNormalizer.from_state_dict(d)
    return RunningMeanStd.from_state_dict(d)


def _mlp(sizes: list[int], activation: type[nn.Module], out_gain: float) -> nn.Sequential:
    """MLP with orthogonal init: sqrt(2) on hidden layers, ``out_gain`` on the head."""
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        linear = nn.Linear(n_in, n_out)
        nn.init.orthogonal_(linear.weight, out_gain if last else 2**0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs over a (normalised) observation."""

    _ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}

    def __init__(
        self,
        obs_dim: int = 7,
        n_actions: int = 4,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.kwargs: dict[str, Any] = {
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden_sizes": list(hidden_sizes),
            "activation": activation,
        }
        act = self._ACTIVATIONS[activation]
        hidden = list(hidden_sizes)
        self.actor = _mlp([obs_dim, *hidden, n_actions], act, 0.01)
        self.critic = _mlp([obs_dim, *hidden, 1], act, 1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits [B, n_actions], value [B])."""
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, entropy, value); samples with ``generator`` if no action."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Argmax action for each row of ``obs``."""
        return self.actor(obs).argmax(dim=-1)


def save_checkpoint(
    path: str | Path,
    model: ActorCritic,
    obs_norm: RunningMeanStd | FixedScaleNormalizer,
    config: dict,
    extra: dict | None = None,
) -> None:
    """Write model weights, normaliser stats and config as plain types (weights_only-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "policy_kwargs": model.kwargs,
        "obs_norm": obs_norm.state_dict(),
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[ActorCritic, RunningMeanStd, dict]:
    """Load a checkpoint; returns (model in eval mode, normaliser, raw payload)."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = ActorCritic(**payload["policy_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, obs_normalizer_from_state_dict(payload["obs_norm"]), payload


class PolicyController:
    """Wraps a trained policy as an ``act(obs) -> int`` controller with frozen normalisation."""

    def __init__(
        self,
        model: ActorCritic,
        obs_norm: RunningMeanStd,
        device: str = "cpu",
        deterministic: bool = True,
        seed: int = 0,
    ):
        self.model = model.to(device).eval()
        self.obs_norm = obs_norm
        self.device = device
        self.deterministic = deterministic
        self._generator = torch.Generator().manual_seed(seed)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu", deterministic: bool = True
    ) -> PolicyController:
        model, obs_norm, _ = load_checkpoint(path, device)
        return cls(model, obs_norm, device=device, deterministic=deterministic)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.as_tensor(self.obs_norm.normalize(obs), device=self.device).unsqueeze(0)
        if self.deterministic:
            return int(self.model.greedy_action(x).item())
        logits, _ = self.model(x)
        probs = torch.softmax(logits, dim=-1).cpu()
        return int(torch.multinomial(probs, 1, generator=self._generator).item())
