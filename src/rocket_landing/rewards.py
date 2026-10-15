"""Versioned reward functions.

Reward v1 is documented in docs/reward-design.md. Never change the semantics
of an existing version in place: add a new version string and function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from rocket_landing.state import Action, Outcome, RocketState

REWARD_VERSIONS: tuple[str, ...] = ("v1",)


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-step reward. ``total`` always equals ``sum(terms.values())``."""

    total: float
    terms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardV1Config:
    """All v1 weights. Defaults define reward v1."""

    # Potential phi(s) = -(dist/dist_scale*w_dist + speed/speed_scale*w_speed + |angle|*w_angle)
    w_dist: float = 1.0
    dist_scale: float = 10.0  # metres per potential unit
    w_speed: float = 1.0
    speed_scale: float = 10.0  # m/s per potential unit
    w_angle: float = 1.0  # per radian
    # Fuel: penalty per step of full main-engine burn (0.05/s * 0.05 s).
    w_fuel: float = 0.03
    full_burn_fuel_per_step: float = 0.0025
    # Constant per-step cost.
    time_penalty: float = 0.01
    # Terminal rewards.
    landed: float = 100.0
    crashed: float = -100.0
    out_of_bounds: float = -100.0
    timeout: float = -50.0
    # Landing precision bonus (LANDED only), up to this value.
    precision_bonus: float = 20.0
    pad_half_width: float = 10.0
    safe_vx: float = 1.0
    safe_vy: float = 2.0
    safe_angle: float = 0.2
    safe_angular_velocity: float = 0.5


def potential(state: RocketState, cfg: RewardV1Config | None = None) -> float:
    """Shaping potential phi(s); higher is closer / slower / more upright."""
    cfg = cfg or RewardV1Config()
    dist = math.hypot(state.x, state.y)
    speed = math.hypot(state.vx, state.vy)
    return -(
        cfg.w_dist * dist / cfg.dist_scale
        + cfg.w_speed * speed / cfg.speed_scale
        + cfg.w_angle * abs(state.angle)
    )


def _precision(state: RocketState, cfg: RewardV1Config) -> float:
    """0..precision_bonus: 1 at pad centre with zero residual motion, 0 at the safe limits."""
    centred = max(0.0, 1.0 - abs(state.x) / cfg.pad_half_width)
    gentle = 1.0 - max(
        abs(state.vx) / cfg.safe_vx,
        abs(state.vy) / cfg.safe_vy,
        abs(state.angle) / cfg.safe_angle,
        abs(state.angular_velocity) / cfg.safe_angular_velocity,
    )
    return cfg.precision_bonus * centred * max(0.0, min(1.0, gentle))


def _reward_v1(
    prev_state: RocketState,
    state: RocketState,
    action: Action,
    outcome: Outcome,
    cfg: RewardV1Config,
) -> dict[str, float]:
    shaping = potential(state, cfg) - potential(prev_state, cfg)
    burned = max(0.0, prev_state.fuel - state.fuel)
    fuel = -cfg.w_fuel * burned / cfg.full_burn_fuel_per_step
    terminal = {
        Outcome.LANDED: cfg.landed,
        Outcome.CRASHED: cfg.crashed,
        Outcome.OUT_OF_BOUNDS: cfg.out_of_bounds,
        Outcome.TIMEOUT: cfg.timeout,
    }.get(outcome, 0.0)
    precision = _precision(state, cfg) if outcome is Outcome.LANDED else 0.0
    return {
        "shaping": shaping,
        "fuel": fuel,
        "time": -cfg.time_penalty,
        "terminal": terminal,
        "precision": precision,
    }


_DISPATCH: dict[str, Callable[..., dict[str, float]]] = {"v1": _reward_v1}


def compute_reward(
    prev_state: RocketState,
    state: RocketState,
    action: Action,
    outcome: Outcome,
    *,
    version: str = "v1",
    config: RewardV1Config | None = None,
) -> RewardBreakdown:
    if version not in _DISPATCH:
        raise ValueError(f"unknown reward version {version!r}; known: {REWARD_VERSIONS}")
    terms = {
        k: float(v)
        for k, v in _DISPATCH[version](
            prev_state, state, action, outcome, config or RewardV1Config()
        ).items()
    }
    return RewardBreakdown(total=float(sum(terms.values())), terms=terms)
