"""Hand-written heuristic controller (no learning) and evaluation helpers.

Cascade, evaluated every step from the observation only (deterministic):

1. Desired tilt from horizontal error: a target vx (proportional to -x,
   capped) and a P law on the vx error give a desired horizontal acceleration; since ``ax = -sin(angle) * thrust`` the nose is tilted toward
   +x (negative angle) to move toward x = 0. The tilt is clipped to a limit
   that shrinks linearly to ``min_tilt`` near the ground.
2. Attitude PD (angle error, angular velocity) gives a desired angular
   acceleration; if its magnitude exceeds ``att_deadband`` fire LEFT/RIGHT.
3. Otherwise choose MAIN vs NOOP: fire MAIN when vy is below an
   altitude-dependent target descent speed (a constant-deceleration profile,
   ``-sqrt(2 * a * y)`` clipped to ``[-v_max, -v_touch]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rocket_landing.environment import EnvConfig, RocketLandingEnv
from rocket_landing.state import Action, Outcome

DEFAULT_GAINS: dict[str, float] = {
    "kx": 0.298,  # target vx per metre of x error (1/s)
    "vx_max": 8.94,  # m/s, cap on target horizontal speed
    "kvx": 0.5825,  # desired ax per m/s of vx error (1/s)
    "max_tilt": 0.45,  # rad, tilt limit at altitude
    "min_tilt": 0.037,  # rad, tilt limit at the ground
    "tilt_fade_height": 25.7,  # m, below this the tilt limit shrinks
    "kp_angle": 4.0,  # desired angular accel per rad of angle error
    "kd_angle": 1.74,  # desired angular accel per rad/s of omega
    "att_deadband": 0.205,  # |desired alpha| above this fires an attitude thruster
    "descent_decel": 1.0,  # m/s^2 assumed net braking for the target profile
    "v_max": 20.0,  # m/s, cap on target descent speed
    "v_touch": 1.02,  # m/s, target descent speed at the ground
    "vy_margin": 0.07,  # fire MAIN when vy < target - margin
}


class HeuristicController:
    """Deterministic cascade controller for the discrete action space."""

    def __init__(self, config: EnvConfig | None = None, **gains: float):
        unknown = set(gains) - set(DEFAULT_GAINS)
        if unknown:
            raise TypeError(f"unknown gains: {sorted(unknown)}")
        self.config = config or EnvConfig()
        self.gains = {**DEFAULT_GAINS, **gains}

    def act(self, obs: np.ndarray) -> int:
        g = self.gains
        x, y, vx, vy, angle, omega = (float(v) for v in obs[:6])
        cfg = self.config

        # (a) desired tilt
        vx_target = float(np.clip(-g["kx"] * x, -g["vx_max"], g["vx_max"]))
        ax_des = g["kvx"] * (vx_target - vx)  # desired horizontal accel
        a_ref = max(cfg.gravity, 1e-6)
        tilt_lim = g["min_tilt"] + (g["max_tilt"] - g["min_tilt"]) * min(
            1.0, max(0.0, y / g["tilt_fade_height"])
        )
        angle_des = float(np.clip(-ax_des / a_ref, -tilt_lim, tilt_lim))

        # (b) attitude correction
        alpha_des = g["kp_angle"] * (angle_des - angle) - g["kd_angle"] * omega
        if alpha_des > g["att_deadband"]:
            return int(Action.LEFT)
        if alpha_des < -g["att_deadband"]:
            return int(Action.RIGHT)

        # (c) vertical: suicide-burn style target descent speed
        v_target = -min(
            g["v_max"],
            max(g["v_touch"], math.sqrt(2.0 * g["descent_decel"] * max(y, 0.0))),
        )
        return int(Action.MAIN if vy < v_target - g["vy_margin"] else Action.NOOP)


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    outcome: str
    steps: int
    total_reward: float
    fuel_used: float
    landing_vx: float | None
    landing_vy: float | None
    landing_angle: float | None
    landing_x: float | None
    landing_angular_velocity: float | None = None


def run_episode(env: RocketLandingEnv, controller, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total, fuel, steps = 0.0, 0.0, 0
    terminated = truncated = False
    landing = None
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(controller.act(obs))
        total += r
        fuel += info["fuel_used"]
        steps += 1
        landing = info.get("landing", landing)
    return EpisodeResult(
        seed=seed,
        outcome=str(info["outcome"]),
        steps=steps,
        total_reward=float(total),
        fuel_used=float(fuel),
        landing_vx=None if landing is None else float(landing["vx"]),
        landing_vy=None if landing is None else float(landing["vy"]),
        landing_angle=None if landing is None else float(landing["angle"]),
        landing_x=None if landing is None else float(landing["x"]),
        landing_angular_velocity=(
            None if landing is None else float(landing["angular_velocity"])
        ),
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@dataclass(frozen=True)
class EvalSummary:
    n: int
    success_rate: float
    crash_rate: float
    out_of_bounds_rate: float
    timeout_rate: float
    mean_reward: float
    mean_fuel_used: float
    mean_steps: float
    mean_landing_speed: float  # mean sqrt(vx^2 + vy^2) over touchdowns
    mean_landing_tilt: float  # mean |angle| over touchdowns
    mean_abs_vx: float
    mean_abs_vy: float
    episodes: list[EpisodeResult] = field(default_factory=list)


def evaluate(
    controller,
    n_episodes: int,
    seed: int = 0,
    config: EnvConfig | None = None,
    reward_version: str = "v1",
) -> EvalSummary:
    env = RocketLandingEnv(config, reward_version=reward_version)
    eps = [run_episode(env, controller, seed + i) for i in range(n_episodes)]
    n = len(eps)

    def rate(o: Outcome) -> float:
        return sum(e.outcome == o.value for e in eps) / n if n else float("nan")

    td = [e for e in eps if e.landing_vx is not None]
    return EvalSummary(
        n=n,
        success_rate=rate(Outcome.LANDED),
        crash_rate=rate(Outcome.CRASHED),
        out_of_bounds_rate=rate(Outcome.OUT_OF_BOUNDS),
        timeout_rate=rate(Outcome.TIMEOUT),
        mean_reward=_mean([e.total_reward for e in eps]),
        mean_fuel_used=_mean([e.fuel_used for e in eps]),
        mean_steps=_mean([e.steps for e in eps]),
        mean_landing_speed=_mean([math.hypot(e.landing_vx, e.landing_vy) for e in td]),
        mean_landing_tilt=_mean([abs(e.landing_angle) for e in td]),
        mean_abs_vx=_mean([abs(e.landing_vx) for e in td]),
        mean_abs_vy=_mean([abs(e.landing_vy) for e in td]),
        episodes=eps,
    )
