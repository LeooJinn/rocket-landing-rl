"""Record full episodes as JSON-serialisable trajectories for the browser viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rocket_landing.environment import EnvConfig, RocketLandingEnv
from rocket_landing.state import Outcome

FIELDS = ("x", "y", "vx", "vy", "angle", "angular_velocity", "fuel")


def _frame(t: int, state, action: int, reward: float, cum: float) -> dict[str, Any]:
    return {
        "t": t,
        "x": float(state.x),
        "y": float(state.y),
        "vx": float(state.vx),
        "vy": float(state.vy),
        "angle": float(state.angle),
        "av": float(state.angular_velocity),
        "fuel": float(state.fuel),
        "action": int(action),
        "reward": float(reward),
        "cum_reward": float(cum),
    }


def record_episode(
    controller: Any,
    seed: int,
    env_config: EnvConfig | None = None,
    reward_version: str = "v1",
    greedy: bool = True,
    run: str | None = None,
    checkpoint: str = "policy",
    update: int | None = None,
) -> dict[str, Any]:
    """Roll out ``controller.act(obs) -> int`` for one episode seeded with ``seed``.

    ``greedy`` is applied to controllers exposing a ``deterministic`` attribute
    (PolicyController); heuristic controllers are always deterministic.
    """
    if hasattr(controller, "deterministic"):
        controller.deterministic = bool(greedy)
    config = env_config or EnvConfig()
    env = RocketLandingEnv(config, reward_version=reward_version)
    obs, info = env.reset(seed=int(seed))
    cum = 0.0
    frames = [_frame(0, env.state, -1, 0.0, 0.0)]
    done = False
    while not done:
        action = int(controller.act(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        cum += reward
        frames.append(_frame(env.steps, env.state, action, reward, cum))
        done = terminated or truncated
    outcome = env.outcome
    if outcome is Outcome.IN_PROGRESS:  # pragma: no cover - defensive
        outcome = Outcome.TIMEOUT
    return {
        "meta": {
            "run": run,
            "checkpoint": checkpoint,
            "update": update,
            "seed": int(seed),
            "reward_version": reward_version,
            "dt": config.dt,
            "world": {
                "half_width": config.world_half_width,
                "height": config.world_height,
                "pad_half_width": config.pad_half_width,
            },
            "limits": {
                "safe_vx": config.max_landing_vx,
                "safe_vy": config.max_landing_vy,
                "safe_angle": config.max_landing_angle,
                "safe_angular_velocity": config.max_landing_angular_velocity,
            },
        },
        "outcome": outcome.value,
        "total_reward": cum,
        "frames": frames,
    }


def load_policy_from_checkpoint(path: str | Path, greedy: bool = True):
    """Load a checkpoint as a PolicyController (saved obs normaliser, argmax action)."""
    from rocket_landing.policy import PolicyController

    return PolicyController.from_checkpoint(path, deterministic=greedy)


def record_baseline_episode(
    seed: int, env_config: EnvConfig | None = None, reward_version: str = "v1"
) -> dict[str, Any]:
    """Episode flown by the hand-written heuristic controller."""
    from rocket_landing.baseline import HeuristicController

    config = env_config or EnvConfig()
    return record_episode(
        HeuristicController(config),
        seed,
        env_config=config,
        reward_version=reward_version,
        checkpoint="baseline",
    )
