"""Deterministic NumPy 2D rocket-landing simulator behind the Gymnasium API.

Units: metres, seconds, radians. Angle is positive counter-clockwise and the
body "up" vector is ``(-sin(angle), cos(angle))``. The ground is ``y = 0`` and
the pad is centred on ``x = 0``. Observations are raw physical units
(float32): ``[x, y, vx, vy, angle, angular_velocity, fuel]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from rocket_landing.rewards import REWARD_VERSIONS, compute_reward
from rocket_landing.state import Action, Outcome, RocketState

OBSERVATION_FIELDS: tuple[str, ...] = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "fuel",
)


@dataclass(frozen=True)
class EnvConfig:
    """Simulator parameters (see field comments for units)."""

    dt: float = 0.05  # s, fixed timestep (20 Hz)
    gravity: float = 9.81  # m/s^2, downward
    main_engine_accel: float = 19.62  # m/s^2 along body up axis (T/W = 2.0)
    side_engine_angular_accel: float = 1.0  # rad/s^2 from LEFT (+) / RIGHT (-)
    main_fuel_rate: float = 0.05  # fuel fraction per second while MAIN fires
    side_fuel_rate: float = 0.01  # fuel fraction per second while LEFT/RIGHT fires
    max_steps: int = 1000  # truncation (timeout) after this many steps
    pad_half_width: float = 10.0  # m
    world_half_width: float = 100.0  # |x| beyond this -> OUT_OF_BOUNDS
    world_height: float = 200.0  # y beyond this -> OUT_OF_BOUNDS
    max_landing_vx: float = 1.0  # m/s
    max_landing_vy: float = 2.0  # m/s (downward speed magnitude at touchdown)
    max_landing_angle: float = 0.2  # rad
    max_landing_angular_velocity: float = 0.5  # rad/s
    # Initial-state sampling ranges (uniform), each (low, high).
    init_x: tuple[float, float] = (-40.0, 40.0)
    init_y: tuple[float, float] = (80.0, 120.0)
    init_vx: tuple[float, float] = (-5.0, 5.0)
    init_vy: tuple[float, float] = (-10.0, 0.0)
    init_angle: tuple[float, float] = (-0.2, 0.2)
    init_angular_velocity: tuple[float, float] = (-0.1, 0.1)
    init_fuel: float = 1.0


def state_to_observation(state: RocketState) -> np.ndarray:
    """Convert a state to the float32 observation vector (shape ``(7,)``)."""
    return np.array(
        [getattr(state, name) for name in OBSERVATION_FIELDS], dtype=np.float32
    )


def action_to_controls(action: int | Action) -> tuple[float, float]:
    """Map a discrete action to ``(main_throttle in [0,1], side_throttle in [-1,1])``."""
    a = Action(int(action))
    if a is Action.MAIN:
        return 1.0, 0.0
    if a is Action.LEFT:
        return 0.0, 1.0
    if a is Action.RIGHT:
        return 0.0, -1.0
    return 0.0, 0.0


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped


def simulate_step(
    state: RocketState, main_throttle: float, side_throttle: float, config: EnvConfig
) -> RocketState:
    """Advance physics by one ``dt`` (semi-implicit Euler). Pure; no RNG.

    If the requested fuel burn exceeds the remaining fuel, both throttles are
    scaled by ``fuel / requested`` so the tank ends exactly empty. With no fuel
    the engines produce nothing. No ground handling happens here.
    """
    dt = config.dt
    requested = (
        main_throttle * config.main_fuel_rate
        + abs(side_throttle) * config.side_fuel_rate
    ) * dt
    if state.fuel <= 0.0 or requested <= 0.0:
        scale = 0.0 if state.fuel <= 0.0 else 1.0
        burned = 0.0
    elif requested > state.fuel:
        scale = state.fuel / requested
        burned = state.fuel
    else:
        scale = 1.0
        burned = requested
    main = main_throttle * scale
    side = side_throttle * scale

    ax = -math.sin(state.angle) * config.main_engine_accel * main
    ay = math.cos(state.angle) * config.main_engine_accel * main - config.gravity
    alpha = config.side_engine_angular_accel * side

    vx = state.vx + ax * dt
    vy = state.vy + ay * dt
    omega = state.angular_velocity + alpha * dt
    return RocketState(
        x=state.x + vx * dt,
        y=state.y + vy * dt,
        vx=vx,
        vy=vy,
        angle=wrap_angle(state.angle + omega * dt),
        angular_velocity=omega,
        fuel=max(0.0, state.fuel - burned),
    )


def _fuel_burned(prev: RocketState, new: RocketState) -> float:
    return prev.fuel - new.fuel


def _parse_action(action: Any) -> Action:
    """Validate and convert an action; raise ValueError for anything unsupported."""
    if isinstance(action, (bool, np.bool_)):
        raise ValueError(f"action must be an integer in 0..3, got bool {action!r}")
    if isinstance(action, np.ndarray):
        if action.ndim != 0 or not np.issubdtype(action.dtype, np.integer):
            raise ValueError(
                "array action must be a 0-d integer array, got "
                f"shape {action.shape} dtype {action.dtype}"
            )
        value = int(action)
    elif isinstance(action, (int, np.integer)):
        value = int(action)
    else:
        raise ValueError(
            f"action must be an Action, int, or numpy integer, got {type(action).__name__}"
        )
    if value not in (0, 1, 2, 3):
        raise ValueError(f"action must be in {{0, 1, 2, 3}}, got {value}")
    return Action(value)


class RocketLandingEnv(gymnasium.Env):
    """Discrete-action rocket landing task.

    Actions: 0 NOOP, 1 MAIN, 2 LEFT (+angular accel), 3 RIGHT (-angular accel).
    Observation: ``[x, y, vx, vy, angle, angular_velocity, fuel]`` (float32).
    Observation bounds are generous finite limits (positions 2x the world,
    velocities +-200 m/s and rad/s) so every reachable state is inside.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, reward_version: str = "v1"):
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(
                f"unknown reward version {reward_version!r}; known: {REWARD_VERSIONS}"
            )
        super().__init__()
        self.config = config or EnvConfig()
        self.reward_version = reward_version
        c = self.config
        high = np.array(
            [
                2 * c.world_half_width,
                2 * c.world_height,
                200.0,
                200.0,
                math.pi,
                200.0,
                1.0,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [-high[0], 0.0, -200.0, -200.0, -math.pi, -200.0, 0.0], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._state: RocketState | None = None
        self._outcome = Outcome.IN_PROGRESS
        self._steps = 0
        self._done = True

    # ------------------------------------------------------------------ props
    @property
    def state(self) -> RocketState:
        if self._state is None:
            raise RuntimeError("call reset() before accessing state")
        return self._state

    @property
    def outcome(self) -> Outcome:
        return self._outcome

    @property
    def steps(self) -> int:
        return self._steps

    # -------------------------------------------------------------- internals
    def _sample_state(self) -> RocketState:
        c, rng = self.config, self.np_random

        def u(r: tuple[float, float]) -> float:
            return float(rng.uniform(r[0], r[1]))

        return RocketState(
            x=u(c.init_x),
            y=u(c.init_y),
            vx=u(c.init_vx),
            vy=u(c.init_vy),
            angle=u(c.init_angle),
            angular_velocity=u(c.init_angular_velocity),
            fuel=c.init_fuel,
        )

    def _info(self, state: RocketState) -> dict[str, Any]:
        return {
            "outcome": self._outcome.value,
            "steps": self._steps,
            "fuel_exhausted": state.fuel <= 0.0,
            "state": state,
        }

    # -------------------------------------------------------------------- API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and options.get("state") is not None:
            start = options["state"]
            if not isinstance(start, RocketState):
                raise ValueError("options['state'] must be a RocketState")
            if not start.y > 0.0:
                raise ValueError(f"initial y must be > 0, got {start.y}")
        else:
            start = self._sample_state()
        self._state = start
        self._outcome = Outcome.IN_PROGRESS
        self._steps = 0
        self._done = False
        return state_to_observation(start), self._info(start)

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode has ended; call reset() before step()")
        act = _parse_action(action)
        c = self.config
        prev = self._state
        main, side = action_to_controls(act)
        new = simulate_step(prev, main, side, c)
        self._steps += 1

        terminated = truncated = False
        landing: dict[str, float] | None = None
        if new.y <= 0.0:
            new = replace(new, y=0.0)
            on_pad = abs(new.x) <= c.pad_half_width
            soft = (
                abs(new.vx) <= c.max_landing_vx
                and abs(new.vy) <= c.max_landing_vy
                and abs(new.angle) <= c.max_landing_angle
                and abs(new.angular_velocity) <= c.max_landing_angular_velocity
            )
            self._outcome = Outcome.LANDED if (on_pad and soft) else Outcome.CRASHED
            terminated = True
            landing = {
                "vx": new.vx,
                "vy": new.vy,
                "angle": new.angle,
                "angular_velocity": new.angular_velocity,
                "x": new.x,
            }
        elif abs(new.x) > c.world_half_width or new.y > c.world_height:
            self._outcome = Outcome.OUT_OF_BOUNDS
            terminated = True
        elif self._steps >= c.max_steps:
            self._outcome = Outcome.TIMEOUT
            truncated = True
        else:
            self._outcome = Outcome.IN_PROGRESS

        self._state = new
        self._done = terminated or truncated
        breakdown = compute_reward(
            prev, new, act, self._outcome, version=self.reward_version
        )
        info = self._info(new)
        info["reward_terms"] = breakdown.terms
        info["fuel_used"] = _fuel_burned(prev, new)
        if landing is not None:
            info["landing"] = landing
        return (
            state_to_observation(new),
            float(breakdown.total),
            terminated,
            truncated,
            info,
        )
