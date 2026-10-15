"""Shared types used by the environment, reward functions, and controllers.

This module is the contract between ``environment.py`` and ``rewards.py``.
Keep it free of physics or reward logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Action(IntEnum):
    """Discrete action space.

    NOOP  - all engines off.
    MAIN  - main engine on: thrust along the rocket's body "up" axis.
    LEFT  - attitude thruster: positive angular acceleration (counter-clockwise,
            nose tilts toward -x).
    RIGHT - attitude thruster: negative angular acceleration (clockwise,
            nose tilts toward +x).
    """

    NOOP = 0
    MAIN = 1
    LEFT = 2
    RIGHT = 3


class Outcome(str, Enum):
    """Episode status after a step."""

    IN_PROGRESS = "in_progress"
    LANDED = "landed"
    CRASHED = "crashed"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self is not Outcome.IN_PROGRESS


@dataclass(frozen=True)
class RocketState:
    """Physical state of the rocket. Units: metres, seconds, radians.

    Coordinates are pad-relative: the pad centre is at (0, 0) and ``y`` is the
    height of the rocket's base above the ground (ground is y = 0).

    angle: 0 is upright, positive is counter-clockwise, wrapped to [-pi, pi].
    fuel:  remaining fuel as a fraction of capacity, in [0, 1].
    """

    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    fuel: float
