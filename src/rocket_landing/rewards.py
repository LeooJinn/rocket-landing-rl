"""Versioned reward functions. (STUB - to be implemented by builder B.)"""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.state import Action, Outcome, RocketState

REWARD_VERSIONS: tuple[str, ...] = ("v1",)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    terms: dict[str, float] = field(default_factory=dict)


def compute_reward(
    prev_state: RocketState,
    state: RocketState,
    action: Action,
    outcome: Outcome,
    *,
    version: str = "v1",
) -> RewardBreakdown:
    if version not in REWARD_VERSIONS:
        raise ValueError(f"unknown reward version {version!r}; known: {REWARD_VERSIONS}")
    return RewardBreakdown(total=0.0, terms={})
