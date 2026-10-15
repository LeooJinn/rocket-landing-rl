"""Rocket landing reinforcement-learning environment."""

import gymnasium

from rocket_landing.environment import EnvConfig, RocketLandingEnv
from rocket_landing.state import Action, Outcome, RocketState

ENV_ID = "RocketLanding-v0"

if ENV_ID not in gymnasium.registry:
    gymnasium.register(
        id=ENV_ID, entry_point="rocket_landing.environment:RocketLandingEnv"
    )

__all__ = ["Action", "EnvConfig", "Outcome", "RocketLandingEnv", "RocketState"]
