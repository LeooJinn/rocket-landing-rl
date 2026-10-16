"""Mean episode return of scripted behaviours under reward v2 / v3a / v3b (seeds 50000..).

    uv run python scripts/reward_scenarios.py [--episodes 100] [--out docs/reward_v3_scenarios.json]

Rewards never change the dynamics, so the same episodes are re-scored under each version
by rolling the same controller in three environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rocket_landing.baseline import HeuristicController, run_episode
from rocket_landing.environment import RocketLandingEnv
from rocket_landing.state import Action

VERSIONS = ("v2", "v3a", "v3b")


class FreeFall:
    """Never fires anything: the 'dive' / free-fall crash."""

    def act(self, obs) -> int:
        return int(Action.NOOP)


class HoverClock:
    """Heuristic attitude/horizontal control, but the vertical rule is 'hold altitude' (MAIN when
    vy <= 0): hovers over the pad until the tank is empty, then falls (the greedy-collapse mode)."""

    def __init__(self):
        self._h = HeuristicController()

    def act(self, obs) -> int:
        a = self._h.act(obs)
        if a in (int(Action.MAIN), int(Action.NOOP)):
            return int(Action.MAIN if obs[3] <= 0.0 else Action.NOOP)
        return a


SCENARIOS = {
    "heuristic (fast, frugal)": lambda: HeuristicController(),
    "heuristic descent_decel=0.5": lambda: HeuristicController(descent_decel=0.5),
    "heuristic v_max=4": lambda: HeuristicController(v_max=4.0),
    "heuristic descent_decel=0.2 (slow)": lambda: HeuristicController(descent_decel=0.2),
    "heuristic descent_decel=0.1 (slowest)": lambda: HeuristicController(descent_decel=0.1),
    "hover clock (hold altitude until empty)": HoverClock,
    "free fall (NOOP only)": FreeFall,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=50000)
    ap.add_argument("--out")
    a = ap.parse_args()
    table = {}
    print("| scenario | landed | steps | fuel used | fuel left | " + " | ".join(f"return {v}" for v in VERSIONS) + " |")
    print("|---|---|---|---|---|" + "---|" * len(VERSIONS))
    for name, make in SCENARIOS.items():
        per = {}
        for v in VERSIONS:
            env = RocketLandingEnv(reward_version=v)
            per[v] = [run_episode(env, make(), a.seed_start + i) for i in range(a.episodes)]
        e = per["v2"]
        row = {
            "landed": float(np.mean([x.outcome == "landed" for x in e])),
            "crashed": float(np.mean([x.outcome == "crashed" for x in e])),
            "steps": float(np.mean([x.steps for x in e])),
            "fuel_used": float(np.mean([x.fuel_used for x in e])),
            "return": {v: float(np.mean([x.total_reward for x in per[v]])) for v in VERSIONS},
        }
        table[name] = row
        print(f"| {name} | {row['landed']:.0%} | {row['steps']:.0f} | {row['fuel_used']:.2f} | "
              f"{1 - row['fuel_used']:.2f} | " + " | ".join(f"{row['return'][v]:.1f}" for v in VERSIONS) + " |")
    if a.out:
        Path(a.out).write_text(json.dumps({"episodes": a.episodes, "seed_start": a.seed_start,
                                           "scenarios": table}, indent=1) + "\n")


if __name__ == "__main__":
    main()
