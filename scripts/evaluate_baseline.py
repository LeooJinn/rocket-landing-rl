"""Evaluate the heuristic baseline controller.

Usage: uv run python scripts/evaluate_baseline.py --episodes 100 --seed 0 [--reward-version v1] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict

from rocket_landing.baseline import EpisodeResult, HeuristicController, evaluate
from rocket_landing.environment import EnvConfig


def crash_reasons(e: EpisodeResult, c: EnvConfig) -> list[str]:
    """Which landing thresholds a CRASHED episode violated."""
    reasons = []
    if abs(e.landing_x) > c.pad_half_width:
        reasons.append("off-pad")
    if abs(e.landing_vx) > c.max_landing_vx:
        reasons.append("vx")
    if abs(e.landing_vy) > c.max_landing_vy:
        reasons.append("vy")
    if abs(e.landing_angle) > c.max_landing_angle:
        reasons.append("angle")
    if abs(e.landing_angular_velocity) > c.max_landing_angular_velocity:
        reasons.append("spin")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-version", default="v1")
    ap.add_argument("--json", metavar="PATH", help="write summary and episodes as JSON")
    args = ap.parse_args()

    config = EnvConfig()
    summary = evaluate(
        HeuristicController(config),
        args.episodes,
        seed=args.seed,
        config=config,
        reward_version=args.reward_version,
    )
    rows = [
        ("episodes", f"{summary.n}"),
        ("success rate", f"{summary.success_rate:.1%}"),
        ("crash rate", f"{summary.crash_rate:.1%}"),
        ("out-of-bounds rate", f"{summary.out_of_bounds_rate:.1%}"),
        ("timeout rate", f"{summary.timeout_rate:.1%}"),
        ("mean reward", f"{summary.mean_reward:.2f}"),
        ("mean fuel used", f"{summary.mean_fuel_used:.3f}"),
        ("mean steps", f"{summary.mean_steps:.1f}"),
        ("mean touchdown |vx| (m/s)", f"{summary.mean_abs_vx:.3f}"),
        ("mean touchdown |vy| (m/s)", f"{summary.mean_abs_vy:.3f}"),
        ("mean touchdown |angle| (rad)", f"{summary.mean_landing_tilt:.3f}"),
    ]
    print(
        f"Heuristic baseline  seeds {args.seed}..{args.seed + args.episodes - 1}  "
        f"reward {args.reward_version}"
    )
    for name, value in rows:
        print(f"  {name:<30} {value:>10}")

    crashed = [e for e in summary.episodes if e.outcome == "crashed"]
    reasons: Counter[str] = Counter()
    for e in crashed:
        reasons.update(crash_reasons(e, config))
    print("Failure breakdown")
    print(
        f"  crashed {len(crashed)}, out_of_bounds "
        f"{sum(e.outcome == 'out_of_bounds' for e in summary.episodes)}, "
        f"timeout {sum(e.outcome == 'timeout' for e in summary.episodes)}"
    )
    if crashed:
        print("  crash threshold violations (an episode may violate several):")
        for name in ("off-pad", "vx", "vy", "angle", "spin"):
            print(f"    {name:<8} {reasons.get(name, 0)}")
        print("  crashed seeds: " + ", ".join(str(e.seed) for e in crashed))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "summary": {
                        k: v for k, v in asdict(summary).items() if k != "episodes"
                    },
                    "crash_reasons": dict(reasons),
                    "episodes": [asdict(e) for e in summary.episodes],
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
