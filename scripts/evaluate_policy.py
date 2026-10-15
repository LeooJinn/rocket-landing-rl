"""Evaluate a trained policy checkpoint (greedy) against the baseline.

Usage: uv run python scripts/evaluate_policy.py --checkpoint PATH --episodes 100 --seed 0 [--compare-baseline] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict

from evaluate_baseline import crash_reasons

from rocket_landing.baseline import HeuristicController, evaluate
from rocket_landing.environment import EnvConfig
from rocket_landing.policy import PolicyController

ROWS = [
    ("episodes", "n", "{:d}"),
    ("success rate", "success_rate", "{:.1%}"),
    ("crash rate", "crash_rate", "{:.1%}"),
    ("out-of-bounds rate", "out_of_bounds_rate", "{:.1%}"),
    ("timeout rate", "timeout_rate", "{:.1%}"),
    ("mean reward", "mean_reward", "{:.2f}"),
    ("mean fuel used", "mean_fuel_used", "{:.3f}"),
    ("mean steps", "mean_steps", "{:.1f}"),
    ("mean touchdown |vx| (m/s)", "mean_abs_vx", "{:.3f}"),
    ("mean touchdown |vy| (m/s)", "mean_abs_vy", "{:.3f}"),
    ("mean touchdown |angle| (rad)", "mean_landing_tilt", "{:.3f}"),
]


def failure_breakdown(summary, config: EnvConfig) -> Counter:
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
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-version", default="v1")
    ap.add_argument("--compare-baseline", action="store_true")
    ap.add_argument("--json", metavar="PATH", help="write summary and episodes as JSON")
    args = ap.parse_args()

    config = EnvConfig()
    controller = PolicyController.from_checkpoint(args.checkpoint, deterministic=True)
    kw = dict(seed=args.seed, config=config, reward_version=args.reward_version)
    policy = evaluate(controller, args.episodes, **kw)
    base = evaluate(HeuristicController(config), args.episodes, **kw) if args.compare_baseline else None

    print(
        f"Policy {args.checkpoint}  seeds {args.seed}..{args.seed + args.episodes - 1}  "
        f"reward {args.reward_version}"
    )
    if base is None:
        for name, attr, fmt in ROWS:
            print(f"  {name:<30} {fmt.format(getattr(policy, attr)):>10}")
    else:
        print(f"  {'':<30} {'policy':>10} {'baseline':>10}")
        for name, attr, fmt in ROWS:
            print(
                f"  {name:<30} {fmt.format(getattr(policy, attr)):>10} "
                f"{fmt.format(getattr(base, attr)):>10}"
            )
    reasons = failure_breakdown(policy, config)

    if args.json:
        out = {
            "checkpoint": args.checkpoint,
            "summary": {k: v for k, v in asdict(policy).items() if k != "episodes"},
            "crash_reasons": dict(reasons),
            "episodes": [asdict(e) for e in policy.episodes],
        }
        if base is not None:
            out["baseline_summary"] = {
                k: v for k, v in asdict(base).items() if k != "episodes"
            }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
