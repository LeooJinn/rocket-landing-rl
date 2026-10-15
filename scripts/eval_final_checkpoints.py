"""Evaluate the FINAL checkpoint of each run on held-out seeds (default 40000..40199).

Usage: uv run python scripts/eval_final_checkpoints.py --out experiments/norm_experiment_eval.json \
    LABEL=RUN_DIR [LABEL=RUN_DIR ...]
Greedy policy, reward_version v2, no training/in-training-eval seeds are used.
"""

from __future__ import annotations

import argparse
import json

from rocket_landing.baseline import evaluate
from rocket_landing.policy import PolicyController


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--reward-version", default="v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = {}
    for item in args.runs:
        label, run = item.split("=", 1)
        ctrl = PolicyController.from_checkpoint(f"{run}/final.pt", deterministic=True)
        s = evaluate(ctrl, args.episodes, seed=args.seed, reward_version=args.reward_version)
        results[label] = {
            "run": run,
            "success_rate": s.success_rate,
            "crash_rate": s.crash_rate,
            "out_of_bounds_rate": s.out_of_bounds_rate,
            "timeout_rate": s.timeout_rate,
            "mean_reward": s.mean_reward,
            "episodes": s.n,
            "seeds": [args.seed, args.seed + args.episodes - 1],
        }
        print(label, f"{s.success_rate:.3f}", f"R {s.mean_reward:.1f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
