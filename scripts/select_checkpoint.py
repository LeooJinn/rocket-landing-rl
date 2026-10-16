"""Pick a robust checkpoint of one training run using a held-out VALIDATION seed range.

    uv run python scripts/select_checkpoint.py --run ~/projects/.../experiments/results/hl_v2_s8 \
        --out-dir experiments/selection/hl_v2_s8 [--rule stable3] [--procs 3]

Evaluates every ``checkpoints/ckpt_*.pt`` plus ``final.pt`` greedily on validation seeds
70000..70199 (asserted disjoint from training, in-training eval 20000-20049, the TEST range
50000..50499 and the baseline seeds), applies the rule, and writes into ``--out-dir``:
``validation_curve.json`` (per-checkpoint outcomes, reused on re-runs), ``selected.json``
(rule, chosen checkpoint, validation numbers, every rule's choice) and ``selected.pt`` (the chosen
checkpoint as an ordinary checkpoint; load it with ``PolicyController.from_checkpoint``).
The run directory is only read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rocket_landing.selection import (
    DEFAULT_RULE,
    RULES,
    VALIDATION_EPISODES,
    VALIDATION_SEED_START,
    select_run,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (or name under --results-dir)")
    ap.add_argument("--out-dir", type=Path, required=True, help="where to write selected.json etc.")
    ap.add_argument("--rule", default=DEFAULT_RULE, choices=sorted(RULES))
    ap.add_argument("--seed-start", type=int, default=VALIDATION_SEED_START)
    ap.add_argument("--episodes", type=int, default=VALIDATION_EPISODES)
    ap.add_argument("--procs", type=int, default=3, help="evaluation worker processes (keep <= 3 on a shared box)")
    ap.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    args = ap.parse_args(argv)

    sel = select_run(args.run, args.out_dir, args.rule, args.seed_start, args.episodes, args.procs,
                     args.results_dir)
    v = sel["validation"]
    print(f"{sel['run']}: validation seeds {v['seed_start']}..{v['seed_end']} ({v['episodes']} episodes), "
          f"{v['n_checkpoints']} checkpoints")
    print(" update  name        success   95% Wilson")
    for c in v["curve"]:
        mark = "  <- chosen" if c["name"] == sel["chosen"]["name"] else ""
        print(f" {c['update']:>6}  {c['name']:<10}  {c['rate']:>6.1%}   "
              f"[{c['ci95'][0]:.1%}, {c['ci95'][1]:.1%}]{mark}")
    ch = sel["chosen"]
    print(f"rule {sel['rule']}: chose {ch['name']} (update {ch['update']}), validation "
          f"{ch['successes']}/{ch['n']}; final.pt validation {sel['final']['successes']}/{sel['final']['n']}")
    print("other rules: " + ", ".join(f"{r}={a['name']}" for r, a in sel["alternatives"].items()))
    print(f"wrote {args.out_dir}/selected.json and selected.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
