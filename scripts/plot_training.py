"""Make training/evaluation figures from run directories.

Per-run figures and one comparison figure (grouped by config keys). Examples:

    uv sync --extra plots
    uv run python scripts/plot_training.py --results ~/projects/rocket-landing-rl/experiments/results \
        --run v2_s1 --run sweep_g999e03_s1
    uv run python scripts/plot_training.py --results DIR --compare 'v1=sweep_g999e03_s*' \
        --compare 'v2=v2_s*' --group-by reward_version --out docs/figures/v1_vs_v2.png

Read-only on the results directory; safe on runs still in progress.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from rocket_landing.plotting import (  # noqa: E402
    final_table,
    group_runs,
    load_run,
    load_runs,
    plot_comparison,
    plot_run,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", required=True, help="directory containing run directories")
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--run", action="append", default=[], help="run name; one per-run figure each")
    ap.add_argument(
        "--compare", action="append", default=[],
        help="glob of run names (relative to --results) to include in the comparison; repeatable",
    )
    ap.add_argument("--group-by", nargs="+", default=["reward_version"],
                    help="flattened config keys defining groups, e.g. reward_version ppo.gamma")
    ap.add_argument("--out", help="comparison output file (default: <out-dir>/comparison.png)")
    ap.add_argument("--title")
    args = ap.parse_args()

    results = Path(args.results).expanduser()
    out_dir = Path(args.out_dir)
    for name in args.run:
        path = plot_run(load_run(results / name), out_dir / f"run_{name}.png")
        print("wrote", path)
    if args.compare:
        seen, runs = set(), []
        for pat in args.compare:
            for r in load_runs(results, pat):
                if r.name not in seen:
                    seen.add(r.name)
                    runs.append(r)
        groups = group_runs(runs, args.group_by)
        out = Path(args.out) if args.out else out_dir / "comparison.png"
        print("wrote", plot_comparison(groups, out, title=args.title))
        for row in final_table(groups):
            fin, best = row["final_success"], row["best_success"]
            print(
                f"{row['group']:<24} {row['run']:<20} seed={row['seed']} "
                f"final={fin if fin is None else f'{fin:.2f}'} "
                f"best-so-far={best if best is None else f'{best:.2f}'} "
                f"{'(in progress)' if row['in_progress'] else ''}"
            )


if __name__ == "__main__":
    main()
