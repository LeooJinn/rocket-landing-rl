"""In-training greedy eval success per run, as a compact string (claim: collapses happen throughout).

Usage: uv run python analysis/collapse_diagnosis/eval_curves.py

One character per in-training eval (every 10 updates, 50 episodes, seeds 20000..20049):
digit d = success in [d/10, (d+1)/10), X = 100%. Also prints the first update with success >= 90%,
the maximum in-training success, and the number of evals >= 50%.
"""

import numpy as np

from _lib import ALL_RUNS, load

for run in ALL_RUNS:
    _, ev = load(run)
    s = np.array([r["success_rate"] for r in ev])
    trace = "".join("X" if x >= 1 else str(min(9, int(x * 10))) for x in s)
    first = next((r["update"] for r in ev if r["success_rate"] >= 0.9), None)
    print(f"{run}: first>=.9 at upd {first}  max {s.max():.2f} at upd {ev[int(s.argmax())]['update']}"
          f"  evals>=.5: {(s >= 0.5).sum()}  n_eval {len(s)}")
    print("   ", trace)
