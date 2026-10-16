"""Training (sampled) episodes vs in-training greedy eval, entropy, and fuel/length per run.

Usage: uv run python analysis/collapse_diagnosis/training_vs_greedy.py

Part 1: entropy at updates 100/300/600/900 (+-10 updates) and last 20 updates, training success over
the last 200 updates vs greedy eval success over the last 20 evals, and the number of evals with
greedy < 20% while the training success in the same 10-update window is > 80%.
Part 2: training (sampled) fuel_used_mean and ep_length_mean at updates 300/600/900/last 100.
"""

import warnings

import numpy as np

from _lib import ALL_RUNS, col, load

warnings.filterwarnings("ignore")
print("PART 1")
print("run    ent@100 ent@300 ent@600 ent@900 ent@end | trainSucc(last200upd) greedyEval(last20 evals) | "
      "n evals with greedy<.2 while train>.8")
for run in ALL_RUNS:
    m, e = load(run)
    ent, ts = col(m, "entropy"), col(m, "success_rate")
    w = lambda u: ent[u - 10:u + 10].mean()
    ev = np.array([r["success_rate"] for r in e])
    tr = np.array([np.nanmean(ts[max(0, r["update"] - 10):r["update"]]) for r in e])
    n_such = int(np.sum((ev < 0.2) & (tr > 0.8)))
    print(f"{run} {w(100):.2f} {w(300):.2f} {w(600):.2f} {w(900):.2f} {ent[-20:].mean():.2f} | "
          f"train {np.nanmean(tr[-20:]):.2f} greedy {ev[-20:].mean():.2f} | n={n_such}")

print("\nPART 2: training (sampled) episodes: fuel_used_mean and ep_length_mean")
print("run    | fuel @300 @600 @900 last100 | len @300 @600 @900 last100 | succ last100")
for run in ALL_RUNS:
    m, _ = load(run)
    fu, ln, sr = col(m, "fuel_used_mean"), col(m, "ep_length_mean"), col(m, "success_rate")
    w = lambda a, u: np.nanmean(a[u - 20:u + 20])
    print(f"{run} | {w(fu, 300):.2f} {w(fu, 600):.2f} {w(fu, 900):.2f} {np.nanmean(fu[-100:]):.2f} | "
          f"{w(ln, 300):4.0f} {w(ln, 600):4.0f} {w(ln, 900):4.0f} {np.nanmean(ln[-100:]):4.0f} | "
          f"{np.nanmean(sr[-100:]):.2f}")
