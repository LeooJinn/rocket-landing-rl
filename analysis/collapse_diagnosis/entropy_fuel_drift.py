"""Does training-time fuel use drift together with policy entropy? (reads only metrics.jsonl)

Usage: uv run python analysis/collapse_diagnosis/entropy_fuel_drift.py

For each learned run, from its first update with a 10-update mean training success >= 0.9, take
20-update block means of policy entropy and of the SAMPLED training episodes' fuel_used_mean, and print
the Spearman correlation across blocks, plus the min/max block fuel_used and how the fuel use changed
from the first block to the last. Descriptive only: it does not show that entropy causes the drift.
"""

import warnings

import numpy as np

from _lib import V1_LEARNED, V2_LEARNED, col, load

warnings.filterwarnings("ignore")
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
print("run        blocks  spearman(entropy, fuel_used)  fuel_used first block -> last block  (min..max block)  success last 100 upd")
rhos = []
for run in V2_LEARNED + V1_LEARNED:
    m, _ = load(run)
    ent, fu, sr = col(m, "entropy"), col(m, "fuel_used_mean"), col(m, "success_rate")
    roll = np.array([np.nanmean(sr[i:i + 10]) for i in range(len(sr) - 9)])
    u0 = int(np.argmax(roll >= 0.9))
    blocks = [(np.nanmean(ent[i:i + 20]), np.nanmean(fu[i:i + 20])) for i in range(u0, len(ent) - 19, 20)]
    b = np.array(blocks)
    rho = np.corrcoef(rank(b[:, 0]), rank(b[:, 1]))[0, 1]
    rhos.append(rho)
    print(f"{run:9s} {len(b):5d}   {rho:+.2f}                          {b[0, 1]:.2f} -> {b[-1, 1]:.2f}"
          f"   ({b[:, 1].min():.2f}..{b[:, 1].max():.2f})   {np.nanmean(sr[-100:]):.2f}")
print(f"median Spearman over {len(rhos)} runs: {np.median(rhos):+.2f}; runs with rho > 0: {sum(r > 0 for r in rhos)}/{len(rhos)}")
