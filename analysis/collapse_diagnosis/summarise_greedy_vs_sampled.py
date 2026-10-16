"""Aggregate raw/greedy_vs_sampled_*.json over the late checkpoints (1000..1200 + final).

Usage: uv run python analysis/collapse_diagnosis/summarise_greedy_vs_sampled.py   (reads raw/ only)
"""

import glob
import json
import statistics as st

from _lib import RAW_DIR

rows = []
for f in sorted(glob.glob(str(RAW_DIR / "greedy_vs_sampled_hl_v2_s*.json"))):
    n = None
    for r in json.load(open(f)):
        if r["upd"] == "final" or r["upd"] >= 1000:
            n = sum(r["greedy"].values())
            rows.append((r["run"], r["upd"], r["greedy"]["landed"] / n, r["sample"]["landed"] / n))
runs = sorted({r[0] for r in rows})
print("runs:", runs, f"| checkpoints 1000..1200 + final, n = {len(rows)} (episodes per cell: {n})")
print(f"mean greedy {st.mean(r[2] for r in rows):.2f}  mean sampled {st.mean(r[3] for r in rows):.2f}")
print("greedy<0.2 & sampled>=0.8:", sum(1 for r in rows if r[2] < 0.2 and r[3] >= 0.8))
print("greedy<0.5 & sampled>=0.8:", sum(1 for r in rows if r[2] < 0.5 and r[3] >= 0.8))
print("sampled<0.2 & greedy>=0.8:", sum(1 for r in rows if r[3] < 0.2 and r[2] >= 0.8))
print("greedy<0.8:", sum(1 for r in rows if r[2] < 0.8), " sampled<0.8:", sum(1 for r in rows if r[3] < 0.8))
print("per run (min greedy, min sampled, mean greedy, mean sampled):")
for run in runs:
    rr = [r for r in rows if r[0] == run]
    print(f"  {run}: {min(r[2] for r in rr):.2f} {min(r[3] for r in rr):.2f} "
          f"{st.mean(r[2] for r in rr):.2f} {st.mean(r[3] for r in rr):.2f}")
