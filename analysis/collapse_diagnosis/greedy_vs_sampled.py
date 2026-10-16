"""Claim 2: the collapse is greedy-only. Greedy vs sampled (T=1) success for every saved checkpoint.

Usage: uv run python analysis/collapse_diagnosis/greedy_vs_sampled.py RUN [RUN ...] [--episodes 100] [--procs 10]
Example: ... greedy_vs_sampled.py hl_v2_s8 hl_v2_s7
Needs checkpoints (50..1200 every 50, plus final.pt). Held-out reset seeds 50000.. (diagnostic only).
Writes raw/greedy_vs_sampled_RUN.json and prints one line per checkpoint.
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

from _lib import HELDOUT_START, RAW_DIR, NumpyPolicy, ckpt_path, rollout


def work(args):
    run, upd, n = args
    pol = NumpyPolicy(ckpt_path(run, upd))
    rng = np.random.default_rng(1234)
    out = {"run": run, "upd": upd}
    for mode in ("greedy", "sample"):
        oc = {"landed": 0, "crashed": 0, "out_of_bounds": 0, "timeout": 0}
        for s in range(HELDOUT_START, HELDOUT_START + n):
            info, _, _ = rollout(pol, s, mode, rng)
            oc[info["outcome"]] += 1
        out[mode] = oc
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    ups = list(range(50, 1201, 50)) + ["final"]
    RAW_DIR.mkdir(exist_ok=True)
    for run in a.runs:
        with Pool(a.procs) as p:
            res = p.map(work, [(run, u, a.episodes) for u in ups])
        n = a.episodes
        print(f"{run} upd: greedy_succ / sampled_succ  (n={n}, held-out seeds {HELDOUT_START}..)")
        for r in res:
            g, s = r["greedy"], r["sample"]
            print(f"{str(r['upd']):>6}: greedy {g['landed'] / n:5.2f} ({g['crashed']}c {g['out_of_bounds']}o "
                  f"{g['timeout']}t)   sampled {s['landed'] / n:5.2f} ({s['crashed']}c {s['out_of_bounds']}o "
                  f"{s['timeout']}t)")
        with open(RAW_DIR / f"greedy_vs_sampled_{run}.json", "w") as f:
            json.dump(res, f)
