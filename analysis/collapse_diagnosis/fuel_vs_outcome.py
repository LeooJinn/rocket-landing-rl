"""Claim 1: failures are fuel exhaustion. Relates in-training greedy eval outcome to fuel used.

Usage: uv run python analysis/collapse_diagnosis/fuel_vs_outcome.py      (reads only eval.jsonl)

Pools all 2440 in-training evals (20 runs x 122 evals, 50 episodes each, seeds 20000..20049) and
prints mean_fuel_used for success >= 90%, crash >= 90% (overall, after update 600, split by whether
the run ever reached 90%), the share of crash-dominated evals with fuel_used >= 0.99, greedy success
by fuel_used bin, and per learned run the collapsed evals (success < 20%) after its first >= 90% eval.
"""

import warnings

import numpy as np

from _lib import ALL_RUNS, load

warnings.filterwarnings("ignore")
rows = []
for run in ALL_RUNS:
    _, ev = load(run)
    best = max(r["success_rate"] for r in ev)
    for r in ev:
        rows.append((run, r["update"], r["success_rate"], r["crash_rate"], r["out_of_bounds_rate"],
                     r["timeout_rate"], r["mean_fuel_used"], r["mean_steps"], best))
A = np.array([r[1:] for r in rows], float)
upd, succ, crash, oob, to, fu, st, best = A.T
learned = best >= 0.9
print("total evals", len(A))


def summ(name, mask):
    if mask.sum() == 0:
        print(f"{name:62s} n=0")
        return
    print(f"{name:62s} n={mask.sum():4d} fuel_used mean {fu[mask].mean():.3f} "
          f"p10 {np.percentile(fu[mask], 10):.3f} p50 {np.percentile(fu[mask], 50):.3f} | "
          f"steps mean {st[mask].mean():.0f}")


summ("all evals, success>=0.9", succ >= 0.9)
summ("all evals, crash>=0.9", crash >= 0.9)
summ("all evals, oob>=0.9", oob >= 0.9)
summ("upd>=600, crash>=0.9 (any run)", (crash >= 0.9) & (upd >= 600))
summ("upd>=600, crash>=0.9, run reached >=90% at some point", (crash >= 0.9) & (upd >= 600) & learned)
summ("upd>=600, crash>=0.9, run NEVER reached 90%", (crash >= 0.9) & (upd >= 600) & ~learned)
summ("upd>=600, 0.1<succ<0.9", (succ > 0.1) & (succ < 0.9) & (upd >= 600))
mk = (crash >= 0.9) & (upd >= 600)
print(f"\nupd>=600 crash>=0.9 evals: fuel_used>=0.99: {(fu[mk] >= 0.99).sum()} / {mk.sum()}; "
      f"0.95-0.99: {((fu[mk] >= 0.95) & (fu[mk] < 0.99)).sum()}; <0.95: {(fu[mk] < 0.95).sum()}")
print("\nupd>=600 evals pooled: greedy success vs mean_fuel_used bin (excluding oob>=0.5)")
bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
for lo, hi in zip(bins[:-1], bins[1:]):
    mk = (fu >= lo) & (fu < hi) & (oob < 0.5) & (upd >= 600)
    if mk.sum():
        print(f"  fuel_used [{lo:.2f},{hi:.2f}): n={mk.sum():4d}  mean greedy success {succ[mk].mean():.2f}")
print("\nPer run: after the first >=90% eval, evals with success<20% and how many of them have fuel_used>=0.97")
for run in ALL_RUNS:
    _, ev = load(run)
    i0 = next((i for i, r in enumerate(ev) if r["success_rate"] >= 0.9), None)
    if i0 is None:
        continue
    post = ev[i0:]
    c = [r for r in post if r["success_rate"] < 0.2]
    print(f"  {run}: first>=.9 @ {ev[i0]['update']}, evals after: {len(post)}, collapsed(<20%): {len(c)} "
          f"({len(c) / len(post):.0%}), of which fuel_used>=0.97: {sum(r['mean_fuel_used'] >= 0.97 for r in c)}")
