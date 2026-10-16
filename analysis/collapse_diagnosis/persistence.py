"""How long and how often do greedy collapses last? Predictors of collapse. (reads only eval/metrics)

Usage: uv run python analysis/collapse_diagnosis/persistence.py

A "collapsed" in-training eval has greedy success < 20%. Statistics are over the nine v2 seeds that
learned (v2 s6 never did) plus v1 s1-s3, only AFTER each run's first eval with success >= 90%.
Prints: collapse prevalence by training stage; whether mean_fuel_used of a good eval predicts the
next eval collapsing; lag-k persistence; collapse run lengths; per-run entropy / fuel / collapse
fraction and their Spearman correlations (n = 12, descriptive only).
"""

import warnings

import numpy as np

from _lib import V1_LEARNED, V2_LEARNED, col, load

warnings.filterwarnings("ignore")


def first90(ev):
    return next(i for i, x in enumerate(ev) if x["success_rate"] >= 0.9)


print("A) collapse prevalence by training stage (v2 learners, evals after first >=90% eval)")
for lo, hi in [(0, 800), (800, 900), (900, 1000), (1000, 1100), (1100, 1221)]:
    n = c = 0
    for r in V2_LEARNED:
        _, e = load(r)
        for x in e[first90(e):]:
            if lo < x["update"] <= hi:
                n += 1
                c += x["success_rate"] < 0.2
    lr0, lr1 = 3e-4 * (1 - lo / 1220), 3e-4 * (1 - min(hi, 1220) / 1220)
    print(f"  upd {lo + 1:4d}-{hi:4d} (lr {lr0:.1e}->{lr1:.1e}): n={n:3d} evals, collapsed {c:3d} ({c / max(n, 1):.0%})")

print("\nB) does eval fuel_used predict a collapse at the NEXT eval? (eval i with success>=.9 -> eval i+1)")
rows = []
for r in V2_LEARNED + V1_LEARNED:
    _, e = load(r)
    for i in range(len(e) - 2):
        if e[i]["success_rate"] >= 0.9:
            rows.append((e[i]["mean_fuel_used"], e[i + 1]["success_rate"], e[i + 2]["success_rate"]))
rows = np.array(rows)
for lo, hi in [(0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
    mk = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    if mk.sum():
        print(f"  fuel_used in [{lo:.2f},{hi:.2f}): n={mk.sum():4d}, next eval collapsed(<20%): "
              f"{np.mean(rows[mk, 1] < 0.2):.0%}, next eval <90%: {np.mean(rows[mk, 1] < 0.9):.0%}, "
              f"2nd-next collapsed {np.mean(rows[mk, 2] < 0.2):.0%}")

print("\nC) collapse run lengths (consecutive 10-update evals with success<20%), v2 learners")
for r in V2_LEARNED:
    _, e = load(r)
    runs, k = [], 0
    for x in e[first90(e):]:
        if x["success_rate"] < 0.2:
            k += 1
        else:
            if k:
                runs.append(k)
            k = 0
    print(f"  {r}: collapse run lengths {runs}  (+ {k} still collapsed at end of run)")

print("\nD) persistence of collapse at lag k (v2 learners, pooled)")
for k in [1, 2, 3, 5, 10]:
    a = b = n = base = nb = 0
    for r in V2_LEARNED:
        _, e = load(r)
        c = np.array([x["success_rate"] < 0.2 for x in e[first90(e):]])
        for i in range(len(c) - k):
            nb += 1
            base += c[i + k]
            if c[i]:
                n += 1
                a += c[i + k]
            else:
                b += c[i + k]
    print(f"  lag {k * 10:3d} upd: P(collapsed at t+lag | collapsed at t) = {a / n:.2f} (n={n}); "
          f"P(collapsed at t+lag | OK at t) = {b / (nb - n):.2f}; base rate {base / nb:.2f}")

print("\nE) in-training greedy eval (50 eps) at the saved checkpoints 1000,1050,1100,1150,1200,1220")
for r in V1_LEARNED + V2_LEARNED:
    _, e = load(r)
    d = {x["update"]: x["success_rate"] for x in e}
    print(" ", r, [round(d[u], 2) for u in (1000, 1050, 1100, 1150, 1200, 1220)])

print("\nF) per-run collapse fraction vs entropy / fuel use (descriptive, n=12)")
rank = lambda x: np.argsort(np.argsort(x)).astype(float)
rows = []
for r in V2_LEARNED + V1_LEARNED:
    m, e = load(r)
    i0 = first90(e)
    post = e[i0:]
    u0 = post[0]["update"]
    coll = np.mean([x["success_rate"] < 0.2 for x in post])
    ent = np.nanmean(col(m, "entropy")[u0 - 1:])
    ent_last = np.nanmean(col(m, "entropy")[-300:])
    fu_ok = np.mean([x["mean_fuel_used"] for x in post if x["success_rate"] >= 0.9])
    tr_fuel = np.nanmean(col(m, "fuel_used_mean")[u0 - 1:])
    rows.append((r, coll, ent, ent_last, fu_ok, tr_fuel))
print("run       collapse_frac  mean_entropy(after learn)  entropy(last300)  greedy fuel_used(good evals)  train fuel_used")
for r in rows:
    print(f"{r[0]:9s} {r[1]:.2f}           {r[2]:.2f}                       {r[3]:.2f}              "
          f"{r[4]:.2f}                         {r[5]:.2f}")
A = np.array([r[1:] for r in rows])
sp = lambda M, i, j: np.corrcoef(rank(M[:, i]), rank(M[:, j]))[0, 1]
print(f"Spearman (n={len(rows)}): collapse_frac vs mean entropy after learning {sp(A, 0, 1):+.2f}; "
      f"vs entropy last 300 {sp(A, 0, 2):+.2f}; vs greedy fuel_used {sp(A, 0, 3):+.2f}; "
      f"vs training fuel_used {sp(A, 0, 4):+.2f}")
v2 = A[:len(V2_LEARNED)]
print(f"v2 only (n={len(v2)}): vs entropy {sp(v2, 0, 1):+.2f}, vs greedy fuel {sp(v2, 0, 3):+.2f}, "
      f"vs train fuel {sp(v2, 0, 4):+.2f}")
