"""Claim 3: greedy and sampled are different controllers. Success and fuel use vs sampling temperature.

Usage: uv run python analysis/collapse_diagnosis/temperature_sweep.py SPEC [SPEC ...] [--tag NAME] [--episodes 100] [--procs 6]
SPEC is RUN@CKPT, e.g. hl_v2_s8@1200 hl_v2_s8@final hl_v2_s7@1100 hl_v2_s6@final
Temperatures: 0 (greedy), 0.1, 0.25, 0.5, 0.75, 1.0 (logits / T). Held-out seeds 50000.. (diagnostic).
Prints landed count [mean fuel used]; writes raw/temperature_sweep_TAG.json.
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

from _lib import HELDOUT_START, RAW_DIR, NumpyPolicy, make_env, parse_spec, ckpt_path

TEMPS = [0, 0.1, 0.25, 0.5, 0.75, 1.0]


def stats(pol, temp, n, rng):
    r = dict(landed=0, crash_fuel_empty=0, crash_fuel_left=0, oob=0, timeout=0)
    fuel_left, steps, fused = [], [], []
    for s in range(HELDOUT_START, HELDOUT_START + n):
        env = make_env(pol)
        obs, _ = env.reset(seed=s)
        done, t, empty = False, 0, False
        while not done:
            if temp == 0:
                a = int(np.argmax(pol.logits(obs)))
            else:
                p = pol.probs(obs, temp)
                a = min(int(np.searchsorted(np.cumsum(p), rng.random() * p.sum())), 3)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
            t += 1
            if info["state"].fuel <= 0:
                empty = True
        st = info["state"]
        o = info["outcome"]
        steps.append(t)
        fused.append(1 - st.fuel)
        if o == "landed":
            r["landed"] += 1
            fuel_left.append(st.fuel)
        elif o == "crashed":
            r["crash_fuel_empty" if empty else "crash_fuel_left"] += 1
        elif o == "out_of_bounds":
            r["oob"] += 1
        else:
            r["timeout"] += 1
    r["steps"] = float(np.mean(steps))
    r["fuel_used"] = float(np.mean(fused))
    r["fl_p10"] = float(np.percentile(fuel_left, 10)) if fuel_left else None
    return r


def work(args):
    spec, n = args
    run, ck = parse_spec(spec)
    pol = NumpyPolicy(ckpt_path(run, ck))
    rng = np.random.default_rng(11)
    return dict(run=run, upd=ck, res={str(T): stats(pol, T, n, rng) for T in TEMPS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    with Pool(a.procs) as p:
        res = p.map(work, [(s, a.episodes) for s in a.specs])
    RAW_DIR.mkdir(exist_ok=True)
    json.dump(res, open(RAW_DIR / f"temperature_sweep_{a.tag}.json", "w"))
    print(f"landed episodes of {a.episodes} [mean fuel_used] by temperature (T=0 is greedy), seeds {HELDOUT_START}..")
    print("ckpt            " + "  ".join(f"T={T:<9}" for T in TEMPS))
    for x in res:
        print(f"{x['run']}@{str(x['upd']):<6}  " + "  ".join(
            f"{x['res'][str(T)]['landed']:3d}[{x['res'][str(T)]['fuel_used']:.2f}]  " for T in TEMPS))
