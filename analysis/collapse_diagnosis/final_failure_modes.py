"""Claim 1 (rollouts): how each run's final.pt fails. Fuel-out step and altitude at fuel-out.

Usage: uv run python analysis/collapse_diagnosis/final_failure_modes.py [--episodes 20] [--procs 5]
final.pt, greedy, held-out seeds 50000.. (diagnostic). Needs final.pt of all 20 runs.
Type 1 failure = fuel runs out within ~5 m of the ground (slow glide); type 2 = fuel runs out at
altitude (hover until empty).
"""

import argparse
from multiprocessing import Pool

import numpy as np

from _lib import ALL_RUNS, HELDOUT_START, NumpyPolicy, ckpt_path, make_env


def work(args):
    run, n = args
    pol = NumpyPolicy(ckpt_path(run, "final"))
    oc, y_e, t_e, vy_t = [], [], [], []
    for s in range(HELDOUT_START, HELDOUT_START + n):
        env = make_env(pol)
        obs, _ = env.reset(seed=s)
        done, t, e = False, 0, None
        while not done:
            obs, _, te, tr, info = env.step(int(np.argmax(pol.logits(obs))))
            done = te or tr
            t += 1
            if e is None and info["state"].fuel <= 0:
                e = (t, info["state"].y)
        oc.append(info["outcome"])
        vy_t.append(info["state"].vy)
        if e:
            t_e.append(e[0])
            y_e.append(e[1])
    return (run, oc.count("landed"), oc.count("crashed"), oc.count("out_of_bounds"), len(t_e),
            np.mean(t_e) if t_e else np.nan, np.mean(y_e) if y_e else np.nan,
            np.median(y_e) if y_e else np.nan, np.mean(vy_t))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    with Pool(a.procs) as p:
        res = p.map(work, [(r, a.episodes) for r in ALL_RUNS])
    print(f"final.pt greedy, {a.episodes} eps (seeds {HELDOUT_START}..): landed/crashed/oob | episodes where "
          "fuel ran out | mean step fuel ran out | altitude (m) when fuel ran out: mean, median | mean touchdown vy")
    for r in res:
        print(f"{r[0]:8s} {r[1]:2d}/{r[2]:2d}/{r[3]:2d} | out-of-fuel eps {r[4]:2d} | at step {r[5]:5.0f} | "
              f"y {r[6]:5.1f} med {r[7]:5.1f} | vy {r[8]:6.1f}")
