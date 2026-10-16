"""Claim 4 (mechanism): greedy vs sampled controller profile by altitude band.

Usage: uv run python analysis/collapse_diagnosis/altitude_profile.py SPEC [SPEC ...] [--episodes 20]
Example: ... altitude_profile.py hl_v2_s8@1200 hl_v2_s3@final hl_v2_s7@1100
For T=0 (greedy) and T=1 (sampled): steps per episode spent in each altitude band, mean vy, and the
share of MAIN / NOOP / LEFT+RIGHT actions. Held-out seeds 50000.. (diagnostic).
"""

import argparse

import numpy as np

from _lib import HELDOUT_START, NumpyPolicy, make_env

BANDS = [(0, 10), (10, 30), (30, 60), (60, 125)]


def prof(spec, temp, n):
    pol = NumpyPolicy.from_spec(spec)
    rng = np.random.default_rng(1)
    vy = {b: [] for b in BANDS}
    act = {b: np.zeros(4) for b in BANDS}
    for s in range(HELDOUT_START, HELDOUT_START + n):
        env = make_env(pol)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            a = int(np.argmax(pol.logits(obs))) if temp == 0 else int(rng.choice(4, p=pol.probs(obs, temp)))
            for b in BANDS:
                if b[0] <= obs[1] < b[1]:
                    vy[b].append(obs[3])
                    act[b][a] += 1
            obs, _, te, tr, _ = env.step(a)
            done = te or tr
    print(f"{spec} T={temp}:")
    for b in BANDS:
        tot = max(act[b].sum(), 1)
        print(f"   y[{b[0]:3d},{b[1]:3d}): steps/ep {act[b].sum() / n:5.0f} mean vy {np.mean(vy[b]):6.2f} "
              f"MAIN {act[b][1] / tot:.2f} NOOP {act[b][0] / tot:.2f} L/R {(act[b][2] + act[b][3]) / tot:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    for spec in a.specs:
        for T in (0, 1.0):
            prof(spec, T, a.episodes)
