"""One seed, three controllers: the heuristic, a good greedy checkpoint, a collapsed one.

Usage: uv run python analysis/collapse_diagnosis/fuel_race_trace.py [RUN] [SEED] [CKPT_GOOD] [CKPT_BAD]
Default: hl_v2_s8 50003 1150 1200. Prints y, vy, fuel every 50 steps, outcome and fuel left.
"""

import sys

import numpy as np

from _lib import NumpyPolicy, ckpt_path, make_env

from rocket_landing.baseline import HeuristicController
from rocket_landing.environment import RocketLandingEnv

run = sys.argv[1] if len(sys.argv) > 1 else "hl_v2_s8"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 50003
good = int(sys.argv[3]) if len(sys.argv) > 3 else 1150
bad = int(sys.argv[4]) if len(sys.argv) > 4 else 1200


def prof(label, act_fn, env):
    obs, _ = env.reset(seed=seed)
    rows, done, t = [], False, 0
    while not done:
        a = act_fn(obs)
        if t % 50 == 0:
            rows.append((t, obs[0], obs[1], obs[3], obs[4], obs[6], a))
        obs, _, te, tr, info = env.step(a)
        done = te or tr
        t += 1
    s = info["state"]
    print(f"{label}: {info['outcome']} steps={t} fuel_left={s.fuel:.3f} touchdown vy={s.vy:.2f}")
    print("   t:   " + " ".join(f"{r[0]:6d}" for r in rows))
    print("   y:   " + " ".join(f"{r[2]:6.1f}" for r in rows))
    print("   vy:  " + " ".join(f"{r[3]:6.2f}" for r in rows))
    print("   fuel:" + " ".join(f"{r[5]:6.2f}" for r in rows))


prof("heuristic", HeuristicController().act, RocketLandingEnv(reward_version="v2"))
for upd in (good, bad):
    pol = NumpyPolicy(ckpt_path(run, upd))
    prof(f"{run}@{upd} greedy", lambda o: int(np.argmax(pol.logits(o))), make_env(pol))
