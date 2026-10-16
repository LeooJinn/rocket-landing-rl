"""Rollout detail of a checkpoint: action shares, y / vy / x / angle / fuel every 50 steps, P(MAIN) blocks.

Usage: uv run python analysis/collapse_diagnosis/hover_rollout.py SPEC SEED [SEED ...]
Example (hover-until-empty policy of v2 s6): ... hover_rollout.py hl_v2_s6@final 50000 50001
Runs greedy and sampled (rng seed 0) for each reset seed.
"""

import sys

import numpy as np

from _lib import NumpyPolicy, rollout

spec = sys.argv[1]
seeds = [int(x) for x in sys.argv[2:]]
pol = NumpyPolicy.from_spec(spec)
names = ["NOOP", "MAIN", "LEFT", "RIGHT"]
for s in seeds:
    for mode in ("greedy", "sample"):
        rng = np.random.default_rng(0)
        info, ret, tr = rollout(pol, s, mode, rng, record=True)
        O = np.array([o for o, _, _ in tr])
        A = np.array([a for _, a, _ in tr])
        P = np.array([p for *_, p in tr])
        st = info["state"]
        fe = int(np.argmax(O[:, 6] <= 1e-9)) if (O[:, 6] <= 1e-9).any() else -1
        print(f"--- {spec} seed {s} {mode}: {info['outcome']} steps={info['steps']} ret={ret:.1f} "
              f"fuel_empty_at_step={fe} touchdown x={st.x:.1f} vx={st.vx:.2f} vy={st.vy:.2f} "
              f"ang={st.angle:.2f} om={st.angular_velocity:.2f}")
        print("   action share", {names[i]: round(float((A == i).mean()), 2) for i in range(4)},
              " mean entropy %.2f" % (-(P * np.log(P + 1e-12)).sum(1)).mean())
        idx = list(range(0, len(O), 50))
        for lab, j in (("t:   ", None), ("y:   ", 1), ("vy:  ", 3), ("x:   ", 0), ("ang: ", 4), ("fuel:", 6)):
            print("   " + lab + " ".join((f"{i:6d}" if j is None else f"{O[i, j]:6.2f}") for i in idx))
        print("   P(MAIN) mean by 100-step block: " + " ".join(f"{P[i:i + 100, 1].mean():.2f}" for i in range(0, len(P), 100)))
        print("   first 120 actions:", "".join("NMLR"[a] for a in A[:120]))
