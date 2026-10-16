"""Raw vs discounted (gamma=0.999, reward_scale 0.01) return of sampled episodes of checkpoints.

Usage: uv run python analysis/collapse_diagnosis/discounted_returns.py SPEC [SPEC ...]   (40 episodes, seeds 60000..)
Example: ... discounted_returns.py hl_v2_s6@100 hl_v2_s6@final hl_v2_s3@final hl_v2_s8@final
Reset seeds 60000.. are used here (not the headline held-out range); this is a diagnostic only.
"""

import sys

import numpy as np

from _lib import NumpyPolicy, make_env

GAMMA, SCALE, N, START = 0.999, 0.01, 40, 60000
for spec in sys.argv[1:]:
    pol = NumpyPolicy.from_spec(spec)
    rng = np.random.default_rng(3)
    out = []
    for s in range(START, START + N):
        env = make_env(pol)
        obs, _ = env.reset(seed=s)
        done, R, D, t = False, 0.0, 0.0, 0
        while not done:
            obs, r, te, tr, info = env.step(int(rng.choice(4, p=pol.probs(obs))))
            R += r
            D += GAMMA ** t * r * SCALE
            t += 1
            done = te or tr
        out.append((R, D, t, info["outcome"]))
    print(f"{spec:16s} raw return {np.mean([x[0] for x in out]):7.1f} | discounted scaled return "
          f"(gamma={GAMMA}) {np.mean([x[1] for x in out]):+.3f} | ep len {np.mean([x[2] for x in out]):5.0f} | "
          f"landed {sum(x[3] == 'landed' for x in out)}/{len(out)}")
