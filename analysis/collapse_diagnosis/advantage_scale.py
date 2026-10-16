"""Raw GAE advantage scale of one 16 x 256-step batch collected with a checkpoint's sampled policy.

Usage: uv run python analysis/collapse_diagnosis/advantage_scale.py SPEC [SPEC ...]
Mimics train.py's rollout (16 fresh envs, 256 steps, gamma 0.999, lambda 0.95, reward_scale 0.01) with
the checkpoint's own critic. NOTE: 256 steps from a fresh start usually contains NO episode end for the
slow late policies, so this measures advantage scale on non-terminal segments only; the "episodes ended"
column shows how many ended.
"""

import sys

import numpy as np

from _lib import NumpyPolicy, make_env

N_ENVS, N_STEPS, GAMMA, LAM = 16, 256, 0.999, 0.95
for spec in sys.argv[1:]:
    pol = NumpyPolicy.from_spec(spec)
    rng = np.random.default_rng(0)
    envs = [make_env(pol) for _ in range(N_ENVS)]
    obs = np.stack([e.reset(seed=900000 + i)[0] for i, e in enumerate(envs)])
    V = np.zeros((N_STEPS, N_ENVS))
    Rw = np.zeros_like(V)
    D = np.zeros_like(V)
    for t in range(N_STEPS):
        for i, e in enumerate(envs):
            V[t, i] = pol.value(obs[i])
            a = int(rng.choice(4, p=pol.probs(obs[i])))
            o, r, te, tr, _ = e.step(a)
            Rw[t, i] = r * 0.01
            if te or tr:
                D[t, i] = 1
                o, _ = e.reset()
            obs[i] = o
    nv = np.array([pol.value(obs[i]) for i in range(N_ENVS)])
    adv = np.zeros_like(Rw)
    nxt = np.zeros(N_ENVS)
    for t in reversed(range(N_STEPS)):
        nd = 1 - D[t]
        nxt = Rw[t] + GAMMA * nv * nd - V[t] + GAMMA * LAM * nd * nxt
        adv[t] = nxt
        nv = V[t]
    ret = adv + V
    print(f"{spec:16s} raw GAE adv std {adv.std():.4f} (norm_adv rescales to 1.0) | |adv|>0.05 on "
          f"{np.mean(np.abs(adv) > 0.05):.1%} steps | per-step scaled reward std {Rw.std():.4f} | "
          f"value std {V.std():.3f} | explained var {1 - np.var(ret - V) / np.var(ret):.3f} | "
          f"episodes ended {int(D.sum())}")
