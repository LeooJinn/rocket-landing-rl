"""Claim 4: the greedy argmax is nearly tied. Action-probability margins and checkpoint-to-checkpoint change.

Usage: uv run python analysis/collapse_diagnosis/argmax_margins.py
Part A: along each checkpoint's own greedy trajectories (20 episodes, seeds 50000..): mean max prob,
fraction of steps with top1-top2 margin < 0.1 / 0.2, entropy, mean P(MAIN), MAIN share of fuel burned.
Part B: on the states visited by the EARLIER checkpoint's greedy runs, how often does the argmax differ
in the later checkpoint, and KL / P(MAIN) change.
"""

import numpy as np

from _lib import HELDOUT_START, NumpyPolicy, make_env, parse_spec

SPECS = ["hl_v2_s8@1000", "hl_v2_s8@1150", "hl_v2_s8@1200", "hl_v2_s7@1050", "hl_v2_s7@1100",
         "hl_v2_s3@final", "hl_v2_s2@final", "hl_v2_s1@final"]
PAIRS = [("hl_v2_s8@1150", "hl_v2_s8@1200"), ("hl_v2_s8@1100", "hl_v2_s8@1150"),
         ("hl_v2_s8@1000", "hl_v2_s8@1050"), ("hl_v2_s7@1050", "hl_v2_s7@1100"),
         ("hl_v2_s7@1100", "hl_v2_s7@1150")]


def greedy_states(pol, n=20):
    S = []
    for s in range(HELDOUT_START, HELDOUT_START + n):
        env = make_env(pol)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            S.append(obs.copy())
            obs, _, te, tr, _ = env.step(int(np.argmax(pol.logits(obs))))
            done = te or tr
    return np.array(S)


print("A) action-probability margin along each checkpoint's own greedy trajectories (20 eps)")
print("ckpt           steps/ep  mean_maxP  top1-top2<0.1  top1-top2<0.2  mean_entropy  P(MAIN) mean  MAIN share of fuel")
for spec in SPECS:
    pol = NumpyPolicy.from_spec(spec)
    S = greedy_states(pol)
    P = np.array([pol.probs(o) for o in S])
    srt = np.sort(P, 1)
    mar = srt[:, -1] - srt[:, -2]
    ent = -(P * np.log(P + 1e-12)).sum(1)
    A = P.argmax(1)
    mf, sf = (A == 1).sum() * 0.0025, ((A == 2) | (A == 3)).sum() * 0.0005
    print(f"{spec:15s} {len(S) / 20:6.0f}    {P.max(1).mean():.2f}      {np.mean(mar < 0.1):.2f}          "
          f"{np.mean(mar < 0.2):.2f}         {ent.mean():.2f}          {P[:, 1].mean():.2f}        {mf / (mf + sf):.2f}")

print("\nB) policy change between neighbouring checkpoints, on states visited by the earlier checkpoint's greedy runs")
for a, b in PAIRS:
    pa, pb = NumpyPolicy.from_spec(a), NumpyPolicy.from_spec(b)
    S = greedy_states(pa)
    Pa = np.array([pa.probs(o) for o in S])
    Pb = np.array([pb.probs(o) for o in S])
    kl = (Pa * (np.log(Pa + 1e-12) - np.log(Pb + 1e-12))).sum(1)
    print(f"{a} -> {b}: states {len(S)}, argmax changed in {np.mean(Pa.argmax(1) != Pb.argmax(1)):.1%}, "
          f"mean KL(a||b) {kl.mean():.4f}, mean |dP(MAIN)| {np.abs(Pa[:, 1] - Pb[:, 1]).mean():.3f}, "
          f"mean P(MAIN) {Pa[:, 1].mean():.3f} -> {Pb[:, 1].mean():.3f}")
