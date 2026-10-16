"""Per-update metrics around a window of a run, with the in-training greedy eval where available.

Usage: uv run python analysis/collapse_diagnosis/collapse_window.py RUN FIRST_UPDATE LAST_UPDATE [STEP]
Examples (the windows quoted in docs/collapse-diagnosis.md):
    ... collapse_window.py hl_v2_s8 1000 1220      # greedy-only collapse, v2 s8
    ... collapse_window.py hl_v2_s6 80 280 5       # settling into hover-until-empty, v2 s6
    ... collapse_window.py hl_v1_s5 590 760 6      # v1 s5: learned, drifted, lost it permanently
Columns: raw episode stats of the SAMPLED training episodes, PPO diagnostics, then (after '|')
greedy eval success / out-of-bounds / crash / mean_reward / mean_fuel_used / mean_steps.
"""

import argparse

from _lib import load

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("first", type=int)
ap.add_argument("last", type=int)
ap.add_argument("step", type=int, nargs="?", default=1)
a = ap.parse_args()
m, e = load(a.run)
ev = {r["update"]: r for r in e}
nan = float("nan")
g = lambda r, k: r[k] if r.get(k) is not None else nan
print("upd  ep_len fuel  succ  crash oob   ep_rew  entropy  kl      clipfrac gradn  vloss   expl_var  p_loss"
      "   lr       | eval: succ oob crash R fuel_used steps")
for r in m[a.first - 1:a.last:a.step]:
    s = (f"{r['update']:4d} {g(r, 'ep_length_mean'):6.0f} {g(r, 'fuel_used_mean'):.2f} "
         f"{g(r, 'success_rate'):.2f} {g(r, 'crash_rate'):.2f} {g(r, 'out_of_bounds_rate'):.2f} "
         f"{g(r, 'ep_reward_mean'):8.1f} {r['entropy']:.3f}  {r['approx_kl']:.4f}  {r['clip_frac']:.3f}   "
         f"{r['grad_norm']:.3f}  {r['value_loss']:.4f}  {r['explained_variance']:.2f}  "
         f"{r['policy_loss']:+.4f}  {r['lr']:.1e}")
    x = ev.get(r["update"])
    if x:
        s += (f" | {x['success_rate']:.2f} {x['out_of_bounds_rate']:.2f} {x['crash_rate']:.2f} "
              f"{x['mean_reward']:.1f} {x['mean_fuel_used']:.3f} {x['mean_steps']:.0f}")
    print(s)
