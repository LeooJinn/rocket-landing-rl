# Headline results: reward v1 vs v2, fixed-scale normalisation, 10 seeds each

## Setup

- Configs: `experiments/configs/ppo_final_v1.json`, `ppo_final_v2.json`. Identical to `ppo_v2_fixed.json`
  (gamma 0.999, ent_coef 0.03, lr 3e-4 annealed, 16 envs x 256 steps, 5M steps = 1220 updates,
  64x64 tanh actor/critic, `obs_norm: "fixed"`); only `reward_version` differs. `obs_norm: "fixed"` is
  now the code default (`running` stays selectable; `ppo_v1/ppo_v1_tuned/ppo_v2` pin `running`;
  old checkpoints still load, see `tests/test_obs_norm.py`).
- 20 training runs: `hl_v1_s0..s9`, `hl_v2_s0..s9` (train seeds 0-9), `torch_threads=1`, CPU.
  All 20 completed; none crashed or were restarted.
- Evaluation (`scripts/headline_analysis.py eval`, raw: `experiments/headline_eval.json`):
  - Primary: the FINAL checkpoint (`final.pt`, update 1220; never best-so-far) of every run, greedy,
    500 episodes, held-out reset seeds 50000..50499 (disjoint from training seeds >= 1,000,000 and
    in-training eval seeds 20000..20049; asserted by `rocket_landing.evaluate.assert_heldout`).
  - Robustness: the last 5 saved checkpoints of each run (updates 1000, 1050, 1100, 1150, 1200),
    greedy, 200 episodes, seeds 50000..50199 (same seeds as the first 200 of the primary eval).
  - Baselines on the same 500 seeds: heuristic controller and uniform-random policy.
- Statistics (`scripts/headline_analysis.py stats`, `experiments/headline_stats.json`): the unit is the
  training seed (n=10 per arm). 95% Wilson CIs are over the 500 evaluation episodes of one run and
  describe evaluation noise only, not seed-to-seed variation.

## Baselines (500 held-out episodes)

| policy | success | 95% Wilson CI |
|---|---|---|
| heuristic | 99.6% (498/500) | 98.6-99.9% |
| random | 0.0% (0/500) | 0.0-0.8% |

## Per-seed final-checkpoint results

Reward v1:

| seed | final success (500 ep) | 95% Wilson CI | crash/OOB/timeout | last 5 ckpts, success % (200 ep) |
|---|---|---|---|---|
| 0 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 66 / 0 / 14 / 0 |
| 1 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 54 / 100 / 100 / 100 / 100 |
| 2 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 100 / 98 / 60 / 100 / 100 |
| 3 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 0 / 100 / 100 / 54 / 100 |
| 4 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |
| 5 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |
| 6 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |
| 7 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |
| 8 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |
| 9 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |

Reward v2:

| seed | final success (500 ep) | 95% Wilson CI | crash/OOB/timeout | last 5 ckpts, success % (200 ep) |
|---|---|---|---|---|
| 0 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 100 / 70 / 100 / 100 / 100 |
| 1 | 99.8% (499/500) | 98.9-100.0% | 1/0/0 | 100 / 100 / 100 / 92 / 100 |
| 2 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 100 / 100 / 100 / 100 / 100 |
| 3 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 100 / 100 / 100 / 100 / 100 |
| 4 | 70.6% (353/500) | 66.5-74.4% | 147/0/0 | 100 / 100 / 100 / 0 / 50 |
| 5 | 99.8% (499/500) | 98.9-100.0% | 1/0/0 | 94 / 67 / 98 / 96 / 100 |
| 6 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 0 / 0 / 0 / 0 / 0 |
| 7 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 0 / 100 / 0 / 100 / 100 |
| 8 | 0.0% (0/500) | 0.0-0.8% | 500/0/0 | 100 / 100 / 100 / 100 / 0 |
| 9 | 100.0% (500/500) | 99.2-100.0% | 0/0/0 | 0 / 100 / 98 / 94 / 100 |

Failed seeds (final success < 50%): v1 seeds 0, 4, 5, 6, 7, 8, 9 (7/10, all at 0%); v2 seeds 6 and 8
(2/10, both 0%). Every failure is a 100% crash rate (no out-of-bounds or timeouts): the final greedy
policy is a degenerate policy that crashes on every episode, not one that is merely a bit worse.

## Aggregate

| arm | mean | median | min / max | seeds >= 90% | seeds > 50% | seeds at 0% |
|---|---|---|---|---|---|---|
| v1 | 30.0% | 0.0% | 0.0% / 100.0% | 3/10 | 3/10 | 7/10 |
| v2 | 77.0% | 99.9% | 0.0% / 100.0% | 7/10 | 8/10 | 2/10 |

Mean of the last-5-checkpoint means: v1 26.9%, v2 77.1%, so on average the final-checkpoint mean is in line
with its neighbours; but see the next section for per-seed behaviour.

## Is the final checkpoint representative?

Only partly. In v1 4 of 10 seeds and in v2 6 of 10 seeds have a spread (max - min) of more than 20
percentage points across the last five checkpoints (all of those are seeds that learned to land at some
point). Examples: v2 s8 scores 100% on four consecutive checkpoints (1000-1150) then 0% at 1200 and at
the final checkpoint; v2 s7 and v2 s9 alternate between 0% and 100%; v1 s3 goes 0 -> 100 -> 100 -> 54 -> 100.
Seeds that never learned (v1 s4-s9, v2 s6) are 0% throughout. So the final-checkpoint number depends on
where training stopped for seeds that are in the unstable regime; it is an unbiased single draw of "the
checkpoint you would ship if you stopped at 5M steps", but not a stable property of the run. The
headline mean is also noisy for that reason; with only one checkpoint per seed it cannot be separated
from checkpoint-level swings. Note `final.pt` (update 1220) is not among the five scored checkpoints
(the last saved one is update 1200).

## v1 vs v2

Per-seed final success, n=10 per arm:

- Mean difference v2 - v1: +47.0 percentage points (77.0% vs 30.0%).
- Bootstrap 95% CI of that difference (resampling seeds, 20k reps): +7.0 to +80.0 points.
- Exact two-sided permutation test on the difference of means: p = 0.070.
- Exact permutation test on the difference of medians: p = 0.31.
- Seeds >= 90%: 7/10 vs 3/10, Fisher exact two-sided p = 0.18.
- Seeds > 50%: 8/10 vs 3/10, Fisher exact two-sided p = 0.070.

Statement: v2 is better in every summary (mean, median, seeds above 90%, seeds above 50%), but with
10 seeds per arm the evidence is suggestive, not conclusive. The mean-difference permutation test is just
above the conventional 0.05 threshold, the bootstrap interval excludes 0 only barely and is very wide,
and the median and count-based tests are well within noise. Per-seed outcomes are close to bimodal
(0% or ~100%), so means over 10 seeds are highly variable. The experiment was not powered to
detect a difference of this size reliably; do not read it as "v2 is proven better" nor "no difference".
The 500 episodes per seed give tight per-seed CIs; the uncertainty is almost entirely between seeds.

## Figures

- `docs/figures/headline_final_strip.png`: held-out final-checkpoint success per seed (dot, Wilson CI,
  arm mean line, grey ticks = last 5 checkpoints, dashed = heuristic baseline).
- `docs/figures/headline_per_seed_curves.png`: per-seed in-training eval success (50 episodes, seeds
  20000-20049) with the held-out final marked.
- `docs/figures/headline_v1_vs_v2.png`: `scripts/plot_training.py` comparison (in-training eval curves
  with mean, and final vs best-so-far per seed). Its "final" is the last in-training eval on 50 episodes,
  not the held-out number.

## Conclusions

1. With fixed-scale normalisation, v2 reaches >= 99.8% held-out success on 7/10 seeds at the final
   checkpoint (heuristic baseline: 99.6%), one seed is at 70.6% and two are at 0%. v1 reaches ~100% on
   only 3/10 seeds.
2. Fixed normalisation did not remove training instability. 2/10 v2 and 7/10 v1 seeds end in a
   100%-crash policy, and many seeds swing between 0% and 100% across consecutive checkpoints.
3. The earlier normalisation experiment (v2, seeds 0-5: 95% mean, 6/6 seeds >= 70%) is reproduced
   exactly on those seeds (here seeds 0-5 give 100/99.8/100/100/70.6/99.8, mean 95.0%) but seeds 6 and 8
   collapse, so "6/6 seeds fine" overstated reliability; across 10 seeds it is 8/10 above 50%.

## Limitations

- n=10 per arm, p-values are borderline (above). No hyperparameter retuning for v1; v1 may do better
  with its own tuned settings (`ppo_v1_tuned` exists, with running normalisation).
- One training run per seed; checkpoint-to-checkpoint variance is large, so the final-checkpoint number
  is a snapshot (see robustness section). The last-5 scoring uses 200 episodes, not 500.
- Held-out episodes come from the same initial-state distribution as training (no shifted distribution).
- Greedy policy only; the stochastic policy was not evaluated.
- Collapsed policies were not diagnosed (e.g. whether the failure is a policy-entropy collapse or a value
  divergence); only their outcome is reported.
- Wilson CIs treat the 500 episodes as independent for one fixed policy; they say nothing about seed variance.
- Reproduction: `uv run python -m rocket_landing.train --config experiments/configs/ppo_final_v{1,2}.json
  --seed N --run-name hl_v{1,2}_sN`, then `scripts/headline_analysis.py eval|stats|plots`. Checkpoints are
  not committed except `experiments/results/hl_v2_s2/final.pt` (v2 seed 2: 100% final, 100% on all five
  last checkpoints, the most representative run).
