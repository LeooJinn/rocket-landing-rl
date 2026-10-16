# Reward v3 results: does pricing time and fuel remove the fuel-exhaustion failures?

Reward definition, numbers and exploit checks: `docs/reward-design.md` (section Reward v3).
Everything below was run in the worktree `rocket-landing-rl-rv3` (branch `reward-v3`, from `headline`).

## Summary (plain)

- Hypothesis: the headline failures (greedy-only late collapses, never-learning seeds) are fuel
  exhaustion before touchdown, and reward v2 does not price the slow, fuel-hungry mode. v3a adds a
  landing bonus of `40 * fuel_left` and triples the time penalty; v3b adds an extra -25 for crashing
  with an empty tank.
- Pilot (seeds 10-12, not reused later; 2 variants plus a v2 control): v3a was picked. n=3, so this
  only separates a clearly bad variant (v3b had one greedy-only collapse) from a good one.
- Confirmation (v3a, seeds 0-9, 500 held-out episodes each, final checkpoint): greedy success was
  500/500 on all 10 seeds (5000/5000), versus 7 of 10 seeds at >= 70% and two at 0% for v2
  (`hl_v2`, same seeds, same config except `reward_version`). The paired greedy-success difference
  is +23.0 points, bootstrap 95% CI [+0.1, +50.0], exact sign-flip p = 0.062 (5 seeds better, 0 worse,
  5 tied at ~100%): **suggestive, not significant at 0.05 with n=10.**
- Efficiency improved clearly and consistently: on seeds where both arms land, greedy fuel used
  0.53 vs 0.71 and episode length 362 vs 486 steps (7 of 8 seeds lower, sign-flip p = 0.031, n=8).
- Sampled-action success is NOT clearly better: 95.1% vs 82.9% mean, but 6 seeds better / 4 worse,
  p = 0.35, within noise (v2's mean is dragged down by seed 6 at 0%). In v3a the sampled policy is
  slightly worse than the greedy one (4.9% crashes), the reverse of the v2 collapse mode.
- Not fully fixed: four v3a seeds still have one of their last five saved checkpoints at 60-74%
  greedy success (but none below 50%, versus 10 of 50 last-5 checkpoints below 50% in v2).

## What was run

All runs: `ppo_final_v2.json` with only `reward_version` changed (gamma 0.999, ent_coef 0.03,
lr 3e-4 annealed, 16 envs x 256 steps, 5M steps, 64x64 tanh, fixed obs scales, no schedules),
`torch_threads=1`, CPU, queued by `scripts/run_queue.py` (max 3 processes; no run died or was
restarted). Evaluation: `scripts/rv3_analysis.py` (greedy and sampled actions on the same episodes,
`rocket_landing.evaluate.assert_heldout` asserted for every run).

Number of variants tried: **2** (v3a, v3b), with weights fixed in advance from arithmetic on a
scripted heuristic ladder (`scripts/reward_scenarios.py`) before any training; no weight was tuned
on training results, and no other v3 weights were run.

### Pilot (selection; seeds 10, 11, 12)

9 runs `rv3_pilot_{v2,v3a,v3b}_s{10,11,12}`. Final checkpoint, 200 held-out episodes (seeds
50000-50199), plus greedy success of the last 5 saved checkpoints (200 episodes).

Selection rule, decided before the first pilot evaluation was run (written down here afterwards, not
committed beforehand): highest mean greedy success over
the 3 seeds; if the two best are within 0.05, lower mean fuel used wins; if still tied, prefer v3a
(fewer terms).

| variant | seed | greedy succ | sampled succ | fuel (all eps) | steps (all eps) | last-5 ckpt greedy |
|---|---|---|---|---|---|---|
| v2 control | 10 | 0.785 | 0.970 | 0.647 | 458 | 0.21 / 0.79 / 0.96 / 0.07 / 0.92 |
| v2 control | 11 | 0.955 | 0.960 | 0.451 | 320 | 0.90 / 0.89 / 0.03 / 0.94 / 0.96 |
| v2 control | 12 | 1.000 | 0.980 | 0.800 | 566 | 0.88 / 1.0 / 1.0 / 1.0 / 1.0 |
| v3a | 10 | 1.000 | 0.910 | 0.484 | 339 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 |
| v3a | 11 | 1.000 | 0.985 | 0.687 | 468 | 1.0 / 1.0 / 0.0 / 1.0 / 1.0 |
| v3a | 12 | 1.000 | 0.940 | 0.494 | 338 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 |
| v3b | 10 | 1.000 | 0.945 | 0.526 | 373 | 0.77 / 1.0 / 1.0 / 1.0 / 1.0 |
| v3b | 11 | **0.020** | 0.990 | 0.988 | 726 | 0.96 / 0.26 / 0.26 / 0.38 / 0.03 |
| v3b | 12 | 1.000 | 0.975 | 0.659 | 452 | 0.97 / 1.0 / 1.0 / 1.0 / 1.0 |

| variant | mean greedy succ | mean sampled succ | mean fuel (all eps) | mean steps (all eps) |
|---|---|---|---|---|
| v2 control | 0.913 | 0.970 | 0.633 | 448 |
| v3a | **1.000** | 0.945 | **0.555** | **382** |
| v3b | 0.673 | 0.970 | 0.724 | 517 |

v3a wins on both criteria, so it was the only variant confirmed. Reading: with n=3 this is weak
evidence; v3b's failure on seed 11 (greedy 2%, sampled 99%) is the greedy-only collapse again, which
the extra fuel-out penalty did not prevent; v3a's own seed 11 also shows a transient 0.0 checkpoint.
Raw: `experiments/rv3_pilot_eval.json`.

### Confirmation (v3a vs hl_v2, seeds 0-9)

10 runs `rv3_v3a_s0..s9` (train seeds 0-9, identical network initialisation and training-env seeds
as `hl_v2_s0..s9`; pairing is by seed identity, but trajectories diverge because the rewards
differ). Final checkpoint, 500 held-out episodes (seeds 50000-50499), greedy and sampled actions;
last 5 saved checkpoints (updates 1000-1200), greedy, 200 episodes. The `hl_v2` runs were
re-evaluated with the same script (checkpoints from the headline worktree, read-only): the greedy
counts reproduce `experiments/headline_eval.json` exactly (e.g. 353/500 for seed 4, 0/500 for
seeds 6 and 8). Raw: `experiments/rv3_confirm_eval.json`, `experiments/rv3_hlv2_eval.json`,
stats `experiments/rv3_confirm_stats.json`.

Per seed (fuel is the fraction of the tank, steps are episode length, |vy| is the mean touchdown
descent speed of landed episodes; the safe limit is 2 m/s):

| seed | v3a greedy | v3a sampled | v3a fuel / steps / abs vy (greedy) | v3a last-5 greedy % | v2 greedy | v2 sampled | v2 fuel / steps (greedy) | v2 last-5 greedy % |
|---|---|---|---|---|---|---|---|---|
| 0 | 100.0% (500/500) | 89.2% (446/500) | 0.412 / 285 / 0.83 | 100 / 100 / 100 / 100 / 100 | 100.0% (500/500) | 95.8% (479/500) | 0.665 / 439 | 100 / 70 / 100 / 100 / 100 |
| 1 | 100.0% (500/500) | 91.2% (456/500) | 0.545 / 375 / 0.81 | 100 / 100 / 100 / 100 / 100 | 99.8% (499/500) | 98.0% (490/500) | 0.561 / 374 | 100 / 100 / 100 / 92 / 100 |
| 2 | 100.0% (500/500) | 98.0% (490/500) | 0.437 / 311 / 0.64 | 98 / 100 / 100 / 100 / 100 | 100.0% (500/500) | 93.8% (469/500) | 0.614 / 423 | 100 / 100 / 100 / 100 / 100 |
| 3 | 100.0% (500/500) | 97.0% (485/500) | 0.532 / 364 / 0.59 | 100 / 100 / 74 / 100 / 100 | 100.0% (500/500) | 91.8% (459/500) | 0.456 / 321 | 100 / 100 / 100 / 100 / 100 |
| 4 | 100.0% (500/500) | 98.8% (494/500) | 0.665 / 448 / 0.67 | 100 / 100 / 100 / 100 / 100 | 70.6% (353/500) | 96.8% (484/500) | 0.996 / 677 | 100 / 100 / 100 / 0 / 50 |
| 5 | 100.0% (500/500) | 93.2% (466/500) | 0.565 / 395 / 0.69 | 72 / 100 / 100 / 100 / 100 | 99.8% (499/500) | 58.4% (292/500) | 0.633 / 454 | 94 / 67 / 98 / 96 / 100 |
| 6 | 100.0% (500/500) | 93.4% (467/500) | 0.414 / 292 / 0.73 | 100 / 100 / 100 / 100 / 100 | 0.0% (0/500) | 0.0% (0/500) | 1.000 / 858 | 0 / 0 / 0 / 0 / 0 |
| 7 | 100.0% (500/500) | 95.6% (478/500) | 0.533 / 358 / 0.77 | 100 / 100 / 100 / 100 / 100 | 100.0% (500/500) | 99.2% (496/500) | 0.902 / 601 | 0 / 100 / 0 / 100 / 100 |
| 8 | 100.0% (500/500) | 98.2% (491/500) | 0.797 / 542 / 0.83 | 100 / 66 / 100 / 100 / 100 | 0.0% (0/500) | 97.8% (489/500) | 1.000 / 704 | 100 / 100 / 100 / 100 / 0 |
| 9 | 100.0% (500/500) | 96.0% (480/500) | 0.541 / 362 / 0.78 | 100 / 100 / 60 / 100 / 100 | 100.0% (500/500) | 97.2% (486/500) | 0.871 / 597 | 0 / 100 / 98 / 94 / 100 |

Paired comparison, v3a minus v2, unit = training seed, n=10 (n=8 for landed-episode metrics, since
v2 seeds 6 and 8 have no greedy landing). Bootstrap = percentile 95% CI of the mean paired
difference (20000 resamples); p = exact two-sided sign-flip test of mean difference 0 (all 2^n sign
patterns; tied seeds contribute 0). "higher/lower/tied" counts seeds where v3a's value is higher /
lower / equal.

| metric | n | v3a mean | v2 mean | mean diff | bootstrap 95% CI | p | v3a higher / lower / tied |
|---|---|---|---|---|---|---|---|
| greedy success | 10 | 1.000 | 0.770 | +0.230 | [+0.001, +0.500] | 0.062 | 5 / 0 / 5 |
| sampled success | 10 | 0.951 | 0.829 | +0.122 | [-0.018, +0.324] | 0.350 | 6 / 4 / 0 |
| greedy fuel used, all episodes | 10 | 0.544 | 0.770 | -0.226 | [-0.341, -0.116] | 0.008 | 1 / 9 / 0 |
| greedy episode length, all episodes | 10 | 373 | 545 | -172 | [-279, -82] | 0.008 | 2 / 8 / 0 |
| greedy fuel used, landed episodes | 8 | 0.529 | 0.712 | -0.184 | [-0.286, -0.074] | 0.031 | 1 / 7 / 0 |
| greedy episode length, landed episodes | 8 | 362 | 486 | -123 | [-193, -50] | 0.031 | 2 / 6 / 0 |
| greedy touchdown abs vy, landed episodes | 8 | 0.723 | 0.757 | -0.034 | [-0.182, +0.095] | 0.703 | 4 / 4 / 0 |
| sampled fuel used, all episodes | 10 | 0.389 | 0.618 | -0.228 | [-0.371, -0.103] | 0.008 | 1 / 9 / 0 |
| sampled episode length, all episodes | 10 | 265 | 441 | -176 | [-304, -72] | 0.008 | 1 / 9 / 0 |

Other facts from the same data:

- Greedy outcomes (mean over seeds): v3a 100% landed, 0% crash/out of bounds/timeout; v2 77.0% landed,
  23.0% crashed (all failures were crashes; no timeouts and no out-of-bounds).
- Sampled outcomes: v3a 95.1% landed, 4.9% crashed; v2 82.9% landed, 17.1% crashed.
- Last 5 checkpoints (50 checkpoint evaluations per arm, greedy, 200 episodes): v3a 46 of 50 have
  >= 90% success and 0 are below 50%; v2 37 of 50 are >= 90% and 10 are below 50%. v3a seeds with a
  weak late checkpoint: 3 (74%), 5 (72%), 8 (66%), 9 (60%); the final checkpoints of those seeds are
  all 100%, so final-checkpoint reporting still hides some checkpoint-to-checkpoint wobble.
- Heuristic controller on seeds 50000-50499 (`experiments/headline_eval.json`): 99.6% landed, fuel 0.365,
  270 steps. v3a greedy lands in a median of 363 steps using 0.54 of the tank (v2 on its seven healthy
  seeds, those with >= 90% greedy success: 458 steps, 0.67; v3a on the same seven seeds: 350, 0.51).
  v3a's sampled policy is closer to the heuristic: 265 steps, 0.39 of the tank.
- Touchdown speed is unchanged and safe: greedy touchdown |vy| about 0.6-0.8 m/s for v3a (limit 2),
  same as v2 (0.76).

## How to read this

- The efficiency claim is the solid one: v3a uses less fuel in 9 of 10 seeds and is faster in 8 of 10
  (all episodes; 7 and 6 of 8 on landed episodes only), and the same shift shows in the sampled
  policy. This is the direction the reward arithmetic predicts (a fast frugal landing scores 26 points
  more than a 547-step one and more than 40 points more than a 700-step one).
- The robustness claim is promising but not established: 10/10 seeds at the maximum, against 7/10
  healthy for v2, is the pattern expected if fuel exhaustion drove the v2 collapses, but 5 of the 10
  paired seeds are tied at the ceiling, so the test has only five informative pairs and cannot get
  below p = 0.0625. A larger seed set (or a harder stress test) would be needed to call it.
- Sampled success was not shown to improve (p = 0.35); v3a trades a perfect greedy policy for a
  sampled policy with 4.9% crashes. The training entropy coefficient (0.03, held for the whole run)
  keeps the sampled policy noisy (training policy entropy ends at 0.92-0.97 of a possible 1.39 in the runs checked); the Stabiliser's entropy/LR
  schedule is meant to address that and is orthogonal to the reward, but the combination was NOT run here (it needs train.py/ppo.py
  changes from branch `stabilize`, which this branch does not touch).
- Everything is a single configuration of 5M steps, fixed reward weights chosen by arithmetic; the
  v3a weights (40, 0.03) were not tuned, so nothing is known about their optimality.
- Pilot/confirmation seeds are disjoint (10-12 vs 0-9) and the evaluation episodes are held out
  from all training and in-training selection seeds. v3 returns are not comparable to v2 returns
  (different reward), so only outcome metrics are compared.

## Reproducing

```bash
uv sync --extra torch --extra plots
uv run python scripts/run_queue.py --max-procs 3 --jobs \
    experiments/configs/ppo_final_v3a.json:0:rv3_v3a_s0 ...   # seeds 0-9 (pilot: names rv3_pilot_<v>_s{10,11,12})
uv run python scripts/rv3_analysis.py eval --out experiments/rv3_confirm_eval.json \
    --runs rv3_v3a_s0 ... rv3_v3a_s9 --episodes 500 --last-n 5 --procs 3
uv run python scripts/rv3_analysis.py eval --out experiments/rv3_hlv2_eval.json \
    --runs hl_v2_s0 ... hl_v2_s9 --episodes 500 --last-n 5     # needs ../rocket-landing-rl-headline results
uv run python scripts/rv3_analysis.py confirm --new experiments/rv3_confirm_eval.json \
    --base experiments/rv3_hlv2_eval.json --out experiments/rv3_confirm_stats.json
uv run python scripts/rv3_analysis.py pilot --eval experiments/rv3_pilot_eval.json
```

Representative checkpoint kept in the repo: `experiments/results/rv3_v3a_s7/final.pt` (+ `config.json`);
seed 7 is closest to the arm medians (greedy 500/500; sampled 95.6% vs arm mean 95.1%; greedy fuel
0.533 and 358 steps vs arm medians 0.537 and 363; last 5 checkpoints 100/100/100/100/99.5%). All other run directories stay
untracked (`experiments/results/*` is gitignored).

## Fresh-seed replication (added later)

The ten-seed confirmation above used training seeds 0-9. A second, pre-registered round on ten fresh seeds (10-19)
in [stabilisation.md](stabilisation.md) trained `v3a` again with the same config: 9 of 10 seeds at 90%+, mean
97.7%, no seed at 0%, against 5 of 10 and 64.0% for the control recipe. Pooled over the 20 seeds `v3a` is at
98.8% against 70.5% (post hoc, p = 0.004). The fresh-seed comparison alone is p = 0.047 and does not survive a
correction for the three comparisons made in that round; the seeds 10-12 were also used for the v3a-versus-v3b
pilot, so seeds 13-19 are the cleanest (v3a 96.7%, control 52.7%).
