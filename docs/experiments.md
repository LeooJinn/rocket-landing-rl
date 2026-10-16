# Experiments

> **Reading guide.** This file is chronological. It was written with the original running
> observation normaliser (the code default at the time) and covers the first PPO policy, the
> six-seed reward v1/v2 comparison and the run families. Later work supersedes parts of it:
> fixed-scale normalisation ([normalisation-experiment.md](normalisation-experiment.md)) and the
> 10-seed headline comparison ([headline-results.md](headline-results.md)), which is the current
> best evidence. Run directories named below (`v2_s0`, `sweep_*`, `r2_*`, `kl_*`, ...) were kept
> outside this repository; only `experiments/results/ppo_v1_tuned_s1/` and the `hl_v2_s2` final
> checkpoint are committed. Tables are kept as recorded when those runs were made.

## Milestone 6: first PPO policy

PPO is implemented from scratch in PyTorch (`src/rocket_landing/ppo.py`, `policy.py`,
`train.py`): separate 64x64 tanh actor and critic MLPs, running observation normaliser
(saved in the checkpoint), GAE, clipped surrogate + clipped value loss, entropy bonus,
gradient-norm clipping, linear LR annealing, 16 environment instances stepped in lockstep.
Rewards are multiplied by `reward_scale = 0.01` for learning only; every logged reward is
the raw environment reward. Timeouts are treated as terminal in GAE because reward v1 pays
a -50 terminal timeout penalty (the observation has no time feature, see open issues).

### Seed protocol

- Training env `i` of run seed `s` is reset once with `1_000_000 + 1000*s + i`, then reset
  without a seed (its own RNG stream continues).
- Checkpoint selection (`best.pt`) uses validation seeds 20000..20049, evaluated greedily
  every 10 updates.
- Test numbers below use seeds 0..99, which were used neither for training nor for
  selection (the same seeds as the heuristic baseline numbers in `environment.md`).

### Final configuration

`experiments/configs/ppo_v1_tuned.json` (reward v1): `ppo_v1.json` with `gamma = 0.999`,
`ent_coef = 0.03`, `total_timesteps = 5_000_000`. One run took 537 s on one CPU thread
(Apple Silicon, run alone), including 122 greedy validation evaluations of 50 episodes; raw
rollout+update throughput is about 26-30k env steps/s.

```bash
uv run python -m rocket_landing.train --config experiments/configs/ppo_v1_tuned.json --seed 1
uv run python scripts/evaluate_policy.py --checkpoint experiments/results/ppo_v1_tuned_s1/final.pt \
    --episodes 100 --seed 0 --compare-baseline
```

Reproducibility check: re-running seed 1 from this config reproduced all 1220 per-update
metric records exactly (ignoring wallclock) compared with the original sweep run.

The reference run `experiments/results/ppo_v1_tuned_s1/` (metrics, eval log, best and final
checkpoints) is committed; all other runs are gitignored.

### Training curve (reference run, seed 1)

Averages over windows of 122 updates; training columns are over episodes that finished in
those updates (stochastic policy), `val succ` is the greedy validation success rate.

| env steps (M) | train succ | crash | oob | ep reward | ep length | fuel | entropy | val succ |
|---|---|---|---|---|---|---|---|---|
| 0.0-0.5 | 0.00 | 0.89 | 0.11 | -100.5 | 100 | 0.07 | 0.78 | 0.00 |
| 0.5-1.0 | 0.00 | 0.99 | 0.01 | -97.2 | 112 | 0.09 | 0.36 | 0.00 |
| 1.0-1.5 | 0.00 | 0.98 | 0.02 | -114.1 | 745 | 0.93 | 0.89 | 0.00 |
| 1.5-2.0 | 0.00 | 1.00 | 0.00 | -113.1 | 834 | 1.00 | 0.71 | 0.00 |
| 2.0-2.5 | 0.00 | 1.00 | 0.00 | -111.7 | 822 | 1.00 | 0.64 | 0.00 |
| 2.5-3.0 | 0.18 | 0.82 | 0.00 | -70.7 | 698 | 0.87 | 0.64 | 0.14 |
| 3.0-3.5 | 0.46 | 0.54 | 0.00 | -5.9 | 350 | 0.48 | 0.72 | 0.49 |
| 3.5-4.0 | 0.75 | 0.25 | 0.00 | 53.1 | 395 | 0.55 | 0.80 | 0.77 |
| 4.0-4.5 | 0.77 | 0.23 | 0.00 | 58.6 | 366 | 0.51 | 0.83 | 0.87 |
| 4.5-5.0 | 0.87 | 0.13 | 0.00 | 75.5 | 373 | 0.52 | 0.86 | 0.95 |

The run first collapses to free-fall (short, fuel-free crashes), then to hovering until the
tank is empty, and only after about 2.5M steps discovers landing.

### Test results (seeds 0..99, greedy policy)

| | success | crash | out of bounds | timeout | mean reward | fuel | steps |
|---|---|---|---|---|---|---|---|
| Heuristic baseline | 99% | 1% | 0% | 0% | 109.29 | 0.365 | 269.5 |
| PPO seed 1, `best.pt` (update 1070) | 100% | 0% | 0% | 0% | 106.11 | 0.605 | 453.2 |
| PPO seed 1, `final.pt` | 95% | 5% | 0% | 0% | 93.50 | 0.594 | 430.7 |

The 5 failures of `final.pt` (seeds 35, 80, 81, 83, 97) all touched down off the pad with
speed, tilt and spin inside the limits. The PPO policy touches down more gently than the
baseline (mean |vy| 0.82 vs 1.14 m/s, |angle| 0.015 vs 0.033 rad) but takes about 60% more
steps and uses about 60% more fuel. 100 episodes is an estimate, not a guarantee; proper
held-out evaluation is milestone 7.

### Seed reliability (the main caveat)

Same config, six training seeds, test seeds 0..99:

| seed | best.pt success | final.pt success |
|---|---|---|
| 0 | 100% | 0% |
| 1 | 100% | 95% |
| 2 | 68% | 14% |
| 3 | 0% | 0% |
| 4 | 100% | 75% |
| 5 | 100% | 0% |

Four of six seeds produced, at some point, a checkpoint that landed 100/100, but only two
kept a good policy until the end of training, and one never learned. Learning is
unstable: seeds 0 and 5 reached 100% validation success and later collapsed back to the
crash/hover attractors. Picking `best.pt` by validation success is what makes the 100%
numbers possible; it should not be read as "PPO reliably solves the task".

### What was tried (all seed 0 unless noted, 5M steps unless noted)

| change | result |
|---|---|
| defaults (`ppo_v1.json`, gamma 0.99, ent 0.01, 2M) | 0% success; free-fall / tumbling crashes, episodes grew longer to postpone the crash |
| gamma 0.999 | 0%; converged to immediate free-fall |
| gamma 0.999, lambda 0.98 | 0%; hover until fuel empty |
| gamma 0.995 | 0%; hover until fuel empty |
| gamma 0.999, ent 0.005 | best val 30%, ended at 0% |
| gamma 0.999, hidden 128x128 | 0% |
| **gamma 0.999, ent 0.02** | 100% val from about 2.7M steps (seed 0); seeds 1-3: 0%, 22% best, 88% best then 0% |
| **gamma 0.999, ent 0.03** (chosen) | see seed table above |
| gamma 0.9995, ent 0.03, 8M (seeds 0, 1) | seed 1 solved (1.00 at end), seed 0 0% |
| 32 envs x 512 steps, gamma 0.999 or 0.9995, ent 0.03, 8M (seeds 0, 1) | 0%, 42% at end, 0%, 0% |
| ent 0.03, 4 epochs, target_kl 0.015, 6M (seeds 0-5) | 0% on all six (much slower learning) |
| reward v2 (graded crash penalty), gamma 0.999, ent 0.03 (seeds 0-5) | no better than v1 (see below) |

### Diagnosis

Two attractors account for every failed run:

1. **Immediate free-fall** (~85 steps, ~0 fuel, return about -96.5). This is a reward v1
   issue: the crash terminal is a flat -100, and falling collects about +10 distance shaping
   against about -4.4 speed shaping, so a fast crash scores better than any careful
   approach that still crashes. The only terminal-time signal for impact softness is the
   small speed term in the potential.
2. **Hover until the tank is empty, then crash** (~800 steps, return about -115). This is a
   discounting effect: with gamma < 1 a crash 800 steps away is heavily discounted, so
   postponing it looks cheap, although its undiscounted return is worse than free-fall.

Higher entropy (0.02-0.03) and a long horizon (gamma 0.999) were the only changes that let
the agent escape both on some seeds.

### Reward v2 result

Reward v2 (`docs/reward-design.md`) grades the crash penalty from -100 to -50 by how close
the touchdown came to the safe limits, targeting attractor 1. With the chosen PPO settings
on seeds 0-5 it was not better than v1: validation success over the last 20% of training
was 0.22, 0.91, 0.09, 0.00, 0.88, 0.00 (v1: 0.18, 0.91, 0.18, 0.00, 0.91, 0.00). Seeds 1 and 4
succeed under both versions, which is expected because v2 equals v1 except on crash steps,
so early trajectories are shared. v1 was the reward used for the numbers in this section; with fixed-scale
normalisation and 10 seeds per arm v2 looked better (see [headline-results.md](headline-results.md)).

### Reward v3 result

Reward v3a/v3b (`docs/reward-design.md`) price the slow, fuel-hungry mode that the headline analysis
found behind greedy-only collapses: a landing bonus of `40 * fuel_left` and a time penalty of
0.03 per step (v3b adds -25 for an empty-tank crash). Pilot on seeds 10-12 picked v3a; on seeds 0-9
(final checkpoint, 500 held-out episodes) its greedy policy landed 5000/5000 against 77.0% mean for
`hl_v2` (paired, suggestive, p = 0.062), using 0.53 vs 0.71 of the tank and 362 vs 486 steps on the
seeds where v2 lands; sampled-action success was not clearly better (p = 0.35). Full numbers and
caveats: `docs/reward-v3-results.md`.

### Open issues

- Seed sensitivity and late collapse (above). Candidate fixes not yet tried: entropy
  schedule, value normalisation instead of a fixed reward scale, more training steps with
  early stopping on validation, curriculum on initial states.
- The observation has no time-remaining feature, but the episode is truncated at 1000
  steps with a -50 penalty; the value function cannot see the timeout coming.
- The learned policy is slow and fuel-hungry compared with the baseline; v1's fuel and
  time costs are small next to the terminal rewards, by design.
- Plots were added later (see "Figures" below); the curves above come from `metrics.jsonl` and
  `eval.jsonl`.

---

# Run families, figures and limitations

PPO (from scratch, `src/rocket_landing/ppo.py`) on the 2D landing environment
(`docs/environment.md`) with the reward versions in `docs/reward-design.md`.
All numbers below come from the `metrics.jsonl` / `eval.jsonl` / `config.json` /
`summary.json` files of the runs listed; nothing is estimated. Snapshot taken
2026-10-15 (see "Status of runs"). Held-out evaluation numbers are in
[evaluation.md](evaluation.md) (summary under "Held-out evaluation" below).

## Setup

- Algorithm: PPO with GAE, advantage normalisation, clipped value loss, annealed
  learning rate, 16 parallel envs x 256 steps per update (4096 env steps/update) unless
  noted. Separate actor and critic MLPs, 64-64 tanh each (4 discrete actions), observation
  normalisation by running mean/std (the code default when these runs were made; fixed
  scales are the default since the headline experiment). Rewards are scaled by 0.01 for
  learning; logged rewards are raw.
- Seeding: `--seed S` seeds torch/numpy/Python and training env `i` with
  `1_000_000 + S*1000 + i`; seeds 0..99999 are never used for training.
- Evaluation during training (`eval.jsonl`): every 10 updates (every 5 for `r2_B`, `r2_C`),
  50 episodes, deterministic (argmax) policy, fixed start seeds 20000..20049, same
  reward version as training. Outcomes are landed / crashed / out_of_bounds / timeout.
  `metrics.jsonl` is per-update training statistics from the stochastic policy.
- Eval uses the SAME 50 start states at every checkpoint, and success is
  quantised to steps of 2%. It is a monitoring signal, not an unbiased
  performance estimate (see Limitations).

Terminology used in tables and figures:

- **final**: success rate at the last evaluation of the run (what the policy at
  `final.pt` achieves on those 50 episodes).
- **best-so-far**: maximum success rate over all of the run's evaluations (what
  `best.pt` was selected on). It is selected on the same episodes it is reported on, so it
  is optimistic and is never used as a headline number here.

## Run families

Defaults (identical across all runs unless listed): lr 3e-4 annealed, gae_lambda 0.95,
clip 0.2, vf_coef 0.5, max_grad_norm 0.5, 10 epochs, 8 minibatches, MLP 64-64.
Columns "final / best-so-far" are eval success rates (50 episodes) per seed.

| Family (run dirs) | Reward | Changed vs defaults | Steps | Seeds | Final success per seed | Best-so-far per seed |
|---|---|---|---|---|---|---|
| `ppo_v1_s0` (first run, `ppo_v1.json` defaults) | v1 | gamma 0.99, ent 0.01 | 2M | 0 | 0.00 | 0.00 |
| `sweep_g995` | v1 | gamma 0.995, ent 0.01 | 5M | 0 | 0.00 | 0.00 |
| `sweep_g999` | v1 | gamma 0.999, ent 0.01 | 5M | 0 | 0.00 | 0.00 |
| `sweep_g999e005` | v1 | gamma 0.999, ent 0.005 | 5M | 0 | 0.00 | 0.30 |
| `sweep_g999h128` | v1 | gamma 0.999, ent 0.01, hidden 128-128 | 5M | 0 | 0.00 | 0.00 |
| `sweep_g999l98` | v1 | gamma 0.999, ent 0.01, gae_lambda 0.98 | 5M | 0 | 0.00 | 0.00 |
| `sweep_g999e02`, `_s1`, `_s2`, `_s3` | v1 | gamma 0.999, ent 0.02 | 5M | 0-3 | 1.00, 0.00, 0.00, 0.00 | 1.00, 0.00, 0.22, 0.88 |
| `sweep_g999e03_s0` .. `_s5` | v1 | gamma 0.999, ent 0.03 | 5M | 0-5 | 0.00, 0.98, 0.18, 0.00, 0.80, 0.00 | 1.00, 1.00, 0.74, 0.00, 1.00, 1.00 |
| `v2_s0` .. `v2_s5` | **v2** | gamma 0.999, ent 0.03 (`ppo_v2.json`) | 5M | 0-5 | 0.00, 0.84, 0.00, 0.00, 0.84, 0.00 | 1.00, 1.00, 0.86, 0.00, 1.00, 0.00 |
| `r2_A_s0`, `_s1` | v1 | gamma 0.9995, ent 0.03 | 8M | 0-1 | 0.00, 1.00 | 0.26, 1.00 |
| `r2_B_s0`, `_s1` | v1 | gamma 0.999, ent 0.03, 32 envs x 512 steps (16384/update), eval every 5 | 8M | 0-1 | 0.00, 0.42 | 0.00, 0.48 |
| `r2_C_s0`, `_s1` | v1 | gamma 0.9995, ent 0.03, 32 envs x 512 steps, eval every 5 | 8M | 0-1 | 0.00, 0.00 | 0.00, 0.00 |
| `kl_s0` .. `kl_s5` | v1 | gamma 0.999, ent 0.03, update_epochs 4, target_kl 0.015 (early stop) | 6M | 0-5 | 0.00 x6 | 0.00 x6 |

All `v1`/`v2` labels are `reward_version` in each run's `config.json`; the
"changed" column was derived by diffing every `config.json` against `ppo_v2.json`.
`sweep_g999e03_s*` are the v1 control for `v2_s*`: identical config apart from
`reward_version`, same seeds 0-5.

## Status of runs

Snapshot of `~/projects/rocket-landing-rl/experiments/results` at 2026-10-15 ~14:15 local.
All runs listed had written the number of updates implied by their `total_timesteps`
(i.e. had reached their last evaluation). `kl_s0`, `_s1`, `_s2`, `_s4`, `_s5` had no
`summary.json` yet (the training process was still active, i.e. finishing up), so
their numbers could still be overwritten by a final write; figures mark such runs with
a diamond and tables should be regenerated once they finish (`scripts/plot_training.py`
reads whatever is on disk and tolerates a partially written last line).

## Figures

Generated with `scripts/plot_training.py` (see "Reproducing").

Reward v1 vs v2, same hyperparameters, six seeds each. Thin lines are individual
seeds, thick lines the mean over seeds, right panel shows every seed's final (filled)
and best-so-far (hollow) success rate:

![v1 vs v2](figures/v1_vs_v2.png)

A v2 seed that peaks and then collapses (`v2_s0`: best-so-far 100%, final 0%):

![v2_s0](figures/run_v2_s0.png)

A v2 seed that ends well (`v2_s1`: final 84%):

![v2_s1](figures/run_v2_s1.png)

## What was observed

Facts from the logs:

1. **Nothing in the sweep learned reliably early.** Every run sat at 0% eval success
   for the first ~2.8M steps. With reward v1 the early failure modes are the ones
   `docs/reward-design.md` predicted: a fast free-fall crash (~90 steps, return ~-96,
   e.g. `sweep_g999` still ends there with 91.8 mean steps) or hovering until fuel runs
   out and then crashing (~790-870 steps, return ~-114 to -118, e.g. `sweep_g995`,
   `sweep_g999e005`, `sweep_g999h128`, `sweep_g999l98`).
2. **Single-seed sweeps are uninformative.** Of the one-seed runs with ent 0.005-0.01
   (and gamma 0.99/0.995/0.999, hidden 128, lambda 0.98), none finished above 0%
   success and only `sweep_g999e005` ever exceeded 0% (best-so-far 30%). With ent 0.02
   or 0.03 some seeds reached 100%, but the same config gave 0% on other seeds
   (ent 0.02: final 1.00 / 0.00 / 0.00 / 0.00 over seeds 0-3). Higher entropy
   coefficient and gamma >= 0.999 correlate with seeds that learn to land at all, but
   with 1-6 seeds per setting and this much seed variance we cannot rank them.
3. **v1 vs v2 (6 seeds each, same config otherwise).** Final success, mean over seeds:
   v1 0.327 (seeds: 0.00, 0.98, 0.18, 0.00, 0.80, 0.00), v2 0.280 (0.00, 0.84, 0.00,
   0.00, 0.84, 0.00). Mean success over each seed's last 20 evals: v1 0.374, v2 0.350.
   Two seeds per reward version end with a good policy (>= 80%) (v1: seeds 1, 4; v2: seeds 1, 4);
   four v1 seeds and four v2 seeds end at <= 18%. This data does NOT show v2 helping
   success rate; the difference between group means is far inside seed noise.
   Landings first reach >= 50% eval success at 2.8-4.4M steps for v1 and 3.0-4.2M for
   v2 (where they do at all), and v2 seeds 3 and 5 never reach it.
4. **Instability after learning.** Many seeds reach 100% and later fall back to
   0%: `sweep_g999e03_s0`, `_s5`, `v2_s0` have best-so-far 100% and final 0%
   (`v2_s0` is shown above; its training entropy rises from ~0 back to ~1.1 and the
   value loss rises by four orders of magnitude after ~1.5M steps). Eval success
   oscillates between 0 and 1 from one evaluation to the next within the same run,
   e.g. in `v2_s0` after 4M steps. Final-checkpoint and best-checkpoint numbers
   therefore differ a lot and neither describes "the algorithm".
5. **Longer horizons / bigger batches (`r2_*`, 2 seeds each, 8M steps).** The only
   run that ended and stayed at 100% was `r2_A_s1` (gamma 0.9995; 92 of its 196 evals are
   >= 50%), while `r2_A_s0` with the same config never exceeded 26%. The 32x512 batch
   variants B and C gave 0.00/0.42 and 0.00/0.00 final. Two seeds per config say
   little.
6. **KL early stopping (`kl_*`).** No `kl_*` run reached any eval success (best-so-far
   0% for all six seeds at 6M steps) in this configuration (4 epochs, target_kl 0.015).
   With the same gamma and ent as the working `sweep_g999e03_*` runs, this is
   evidence that the setting hurt, but it was also run with fewer epochs, so the
   two changes are confounded.

Interpretation (not measured): the shared pattern is a long plateau at crash/
hover failure modes followed by a sudden jump to landing in some seeds, then
policy collapse in some of them. Entropy coefficient 0.03 with a long discount horizon
appears to help escape the plateau; we have not tested why, nor tested whether
the collapses are from PPO instability, observation normalisation drift (the running
mean/std keeps updating throughout training), value-loss blow-up or the
learning-rate schedule.

## Limitations

- **Seed instability is the main result.** At most 6 seeds per config, most with
  1-2. Final success per seed is bimodal (about 0% or about 80-100%); means over such
  samples are dominated by how many seeds happened to be in the good mode. We report
  every seed; we do not report best seeds as the result.
- **Best-so-far is optimistic.** It is chosen as the maximum over about 120 evaluations
  on the same 50 start states it is reported on; with 0/1-style swings, a seed can
  hit 100% on one evaluation and 0% on the next. `best.pt` is therefore not a
  validated model.
- **Eval set is small and fixed** (50 episodes, seeds 20000..20049; success has +/-
  ~6 percentage points of binomial standard error at 80%). The same episodes were used
  to compare configs, so there is mild selection against this set.
  Held-out evaluation on unseen starts is pending (below).
- **No hyperparameter sweep is controlled.** Sweeps varying gamma, entropy, lambda and
  width are one seed per setting; `kl_*` and `r2_*` change several things at once
  or use different batch sizes and total steps; steps differ across families (2M, 5M, 6M, 8M).
- **Reward comparisons are not apples to apples in eval.** Each run's eval `mean_reward`
  uses its own reward version, so v1 and v2 mean rewards are on slightly different scales
  (a crash is worth -100..-50 in v2 and -100 in v1). Success/crash rates are comparable.
- **Training metrics in `metrics.jsonl` use the stochastic policy** and the few episodes
  that finish in an update (as few as a handful early on), so `ep_reward_mean` and
  success there are noisy and sometimes absent. Figures use eval records for outcome rates.
- Single environment (2D, discrete 4-action thrusters), one network size mostly, no
  comparison to other RL algorithms; the heuristic baseline (docs/environment.md) lands
  about 99% of 100 episodes, which is better than any RL final-checkpoint average here.

## Held-out evaluation

Held-out results (500 episodes on seeds 50000..50499, disjoint from training and
in-training evaluation seeds, with 95% Wilson intervals, greedy and stochastic,
against the heuristic baseline and a random policy) are in `docs/evaluation.md`,
with raw results in `docs/heldout_final.json` and `docs/heldout_best.json`.
Headline: final greedy success averages 40.8% across 9 runs (0% to 100%), with
4 of 9 runs never landing; `best.pt` checkpoints hold up on fresh seeds for the
runs that learned at all.

Note: the `kl_*` rows above come from the "Status of runs" snapshot and were not regenerated after those
runs finished.

## Reproducing

```bash
uv sync --extra plots --extra torch
R=~/projects/rocket-landing-rl/experiments/results
uv run python scripts/plot_training.py --results $R --run v2_s0 --run v2_s1 \
    --compare 'sweep_g999e03_s*' --compare 'v2_s*' --group-by reward_version \
    --out docs/figures/v1_vs_v2.png \
    --title "Reward v1 vs v2 (gamma 0.999, ent 0.03, 5M steps, 6 seeds each)"
```

`--group-by` accepts any flattened config keys (e.g. `ppo.gamma ppo.ent_coef`). Functions
live in `src/rocket_landing/plotting.py`; tests in `tests/test_plotting.py`.
