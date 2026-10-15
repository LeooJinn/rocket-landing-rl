# Experiments

PPO (from scratch, `src/rocket_landing/ppo.py`) on the 2D landing environment
(`docs/environment.md`) with the reward versions in `docs/reward-design.md`.
All numbers below come from the `metrics.jsonl` / `eval.jsonl` / `config.json` /
`summary.json` files of the runs listed; nothing is estimated. Snapshot taken
2026-10-15 (see "Status of runs"). Held-out evaluation numbers are NOT in this
document yet (see the TODO section).

## Setup

- Algorithm: PPO with GAE, advantage normalisation, clipped value loss, annealed
  learning rate, 16 parallel envs x 256 steps per update (4096 env steps/update) unless
  noted. Policy and value share an MLP 64-64 with tanh (4 discrete actions), observation
  normalisation by running mean/std. Rewards are scaled by 0.01 for learning; logged
  rewards are raw.
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

## TODO: held-out evaluation (another builder is producing these)

Do not fill with estimates; paste from the held-out evaluation output once it exists.

| Policy | Checkpoint | Episodes | Start seeds | Success | Crash | Out of bounds | Timeout | Source file |
|---|---|---|---|---|---|---|---|---|
| Heuristic baseline | n/a | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| v1, `sweep_g999e03_s*` per seed | final.pt | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| v1, `sweep_g999e03_s*` per seed | best.pt | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| v2, `v2_s*` per seed | final.pt | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| v2, `v2_s*` per seed | best.pt | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

TODO: add a held-out v1-vs-v2 comparison (final.pt for all seeds, mean and per seed)
and re-assess the observations in item 3 above with it.
TODO: replace the "Status of runs" section after `kl_*` runs finish.

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
