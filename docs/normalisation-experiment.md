# Observation normalisation experiment

Question: does replacing the running observation normaliser with fixed physical scales make
PPO training more stable across seeds? (Hypotheses from code review: the running normaliser
changes during rollouts so buffers mix normalisations, and the near-constant fuel feature is
amplified 30-300x; `clip_vloss` with `reward_scale` 0.01 may slow the critic.)

## Implementation

New config key `obs_norm` (default `"running"`, so old configs and checkpoints behave as
before).

- `"running"`: `RunningMeanStd`, updated on every training step (unchanged).
- `"fixed"`: `FixedScaleNormalizer`, `clip(x / scale, -10, 10)` with scales
  `[x, y, vx, vy, angle, ang_vel, fuel] = [50, 100, 10, 10, 0.3, 0.5, 1]`. Stateless, `update`
  is a no-op.

Scale justification (from `docs/environment.md`): x is within +-100 before out-of-bounds
(start +-40); y within [0, 200] (start 80-120); start velocities are up to 10 m/s and landing
needs |vx| <= 1, |vy| <= 2, so /10 keeps the relevant range near 0.1-1; tilt starts within
+-0.2 and the landing limit is 0.2, so /0.3; angular velocity limit is 0.5 rad/s, so /0.5;
fuel is already a [0, 1] fraction. These were chosen once by reasoning and not tuned.

The normaliser type is saved in the checkpoint (`obs_norm` dict has `"type": "fixed"`); legacy
checkpoints have no `type` key and load as `RunningMeanStd`. `load_checkpoint` and
`PolicyController` pick the right class, so eval and viewer paths work unchanged.
Tests: `tests/test_obs_norm.py`.

## Setup

reward v2, 5M steps, otherwise `experiments/configs/ppo_v2.json` (gamma 0.999, ent_coef 0.03,
lr 3e-4 annealed, eval_every 10). Seeds 0-5 in every arm. Only the named setting differs.

| Arm | Config | Runs |
|---|---|---|
| A (control) | `ppo_v2.json` (running norm, clip_vloss=true) | `v2_s0..s5` from `~/projects/rocket-landing-rl/experiments/results/` (reused, not rerun) |
| B | `ppo_v2_fixed.json` (fixed scales) | `exp_fixed_s0..s5` |
| C | `ppo_v2_fixed_noclip.json` (fixed scales, clip_vloss=false) | `exp_fixednoclip_s0..s5` |

Arm A reuse: the `config.json` of `v2_s*` has the same values as `ppo_v2.json` and the code
commit is the same base (861a0b6) as this branch before my change; I did not rerun A, so I did
not verify bit-for-bit reproducibility of the control with the new code.

Evaluation: the FINAL checkpoint (`final.pt`, last update, not best-so-far) of every run, greedy,
200 episodes with reset seeds 40000..39999+200 (40000..40199), disjoint from training seeds
(1,000,000+) and in-training eval seeds (20000..20049). Script:
`scripts/eval_final_checkpoints.py`; raw output in `experiments/norm_experiment/`.

## Results: final-checkpoint success rate (200 episodes)

| Seed | A running | B fixed | C fixed, no clip_vloss |
|---|---|---|---|
| 0 | 0.000 | 1.000 | 1.000 |
| 1 | 0.860 | 1.000 | **0.000** |
| 2 | 0.000 | 1.000 | 1.000 |
| 3 | 0.000 | 1.000 | 1.000 |
| 4 | 0.795 | **0.700** | 1.000 |
| 5 | 0.000 | 1.000 | **0.000** |
| mean | 0.276 | 0.950 | 0.667 |
| min / max | 0.000 / 0.860 | 0.700 / 1.000 | 0.000 / 1.000 |
| seeds with success >= 0.5 | 2 / 6 | 6 / 6 | 4 / 6 |

Failed seeds (success < 0.5): A: 0, 2, 3, 5. B: none (seed 4 is the weakest at 0.70). C: 1, 5.

## Conclusion

- Fixed scales with the original `clip_vloss=true` (arm B) did help in this experiment: 6/6
  seeds ended with a good policy versus 2/6 for the running-norm control, mean final success
  0.95 versus 0.28. The difference is large relative to the seed spread, though n = 6 per arm
  and it is a single configuration, so it is evidence, not proof. It is consistent with the
  hypothesis that the moving normalisation / amplified fuel feature hurts, but the experiment
  does not isolate which of the two causes (or neither specifically) matters.
- Removing `clip_vloss` (arm C) did not help: 4/6, worse than B (2 collapsed seeds, 1 and 5).
  With n = 6 the B vs C difference is not statistically convincing either way, but there is no
  evidence that disabling value clipping is beneficial.
- Training is not fully stable even with fixed scales. Looking at `eval.jsonl`, success of
  several runs oscillates during training (e.g. `exp_fixed_s4` flips between 1.00 and ~0 on the
  50-episode in-training eval and ends at 0.64-0.74; `exp_fixednoclip_s5` reached 1.00 once and
  was back at 0 afterwards). A final-checkpoint metric therefore partly measures where in an
  oscillation training stopped; arm B's 6/6 could be partly luck of the stopping point. Checking
  stability over the last N checkpoints, or more seeds, would be the next step.
- The control A numbers (2/6 good) match the previously observed "roughly 1 in 3-6 seeds".

## Reproduce

```
uv run python -m rocket_landing.train --config experiments/configs/ppo_v2_fixed.json --seed 0 --run-name exp_fixed_s0
uv run python scripts/eval_final_checkpoints.py --out out.json B_s0=experiments/results/exp_fixed_s0 ...
```
