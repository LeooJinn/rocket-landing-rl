# Experiments

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
so early trajectories are shared. v1 remains the reward used for the reported numbers.

### Open issues

- Seed sensitivity and late collapse (above). Candidate fixes not yet tried: entropy
  schedule, value normalisation instead of a fixed reward scale, more training steps with
  early stopping on validation, curriculum on initial states.
- The observation has no time-remaining feature, but the episode is truncated at 1000
  steps with a -50 penalty; the value function cannot see the timeout coming.
- The learned policy is slow and fuel-hungry compared with the baseline; v1's fuel and
  time costs are small next to the terminal rewards, by design.
- No plots yet (Matplotlib is not a dependency); curves above come from `metrics.jsonl` and
  `eval.jsonl`.
