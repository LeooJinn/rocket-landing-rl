# Why the fixed-scale runs collapse late in training

> **Provenance.** Produced by an analysis agent (role "diagnostician") from the 20 finished headline runs
> `hl_v1_s0..s9` and `hl_v2_s0..s9` ([headline-results.md](headline-results.md)): their `metrics.jsonl`,
> `eval.jsonl` and saved checkpoints. No new training was run. Checkpoint rollouts use reset seeds 50000 and
> up (the headline's held-out seeds, so they are **diagnostic only** and must not be used to select
> checkpoints), 100 episodes per cell unless stated, and a numpy re-implementation of the actor and critic
> (float64, same weights as the torch model; it reproduces the headline greedy results at the same
> checkpoints). Scripts and raw outputs: [../analysis/collapse_diagnosis/](../analysis/collapse_diagnosis/)
> (the checkpoints themselves are git-ignored, see its README). Every claim below is labelled
> **CONFIRMED** (a check was run on the data and its output is in `raw/`) or **SUSPECTED** (consistent with
> the data, but not tested). Correlations are over 9 to 12 seeds and are descriptive only.

## 1. Summary

1. **CONFIRMED. The failures are fuel exhaustion.** Of 601 in-training greedy evaluations at update 600 or
   later whose crash rate is at least 90%, 558 have a mean fuel use of at least 0.99 (the whole tank). The
   final policies of all 9 runs that crash on every episode (v1 s0, s4-s9; v2 s6, s8) run out of fuel in
   20 of 20 episodes (section 3).
2. **CONFIRMED. The collapse is greedy-only.** In v2 s8 at updates 1140 to 1220 the sampled training
   episodes land about 100% of the time while the greedy evaluation is 0% (sections 2 and 4).
3. **CONFIRMED. Greedy and sampled are different controllers.** On every checkpoint that lands at all, fuel
   use falls steadily as the sampling temperature rises (section 4).
4. **CONFIRMED. The argmax is nearly tied in most states**, for healthy and collapsed checkpoints alike
   (section 5).
5. **CONFIRMED description, SUSPECTED cause. The policy sits close to the fuel cliff and nothing in the
   training objective pushes it away.** A 234-step landing and a 662-step landing differ by only 4 points of
   raw return (section 6).
6. **CONFIRMED description, SUSPECTED cause. Seeds that never landed, and seeds that learned and then lost
   it (v1 s5, v1 s8), are in an absorbing "hover until empty" state** (sections 3 and 6).
7. **Ruled out as causes:** entropy collapse, value-function divergence, large policy updates at the moment
   of collapse, and late learning-rate annealing creating the collapses (section 8).

## 2. One collapse in detail: `hl_v2_s8`, updates 1000 to 1220

Source: [`raw/window_hl_v2_s8_1000_1220.txt`](../analysis/collapse_diagnosis/raw/window_hl_v2_s8_1000_1220.txt)
(metrics of the sampled training episodes, then, after `|`, the in-training greedy evaluation on 50
episodes at every 10th update). CONFIRMED. The table shows the updates that carry an evaluation; the
training columns are that update's batch of 16 x 256 steps.

| update | train success | train fuel used | greedy eval success | greedy eval fuel used | greedy eval steps |
|---|---|---|---|---|---|
| 1020-1090 (typical) | 0.61-1.00 | 0.36-0.43 | 1.00 | 0.60-0.66 | 416-497 |
| 1100 | 1.00 | 0.52 | 1.00 | 0.77 | 540 |
| 1110 | 1.00 | 0.79 | 1.00 | 0.95 | 674 |
| 1130 | 1.00 | 0.77 | 0.60 | 0.995 | 702 |
| 1140 | 1.00 | 0.88 | **0.00** | 1.000 | 707 |
| 1150 | 1.00 | 0.80 | 1.00 | 0.96 | 670 |
| 1160 | 1.00 | 0.88 | **0.00** | 1.000 | 702 |
| 1170-1220 | 1.00 | 0.90-0.97 | **0.00** | 1.000 | 703 |

- The sampled policy keeps landing in nearly every training batch (success 1.00 in most updates, 0.71-0.89 in
  a few batches with 1-4 crashes; episode return about 100-105) while the greedy evaluation is at 0%, and its fuel use climbed from about 0.4 of the tank (update 1090) to about 0.95 (update 1170)
  in 80 updates. The sampled policy is therefore also close to the cliff at the end.
- Entropy stays between 0.97 and 1.06 nats (the maximum for four actions is 1.386). At these updates
  `approx_kl` is about 1e-4, `clip_frac` is about 0, the learning rate is 2e-5 and falling to 0,
  `value_loss` is about 1e-4 and `explained_variance` is 0.96 to 0.999. Dips of explained variance (for
  example 0.155 at update 1121) coincide with batches containing a few rare crashes, so they are a
  consequence of the crashes, not their cause.
- So the policy barely moves per update, yet the greedy outcome flips between evaluations 10 updates apart.

## 3. How the collapsed policies fail

**Failure is fuel exhaustion (CONFIRMED).** [`raw/fuel_vs_outcome.txt`](../analysis/collapse_diagnosis/raw/fuel_vs_outcome.txt)
pools all 2440 in-training greedy evaluations (20 runs x 122, 50 episodes each, seeds 20000-20049):

- At update 600 or later, 558 of the 601 evaluations with crash rate of at least 90% have mean fuel use
  of at least 0.99; the never-learned runs look the same (0.994) as the runs that learned and then
  collapsed (0.989).
- Greedy success against the evaluation's mean fuel use (update 600 or later, excluding
  out-of-bounds-dominated evaluations): below 0.90 gives 89-95%; 0.90-0.95 gives 81% (n=59); 0.95-0.99
  gives 48% (n=86); 0.99 and above gives 3% (n=613).
- Among the nine v2 seeds that learned, 89 of the 91 evaluations with success below 20% after the run's
  first 90% evaluation have fuel use of at least 0.97 (the other two are in v2 s9).

**Two failure types** ([`raw/final_failure_modes.txt`](../analysis/collapse_diagnosis/raw/final_failure_modes.txt),
`final.pt`, greedy, 20 episodes, seeds 50000-50019; CONFIRMED). In every row listed, the fuel ran out in all
20 episodes, except v2 s4 (6 of 20):

| type | runs | step fuel ran out | altitude when it ran out | touchdown vy |
|---|---|---|---|---|
| 1. slow glide that runs dry just above the ground | v2 s8 | 690 | 4.2 m | -9.2 m/s |
| | v2 s4 (5 of 20 crash) | 674 | 0.3 m | -1.6 m/s |
| 2. hover or creep until empty at altitude | v2 s6 | 786 | 67.8 m | -36.5 m/s |
| | v1 s0 / s4 | 785 / 787 | 9.5 m / 17.7 m | -13.3 / -18.7 m/s |
| | v1 s5 / s6 / s7 / s8 / s9 | 779-788 | 36.6 / 33.6 / 58.6 / 46.9 / 79.4 m | -25.7 to -39.5 m/s |

The other 10 final policies (v1 s1-s3, v2 s0-s3, s5, s7, s9) never ran out of fuel in 20 episodes.

**Fuel arithmetic (CONFIRMED against the measurements).** A full MAIN burn costs 0.0025 of the tank per
step, LEFT or RIGHT cost 0.0005. A policy that fires MAIN on 52% of steps and LEFT or RIGHT on 34% (greedy v2 s8
@1200, from [`raw/altitude_profile.txt`](../analysis/collapse_diagnosis/raw/altitude_profile.txt)) burns about
0.0015 per step, so the tank lasts about 670-700 steps; the v2 s8 final policy runs dry at step 690 on average. A 50/50 MAIN/NOOP hover (v2 s6) lasts about 780 steps, matching the measured 786. At this duty
cycle fuel use is proportional to time, and the episode length is the fuel budget.

**The fuel race, one seed** ([`raw/fuel_race_trace.txt`](../analysis/collapse_diagnosis/raw/fuel_race_trace.txt),
seed 50003; CONFIRMED). The heuristic lands in 262 steps with 66% of the tank left. `hl_v2_s8` checkpoint 1150
(100% greedy) lands at step 665 with 5.1% left. Checkpoint 1200 (0% greedy) follows the same descent
profile, its tank empties about 1 m above the ground, and it hits the ground at about 6.9 m/s. A healthy
policy such as v2 s3 `final.pt` lands in about 234 steps having used about 35% of the tank.

**The hover policy of v2 s6** ([`raw/hover_rollout_hl_v2_s6.txt`](../analysis/collapse_diagnosis/raw/hover_rollout_hl_v2_s6.txt);
CONFIRMED). Greedy and sampled both play `NMNMNM...` (MAIN probability about 0.5, vertical speed
oscillating between about -0.5 and +0.5 m/s) at 65-77 m for about 790 steps, then free-fall and crash at
35-37 m/s. Success is 0% at every temperature.

## 4. Greedy versus sampled, and temperature

Source: [`raw/temperature_sweep_t1.txt`](../analysis/collapse_diagnosis/raw/temperature_sweep_t1.txt) and
[`raw/greedy_vs_sampled.txt`](../analysis/collapse_diagnosis/raw/greedy_vs_sampled.txt). CONFIRMED.

Landed episodes out of 100 [mean fuel used]; T = 0 is greedy; logits divided by T before sampling:

| checkpoint | T=0 | T=0.25 | T=0.5 | T=0.75 | T=1.0 |
|---|---|---|---|---|---|
| v2 s8 @1000 | 100 [.79] | 100 [.75] | 100 [.71] | 100 [.65] | 97 [.58] |
| v2 s8 @1150 | 99 [.97] | 100 [.94] | 100 [.90] | 99 [.85] | 99 [.79] |
| v2 s8 @1200 | 0 [1.00] | 1 [1.00] | 37 [1.00] | 100 [.96] | 98 [.92] |
| v2 s8 final | 0 [1.00] | 0 [1.00] | 1 [1.00] | 50 [1.00] | 98 [.96] |
| v2 s7 @1000 | 0 [1.00] | 0 [1.00] | 0 [1.00] | 13 [1.00] | 86 [.98] |
| v2 s7 @1050 | 100 [.62] | 100 [.59] | 100 [.52] | 99 [.46] | 81 [.40] |
| v2 s7 @1100 | 0 [1.00] | 0 [1.00] | 14 [1.00] | 80 [.97] | 99 [.91] |
| v2 s7 @1200 | 100 [.92] | 100 [.87] | 100 [.81] | 100 [.75] | 98 [.67] |
| v2 s6 @1200, final | 0 [1.00] at every T | | | | |

(T = 0.1 gives the same as T = 0.) On every checkpoint that lands at all (all except v2 s6), mean fuel use
at T = 1 is below mean fuel use of the greedy policy. The two policies also fail differently: on 10 of the
11 non-s6 checkpoints above, none of the sampled crashes ran out of fuel (they crash with fuel left, 1-19
per 100 episodes), whereas the greedy crashes on the same checkpoints are all fuel-out
([`raw/temperature_sweep_t1.json`](../analysis/collapse_diagnosis/raw/temperature_sweep_t1.json) has the
split). Sampling is not uniformly better: v2 s7 @1050 is 100% greedy but 81% sampled, and all 19 of its
sampled crashes happen with fuel left (it descends faster; the cause of those crashes was not examined).

Greedy against sampled success on the late checkpoints (1000, 1050, ..., 1200 and `final.pt`, 100 episodes
each, all ten v2 runs, so 60 checkpoints; [`raw/greedy_vs_sampled_summary.txt`](../analysis/collapse_diagnosis/raw/greedy_vs_sampled_summary.txt)):
mean greedy success 0.77, mean sampled success 0.84. Greedy is below 20% while sampled is 80% or more on 6
checkpoints; sampled is below 20% while greedy is 80% or more on none. Greedy is below 80% on 16
checkpoints, sampled on 11. Over these checkpoints the lowest sampled success of any v2 seed that learned is
0.46 (v2 s5), whereas the lowest greedy success is 0.00 for v2 s4, s7, s8 and s9. Sampling is therefore
better on average and never collapses completely, but it is not a clean fix for every seed (v2 s5 sampled:
mean 0.75).

**Two controller modes** ([`raw/altitude_profile.txt`](../analysis/collapse_diagnosis/raw/altitude_profile.txt),
greedy, 20-30 episodes per checkpoint; CONFIRMED):

- *Healthy mode* (v2 s3 final, v2 s2 final): MAIN is modulated by altitude (v2 s3: share 0.23 above 60 m,
  0.74 at 30-60 m, 0.59 at 10-30 m), NOOP is 5-32% of steps, descent is 14-16 m/s above 30 m (v2 s2:
  12.7 m/s above 60 m and 8.8 m/s at 30-60 m), 318 steps (v2 s3) and 425 steps (v2 s2). Their sampled training episodes use 0.35 and
  0.48 of the tank at the end of training.
- *Hover-clock mode* (v2 s8 @1150 and @1200, v2 s7 @1100; v2 s7 @1050 shares the zero NOOP share but
  descends faster): NOOP is never the argmax above 30 m (share 0.00), MAIN is a flat 0.49-0.52 at every
  altitude above 10 m with LEFT/RIGHT filling the other half (`LRMMLRMM...`), descent is 5.8-7.3 m/s above 60 m, 3.2-4.2 m/s
  at 30-60 m and 1-2 m/s below 30 m, 677-707 steps in total. The sampled version of the same checkpoint
  chooses NOOP on 9-12% of steps above 30 m and descends faster (v2 s8 @1200 above 60 m: greedy 5.8 m/s,
  sampled 8.1 m/s; steps spent above 30 m: greedy 333, sampled 233).
- Arithmetic: a change of 0.01 in MAIN duty is a vertical acceleration of 0.2 m/s^2 (19.62 x 0.01), about 1 m/s
  every 5 s. In hover-clock mode a few percent of change in a nearly tied argmax decides between landing
  and running out of fuel.

## 5. The argmax is nearly tied

Source: [`raw/argmax_margins.txt`](../analysis/collapse_diagnosis/raw/argmax_margins.txt). CONFIRMED.

- Along each checkpoint's own greedy trajectories (20 episodes), the share of steps whose top-1 minus
  top-2 action probability is below 0.1 is 0.58-0.64 for v2 s8 @1000/@1150/@1200, v2 s7 @1050/@1100, v2 s3
  final and v2 s2 final, and 0.48 for v2 s1 final. Mean maximum probability is 0.38-0.46 and mean entropy
  1.18-1.29 nats. Healthy and collapsed checkpoints look the same on these statistics.
- Between two checkpoints 50 updates apart, the greedy action changes in 33-72% of the states visited by the
  earlier checkpoint (v2 s8 1150 to 1200: 33%; 1100 to 1150: 46%; 1000 to 1050: 55%; v2 s7 1050 to 1100: 47%;
  1100 to 1150: 72%). A single greedy checkpoint is effectively a draw from many nearly tied controllers.

## 6. Why does the policy sit on the cliff? (mostly SUSPECTED)

- **Speed is almost unrewarded (CONFIRMED arithmetic, SUSPECTED consequence).**
  [`raw/discounted_returns.txt`](../analysis/collapse_diagnosis/raw/discounted_returns.txt) (40 sampled
  episodes, seeds 60000-60039): v2 s3 final (234-step landings) has a raw return of 100.6, v2 s8 final
  (662-step landings) 96.6. The per-step costs are small against a +100 landing: time 0.01, a full MAIN
  burn 0.03. With gamma 0.999 and reward_scale 0.01 the scaled discounted returns are +0.80 and +0.49, but
  finishing one step earlier is worth only about 0.1% of the value, which is likely below the noise of the
  policy-gradient estimate. Late in training the raw GAE advantage standard deviation is 0.004-0.007 (v2 s8,
  checkpoints 1000 to final) against 0.033 at v2 s3 checkpoint 300
  ([`raw/advantage_scale.txt`](../analysis/collapse_diagnosis/raw/advantage_scale.txt); 256-step batches
  from fresh starts, which contain no terminal events); `norm_adv` rescales it to 1, so late updates are
  dominated by critic error and rare terminal events (SUSPECTED contributor).
- **Neutral drift towards the cliff, then an absorbing failure (CONFIRMED description).** In v2 s8 the
  sampled training episodes' fuel use rose from about 0.4 to about 0.95 in 80 updates at 100% success
  (section 2). In v1 s5
  ([`raw/window_hl_v1_s5_590_760.txt`](../analysis/collapse_diagnosis/raw/window_hl_v1_s5_590_760.txt)) the
  sampled success was 77-100% (mostly 100%) over updates 602-662 while fuel use wandered 0.42 (update 590),
  0.76 (608), 0.91 (614), 0.68 (620), 0.85 (656), 0.77 (662); by update 686 every episode is
  hover-until-empty (length 718-798, return about -112, `value_loss` 0, explained variance 1.00, entropy
  still about 1.0) and it never recovered in the remaining 530 updates.
  With no landing episode in any batch there is no signal to climb back (SUSPECTED explanation). v1 s8
  shows the same pattern (first 90% evaluation at update 540, lost for good). Across the 12 learned seeds,
  fuel use of the training episodes and policy entropy move together
  ([`raw/entropy_fuel_drift.txt`](../analysis/collapse_diagnosis/raw/entropy_fuel_drift.txt): median
  Spearman +0.42 over 20-update blocks, positive in 10 of 12 runs), which is compatible with the entropy
  bonus pushing towards the flat hover-clock mode (SUSPECTED).
- **Entropy is correlated with collapse (CONFIRMED correlation, SUSPECTED causation).** Share of in-training
  evaluations with greedy success below 20% after the first 90% evaluation:

  | run | collapsed share | run | collapsed share |
  |---|---|---|---|
  | v2 s3 | 3% | v2 s7 | 17% |
  | v2 s2 | 10% | v2 s4 | 21% |
  | v2 s0 | 14% | v2 s8 | 22% |
  | v2 s1 | 15% | v2 s9 | 39% |
  | v2 s5 | 15% | v1 s1 / s2 / s3 | 2% / 6% / 30% |

  Spearman correlation with mean policy entropy after learning: +0.81 (n=12; +0.77 for the nine v2
  seeds), with greedy fuel use in good evaluations +0.66 (n=12). Learned seeds end with entropy 0.90-1.06
  although `ent_coef` is 0.03 and the task only needs a few distinct actions, which fits an entropy bonus
  spreading probability over equivalent actions (SUSPECTED).
- **v2 s6 (CONFIRMED description).** By update 200 to 250 all 16 environments return about -114 to -118
  (episode length 770-850), `value_loss` is 0.0000 and explained variance 1.00
  ([`raw/window_hl_v2_s6_80_280.txt`](../analysis/collapse_diagnosis/raw/window_hl_v2_s6_80_280.txt)). Before
  that it had batches with success 0.38 (update 120) and 0.33 (update 140). The sampled discounted
  return of its checkpoint 100 is -0.31 (4 of 40 episodes land) against -0.52 to -0.59 for checkpoints 150
  to final (checkpoint 50 was already a hover policy at -0.55), so settling into hover did not improve the
  PPO objective: it is an absorbing state with no learning signal, not a better optimum
  ([`raw/discounted_returns.txt`](../analysis/collapse_diagnosis/raw/discounted_returns.txt)). Entropy drifted from about
  1.0 to 0.55 despite the entropy bonus. The learned v2 seeds sit near 1.0 in the same period, but the v1
  non-learners are mixed (v1 s9 is 1.04 at update 600), so an entropy explanation of the hover seeds is weak
  (SUSPECTED).

## 7. Persistence and exchangeability of collapses

Source: [`raw/persistence.txt`](../analysis/collapse_diagnosis/raw/persistence.txt). CONFIRMED. v2 seeds that
learned, in-training evaluations every 10 updates, after each run's first evaluation at 90% or more:

- **Base rate.** 15-16% of evaluations are collapsed. By stage: 17% (updates up to 800), 26% (801-900), 18%
  (901-1000), 6% (1001-1100), 13% (1101-1220). The rate does not rise late; learning-rate annealing neither
  creates the collapses nor prevents them.
- **Persistence.** P(collapsed 10 updates later | collapsed now) = 0.54, against 0.09 if the evaluation is
  fine now. At 50 updates: 0.26 against 0.13. At 100 updates: 0.22 against 0.13. Checkpoints 50 updates apart
  are therefore only weakly dependent.
- **Run lengths** (consecutive collapsed evaluations): v2 s0 [4,1,4], s1 [1,1,4,1,2], s2 [1,2,1,1,1],
  s3 [1,1], s4 [1,2,2,4,1,1,4], s5 [2,2,3,1,1], s7 [1,3,3,1,4,1], s8 [2,1,1] plus 7 still collapsed at the end of
  the run, s9 [1,7,6,2,1]. Typical length is 10-40 updates.
- **Fuel use is a contemporaneous indicator, a weak predictor.** Among evaluations at 90% or more, the next
  evaluation is collapsed with probability 1-5% when fuel use is below 0.8, 14% at 0.8-0.9 and 33% at 0.95
  or more (n=21).
- **Frozen end states.** v2 s8 ends in a collapsed draw (and v2 s4 in a partly collapsed one) and cannot
  repair it because the learning rate is about 1e-5 or less. Earlier collapses do recover while the learning
  rate is higher (v2 s8's collapse around updates 930-970 recovered at about 6e-5). That a nonzero learning-rate
  floor would let PPO repair a late collapse is SUSPECTED, untested.
- v1 s5 and v1 s8 learned (first 90% evaluation at update 620 and 540) and then lost the policy for good, so
  "v1 s4-s9 never learned" would be wrong for them. Only v1 s4, s6, s7 and v2 s6 never exceeded 50% on any
  in-training evaluation.

## 8. What was ruled out

- **Entropy collapse** (RULED OUT): entropy is about 1.0 throughout; it is high, not low.
- **Value-function divergence** (RULED OUT as a cause): explained variance 0.96-0.999 and `value_loss` about
  1e-4 in the collapse windows.
- **Large policy updates at the moment of collapse** (RULED OUT): `approx_kl` about 1e-4, `clip_frac` about 0
  and learning rate at most 2e-5 in v2 s8 at updates 1140-1220.
- **Late learning-rate annealing as the cause** (RULED OUT as a cause; SUSPECTED as the reason the final
  state is frozen): collapse frequency is flat across stages (section 7).
- **Small raw advantage scale as a lever** (RULED OUT): `norm_adv=True` already rescales per minibatch, so
  return or advantage normalisation does not address this. The noise amplification in section 6 is a
  separate, SUSPECTED contributor.

## 9. Recommended fixes

Expected effects are SUSPECTED unless stated; none was tested in training.

1. **Report and select with both greedy and sampled success, and require a contiguous good window of
   checkpoints.** No training cost. CONFIRMED effect on the numbers: sampled success of the v2 s8 final policy is
   97-98% instead of 0%.
2. **Late entropy decay (for example `ent_coef` from 0.03 to 0.005 or less over the last 40%) together with a
   learning-rate floor**, so the trained policy converges to its own greedy policy and PPO sees and can
   correct the fuel-out crashes. Expected: a gap between greedy and sampled success of 5 points or less at
   every checkpoint from update 800.
3. **A fuel- or time-aware reward, as a new reward version (do not modify v2)**: a bonus proportional to the fuel
   remaining at landing, or a larger time penalty, or a lower discount (for example 0.995), so a 700-step
   creep is clearly worse than a 250-step landing. Expected: greedy mean fuel use of 0.6 or less and mean
   steps of 450 or less (the healthy seeds), and some chance to lift the hover seeds out of the no-signal
   state.
4. **For the absorbing hover seeds a late fix cannot help**; an early lever is needed (a higher early
   entropy with a decay schedule, or a penalty or termination for stalling with the tank empty at altitude).
   Weak evidence.
5. **Not supported by the evidence:** raising `ent_coef`, return, value or advantage normalisation, a longer
   learning-rate anneal.
6. **Monitoring targets** (proposals, not validated): evaluation mean fuel use (0.6 or less is healthy, 0.9 or
   more is near the cliff), the gap between greedy and sampled success, and the share of steps with a top-1
   minus top-2 margin below 0.1 (58-64% now; 25% or less as a target).

## 10. Limitations

- 100 episodes per rollout cell (20-30 for the per-trajectory profiles), on the headline's held-out seeds;
  the comparison of controller modes (section 4) rests on v2 s8, s7, s3 and s2 only.
- Correlations are over 9 to 12 seeds and descriptive.
- All mechanisms marked SUSPECTED need a training run to be confirmed; no training was done for this
  analysis.
- This analysis covers the fixed-scale headline runs. The earlier running-normaliser run quoted in the README
  (`v2_s0`, entropy rising and value loss growing by four orders of magnitude) was not re-analysed and may
  have a different mechanism.
- The headline numbers in [headline-results.md](headline-results.md) are unchanged by this analysis.
