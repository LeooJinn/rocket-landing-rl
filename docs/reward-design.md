# Reward design

Implemented in `src/rocket_landing/rewards.py`, tested in `tests/test_rewards.py`.
`compute_reward(prev_state, state, action, outcome, *, version="v1")` is called once per
step after physics and returns a `RewardBreakdown(total, terms)`; `total == sum(terms.values())`.
Weights live in the frozen dataclass `RewardV1Config` (defaults define v1).

## Reward v1

Per-step terms (keys of `terms`):

| Term | Formula | Default |
|---|---|---|
| `shaping` | `phi(state) - phi(prev_state)` | see below |
| `fuel` | `-w_fuel * (prev.fuel - state.fuel) / 0.0025` | `w_fuel = 0.03` |
| `time` | `-0.01` every step | |
| `terminal` | outcome-dependent, only on that step | LANDED +100, CRASHED -100, OUT_OF_BOUNDS -100, TIMEOUT -50 |
| `precision` | LANDED only: `20 * centred * gentle` | max +20 |

Potential: `phi(s) = -(dist/10 + speed/10 + |angle|)` with `dist = sqrt(x^2+y^2)` (m),
`speed = sqrt(vx^2+vy^2)` (m/s), `angle` in rad. So 10 m of distance, 10 m/s of speed and
1 rad of tilt each cost one unit. A typical start (100 m away, ~7 m/s) has phi about -11,
so total shaping over an episode is about +11, small next to the +100 landing.

Fuel: 0.0025 is the fuel fraction of one full main-engine step (0.05/s * 0.05 s), so a
main burn costs 0.03 per step and a side thruster (0.0005) about 0.006. Fuel is measured
from the state difference, not the action.

Precision: `centred = max(0, 1 - |x|/10)`; `gentle = clip(1 - max(|vx|/1, |vy|/2, |angle|/0.2, |omega|/0.5), 0, 1)`.
It is 0 at the edge of the pad or at any safe-landing limit, so every safe landing still
gets at least +100.

## Rationale

- Dense shaping gives a learning signal long before a first landing.
- A large terminal reward defines the real goal; shaping and costs are small beside it.
- Time and fuel costs prefer efficient landings but are far smaller than the crash/landing gap.
- Crash (-100) is strictly below every safe landing (>= +100 terminal), including an off-centre one.

## Why shaping cannot be farmed (hovering)

Shaping is a difference of a potential, so it telescopes: over any state sequence
`s0..sT` the shaping sum is `phi(sT) - phi(s0)`. Hovering, oscillating or circling and
returning to a state earns zero net shaping, while time and fuel costs still accrue.
Hovering to timeout therefore yields about `-0.01*1000 - fuel - 50`, strictly worse than
landing. Shaping also does not change which policies are optimal beyond the terminal terms
(Ng et al., 1999), apart from the finite-horizon caveat below.

## Known limitations and risks

- The potential rewards being near the pad with low speed even when airborne; the
  agent could prefer stopping just above the pad. The per-step time/fuel costs and the
  -50 timeout make this unprofitable but not impossible with a sparse-reward learner.
- Potential-based invariance assumes `gamma = 1` and fixed terminal potential. With
  `gamma < 1` shaping is no longer exactly neutral; with a timeout ending the episode, a
  state's potential is "banked" rather than discounted.
- The distance term uses `y` (height), so descending is rewarded even when badly
  off-centre horizontally; horizontal and vertical error are not weighted separately.
- Crashing right next to the pad costs 100 less than a good landing's terminal but gains
  shaping on the way down; shaping is bounded (about 11) so it never outweighs the crash penalty.
- Out-of-bounds and crash share -100, so the agent is not told which failure is worse.
- Fuel penalty is small; a policy may burn fuel freely if it improves landing odds. That is intended.
- Weights are untuned against trained agents; revisit after the first training runs.

## Reward v2

v2 is v1 with exactly one change: a CRASHED touchdown earns a softness bonus, reported as
its own term `crash_softness` (0.0 on every non-crash step) so `terminal` stays comparable
to v1. Terms are the v1 keys plus `crash_softness`; all other terms and weights are
identical. Config: frozen `RewardV2Config` (subclass of `RewardV1Config`) adds
`crash_softness_bonus = 50.0` and `excess_zero = 5.0`. Run with `reward_version="v2"`;
see `experiments/configs/ppo_v2.json`.

| Term | Formula | Default |
|---|---|---|
| v1 terms | unchanged | see Reward v1 |
| `crash_softness` | CRASHED only: `50 * softness` | 0 to +50 |

On the touchdown `state`:

```
excess   = max(|x|/10, |vx|/1, |vy|/2, |angle|/0.2, |omega|/0.5)
softness = clip((5 - excess) / (5 - 1), 0, 1)
```

So `excess <= 1` (within every safe limit) gives the full +50, `excess >= 5` gives 0, linear
between. The CRASHED return from terminal + bonus is in [-100, -50]; a free-fall crash
(vy = -44, excess 22) gets the full -100.

### Motivation

PPO training with v1 repeatedly converged to "free-fall and crash immediately" (~85 steps,
~0 fuel, return ~-96.5). The CRASHED terminal is a flat -100, so a fast crash after falling
collects net positive distance shaping and avoids fuel/time cost; any careful-but-failed
approach scores worse than suicide. The only signal for impact softness was the small speed
term in the potential. A second attractor (hover until fuel runs out, then crash, ~800
steps, return ~-115) comes from discounting: with gamma < 1 the banked shaping and the
far-off timeout/crash penalty are discounted, so delaying failure looks cheap.

v2 makes crash terminal depend on impact softness, giving a gradient from "crash hard" toward
"almost landed" so careful approaches beat free-fall.

### Invariants (tested in `tests/test_rewards_v2.py`)

- v2 equals v1 on every non-crash step (in progress, landed, out of bounds, timeout).
- Worst crash (-100) is unchanged; best crash terminal is -50, equal to the timeout terminal,
  so hovering to timeout is never preferred over a near-landing.
- Every v2 crash is strictly below every safe landing's terminal (>= +100); the gap is >= 150.

### New risks

- The agent may settle for a soft crash near the pad (-50) instead of landing. Landing still
  dominates by >= 150 (best crash -50 vs >= +100), but a sparse-reward learner may find the
  soft-crash plateau first and stay there.
- The bonus uses the max over normalised errors, so improving a non-worst component gives no
  gradient; only the worst offender matters.
- Softness is evaluated at touchdown only; it says nothing about how the agent got there.
- Soft crash (-50) ties with timeout (-50) at the terminal; time/fuel costs and the
  discounted timeout keep a crash preferable only if it actually happens earlier. Watch for
  hover-then-soft-crash behaviour.
- With gamma < 1 shaping is not exactly neutral (see v1 limitations); v2 does not fix that.

### Result

With the chosen PPO settings (gamma 0.999, ent_coef 0.03, 5M steps, seeds 0-5) v2 was not
better than v1 with the running observation normaliser; see [experiments.md](experiments.md). With
fixed-scale normalisation and 10 seeds per arm, v2 had the higher mean held-out success (77.0% vs 30.0%),
but the evidence is suggestive, not conclusive (permutation p = 0.07); see
[headline-results.md](headline-results.md).

## Reward v3 (v3a, v3b)

v3 is v2 plus two efficiency terms, aimed at the failure mode found in the 20 headline runs
(`docs/headline-results.md`): fuel exhaustion before touchdown. Two registered versions
share one implementation (`_reward_v3`) and differ in one weight; both keep the v2
terminal/crash structure and every v2 term (`crash_softness` etc.) unchanged.
Run with `reward_version="v3a"` or `"v3b"`; configs `experiments/configs/ppo_final_v3a.json`
and `ppo_final_v3b.json` are `ppo_final_v2.json` with only `reward_version` changed.
(There is no plain `"v3"` string: the pilot compared two variants, so each is its own
immutable version.)

| Term | v3a | v3b |
|---|---|---|
| v1/v2 terms | unchanged, except `time` is `-0.03` per step (v2: `-0.01`) | same |
| `landing_fuel` | LANDED only: `40 * fuel_left` (touchdown state), 0 to +40 | same |
| `fuel_out` | always 0 | CRASHED with `fuel <= 0` at touchdown: `-25` |

Config dataclasses: `RewardV3Config(RewardV2Config)` with `time_penalty=0.03`,
`landing_fuel_bonus=40.0`, `fuel_out_crash_penalty=0.0` (v3a); `RewardV3BConfig` sets
`fuel_out_crash_penalty=25.0` (v3b). `compute_reward` type-checks the config per version.

### Motivation (numbers from the headline analysis)

- Late 'collapses' (e.g. `hl_v2_s8`) are greedy-only: the greedy policy is a slow hover-clock
  (MAIN duty flat near 0.5 at every altitude, about 700 steps, whole tank used, empty 1-4 m
  above the ground), whereas healthy policies land in about 230-320 steps using about 0.35 of
  the tank. Never-learning seeds sit in a "hover until empty at altitude" state.
- Nothing in v2 prices the slow mode: a 234-step landing returned 100.6 and a 662-step landing
  96.6 (time 0.01/step, a full burn 0.03/step, terminal +100, `gamma = 0.999`,
  `reward_scale = 0.01`). The whole tank is worth only 400 full-burn steps * 0.03 = 12 points in v2,
  and 450 extra steps cost 4.5.
- MAIN burns 0.0025 of the tank per step, LEFT/RIGHT 0.0005.

### What v3 changes (numbers)

Scripted behaviours rolled out in the real environment (100 episodes, seeds 50000-50099;
`scripts/reward_scenarios.py`, raw numbers in `docs/reward_v3_scenarios.json`); the dynamics
and therefore outcomes, steps and fuel are identical in every column:

| behaviour | landed | steps | fuel used | return v2 | return v3a | return v3b |
|---|---|---|---|---|---|---|
| heuristic (fast, frugal) | 100% | 273 | 0.37 | 111.5 | 131.3 | 131.3 |
| heuristic `descent_decel=0.5` | 100% | 369 | 0.49 | 108.4 | 121.4 | 121.4 |
| heuristic `v_max=4` | 99% | 480 | 0.63 | 104.5 | 109.6 | 109.6 |
| heuristic `descent_decel=0.2` (slow) | 100% | 547 | 0.71 | 104.8 | 105.2 | 105.2 |
| heuristic `descent_decel=0.1` (slowest, often runs dry) | 73% | 718 | 0.93 | 49.4 | 38.0 | 31.2 |
| hover clock (hold altitude until empty, then fall) | 0% | 852 | 1.00 | -113.7 | -130.7 | -155.7 |
| free fall (no action) | 0% | 82 | 0.00 | -96.5 | -98.2 | -98.2 |

- The spread between the fast frugal landing and the slow landings grows from 7 points in v2
  (111.5 vs 104.8) to 26 points in v3 (131.3 vs 105.2); against the 718-step run-dry behaviour
  it is 62 (v2) vs 93 (v3a) and the ordering fast > medium > slow is monotone.
- Synthetic landing, same start and touchdown pose: 250 steps with 65% fuel left returns 47
  points more than 700 steps with an empty tank under v3 (v2: 12). Per step of full burn
  the marginal cost of fuel is `0.03 + 40 * 0.0025 = 0.13` when the episode ends in a landing
  (v2: 0.03), and every extra step costs 0.03 (v2: 0.01).
- Hover-until-empty then falling is now clearly worse than the free-fall crash (v3a -130.7 vs
  -98.2; v3b -155.7), and much worse than any landing. v2 separated the two by only 17.

### Invariants and exploit checks (tested in `tests/test_rewards_v3.py`)

- With `time_penalty=0.01` and both new weights 0, v3a/v3b equal v2 exactly (plus two zero terms).
- Landing bonus is linear in the fuel left and only on LANDED; the fuel-out term is only on a
  CRASHED touchdown with an empty tank and only in v3b.
- Efficiency ordering: the return of a landing is strictly decreasing in steps and in fuel used.
- Terminal-level landing-vs-crash gap stays >= 150 (the new terms only add to landings, or
  subtract from crashes); even the slowest, emptiest, worst safe landing returns more than 100
  points above the best (just too fast) crash, so no early/soft crash beats landing.
- Early crash (20-150 steps, hard impact) is at least 150 below a careful 700-step landing,
  so the larger time penalty does not reward crashing early.
- Diving to save fuel (free fall, or a touchdown just over the limits with the whole tank left)
  is at least 150-200 below the efficient landing.
- Hover to timeout (burning or not burning) is at least 150 below landing; hover-until-empty
  then a hard crash is below the free-fall crash.
- v3b only: landing > soft crash with fuel > soft crash with an empty tank (at the same impact).
- Return is bounded: the largest landing return is about 100 + 20 + 40.
- The heuristic controller produces identical outcomes/steps/fuel under v2, v3a, v3b (the reward
  never touches the dynamics), lands at least 95% of 40 held-out episodes (100% of the 100 in the table above), and its v3 return equals the v2
  return minus 0.02 per step plus `40 * fuel_left`. The scripted ladder above ranks by efficiency.

### Risks

- Pressure to finish early may push a learner toward a harder, faster descent. A touchdown just
  over the limits is still a crash worth at most -50 minus time cost, i.e. at least 150 below a
  landing, so this is a learning-difficulty risk, not a reward exploit.
- The landing bonus is sparse and paid at the end; with `gamma = 0.999` it is nearly undiscounted,
  but credit assignment for 'save fuel' still relies on the critic reading the `fuel` observation.
- A larger time penalty and (v3b) fuel-out penalty make hovering worse relative to a quick
  crash: for a learner that has not yet found landing, 'free-fall and crash' (-98) is now
  relatively more attractive than 'hover until empty' (-131 / -156). v2's crash-softness bonus
  is the gradient away from it; whether that suffices is an empirical question (see
  `docs/reward-v3-results.md`).
- v3b's penalty (25) makes an empty-tank crash at about 6 m/s as bad as a free-fall crash.
- Returns are not comparable across reward versions; compare held-out outcomes, fuel and steps.

### Result

Pilot on seeds 10-12 (v3a, v3b, v2 control) chose v3a; confirmation on seeds 0-9 against the
headline `hl_v2` runs: greedy held-out success 500/500 on all 10 seeds for v3a vs 7/10 healthy
(two at 0%) for v2, paired difference +23 points but only suggestive (sign-flip p = 0.062, 5 seeds
better, 5 tied at the ceiling); fuel used and episode length clearly lower (landed-only: 0.53 vs 0.71
of the tank, 362 vs 486 steps); sampled-action success not clearly better (p = 0.35). Details,
numbers and caveats: `docs/reward-v3-results.md`.

## Adding a new version

1. Add a function `_reward_vN(prev_state, state, action, outcome, cfg)` returning the
   term dict (and a config dataclass if weights differ).
2. Register it in `_DISPATCH` / `_CONFIGS` and append the version string to `REWARD_VERSIONS`.
3. Never change v1 semantics or defaults in place; results recorded under "v1" must stay reproducible.
4. Add tests mirroring `tests/test_rewards.py` (unintended-incentive checks) for the new version
   and record the comparison in `docs/experiments.md`.
