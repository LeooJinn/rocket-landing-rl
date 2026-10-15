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

## Adding a new version

1. Add a function `_reward_v2(prev_state, state, action, outcome, cfg)` returning the
   term dict (and a config dataclass if weights differ).
2. Register it in `_DISPATCH` and append `"v2"` to `REWARD_VERSIONS`.
3. Never change v1 semantics or defaults in place; results recorded under "v1" must stay reproducible.
4. Add tests mirroring `tests/test_rewards.py` (unintended-incentive checks) for the new version
   and record the comparison in `docs/experiments.md`.
