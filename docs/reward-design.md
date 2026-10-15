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

## Adding a new version

1. Add a function `_reward_v2(prev_state, state, action, outcome, cfg)` returning the
   term dict (and a config dataclass if weights differ).
2. Register it in `_DISPATCH` and append `"v2"` to `REWARD_VERSIONS`.
3. Never change v1 semantics or defaults in place; results recorded under "v1" must stay reproducible.
4. Add tests mirroring `tests/test_rewards.py` (unintended-incentive checks) for the new version
   and record the comparison in `docs/experiments.md`.
