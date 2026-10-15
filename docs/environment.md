# Environment

Implemented in `src/rocket_landing/environment.py` (`RocketLandingEnv`, `EnvConfig`) with
shared types in `state.py`. Gymnasium-compatible; also registered as `RocketLanding-v0`.
Units are metres, seconds and radians throughout.

## Observation space

`Box(float32, shape=(7,))`, raw physical units, order fixed by `OBSERVATION_FIELDS`:

| Index | Field | Unit | Meaning | Space bounds |
|---|---|---|---|---|
| 0 | `x` | m | horizontal offset from pad centre | [-200, 200] |
| 1 | `y` | m | height of the base above ground | [0, 400] |
| 2 | `vx` | m/s | horizontal velocity | [-200, 200] |
| 3 | `vy` | m/s | vertical velocity (up positive) | [-200, 200] |
| 4 | `angle` | rad | tilt, 0 upright, counter-clockwise positive, wrapped to [-pi, pi] | [-pi, pi] |
| 5 | `angular_velocity` | rad/s | rate of change of `angle` | [-200, 200] |
| 6 | `fuel` | fraction | remaining fuel, 1 = full | [0, 1] |

Bounds are deliberately generous (2x the world) so every reachable state is inside. The
observation is not normalised; a learner should scale it itself.

## Action space

`Discrete(4)`: `Action.NOOP=0`, `MAIN=1`, `LEFT=2`, `RIGHT=3`. Anything else (floats, bools,
out-of-range, non-scalar arrays) raises `ValueError`. Each maps to a continuous control pair
`(main_throttle in [0,1], side_throttle in [-1,1])` via `action_to_controls`:

| Action | main_throttle | side_throttle |
|---|---|---|
| NOOP | 0 | 0 |
| MAIN | 1 | 0 |
| LEFT | 0 | +1 (positive angular accel, counter-clockwise) |
| RIGHT | 0 | -1 (clockwise) |

`simulate_step(state, main_throttle, side_throttle, config)` already takes the continuous
pair, so a continuous action space can be added without touching the physics. Only one
action fires per step: the rocket cannot thrust and rotate in the same step.

## Physics

Body up vector is `(-sin a, cos a)` for angle `a`. With main throttle `m` and side throttle `s`:

```
ax    = -sin(a) * main_engine_accel * m
ay    =  cos(a) * main_engine_accel * m - gravity
alpha = side_engine_angular_accel * s
vx' = vx + ax*dt;  vy' = vy + ay*dt;  omega' = omega + alpha*dt      (velocities first)
x'  = x + vx'*dt;  y'  = y + vy'*dt;  a'     = wrap(a + omega'*dt)   (then positions, using new velocities)
```

This is semi-implicit (symplectic) Euler with fixed `dt = 0.05 s` (20 Hz). Defaults:
gravity 9.81 m/s^2, `main_engine_accel` 19.62 m/s^2 (thrust-to-weight 2), side angular accel
1 rad/s^2. Because the nose tilting toward +x is a negative angle, moving toward +x needs a
negative angle.

## Fuel model

Fuel is a fraction in [0, 1]. Burn per step is `(m*main_fuel_rate + |s|*side_fuel_rate)*dt`,
with `main_fuel_rate = 0.05/s` (20 s of continuous main burn) and `side_fuel_rate = 0.01/s`.
If the request exceeds what remains, both throttles are scaled down so the tank ends
exactly empty. With an empty tank engines produce nothing. Fuel exhaustion is not terminal:
the rocket keeps falling and `info["fuel_exhausted"]` becomes True.

## Initial state

Uniform sampling from the seeded RNG (ranges in `EnvConfig`): `x` in [-40, 40], `y` in
[80, 120], `vx` in [-5, 5], `vy` in [-10, 0], `angle` in [-0.2, 0.2], `angular_velocity` in
[-0.1, 0.1], fuel 1.0. `reset(options={"state": RocketState(...)})` overrides sampling
(requires `y > 0`), useful for tests.

## Termination and truncation

Checked in this order after each step (first match wins):

1. `y <= 0` (touchdown): `terminated`. `y` is clamped to 0. Outcome is `LANDED` if on the pad
   and all landing thresholds hold, otherwise `CRASHED`.
2. `|x| > world_half_width (100)` or `y > world_height (200)`: `terminated`, `OUT_OF_BOUNDS`.
3. `steps >= max_steps (1000)`: `truncated`, `TIMEOUT`.

Stepping after the episode ends raises `RuntimeError` until `reset()`.

## Landing thresholds

At the first ground contact: `|x| <= 10` (pad half width), `|vx| <= 1 m/s`, `|vy| <= 2 m/s`,
`|angle| <= 0.2 rad`, `|angular_velocity| <= 0.5 rad/s`. All must hold.

## Info dict

Returned by `reset` (first four keys) and `step` (all):

| Key | Meaning |
|---|---|
| `outcome` | `Outcome` value string: `in_progress`, `landed`, `crashed`, `out_of_bounds`, `timeout` |
| `steps` | steps taken this episode |
| `fuel_exhausted` | fuel <= 0 |
| `state` | the full `RocketState` |
| `reward_terms` | (step only) per-term reward breakdown |
| `fuel_used` | (step only) fuel fraction burned this step |
| `landing` | (step only, touchdown only) dict with `vx`, `vy`, `angle`, `angular_velocity`, `x` |

## Determinism and seeding

The physics has no randomness. The only RNG use is initial-state sampling through
Gymnasium's `np_random`, so `reset(seed=s)` followed by the same actions reproduces the same
trajectory, observations and rewards exactly.

## Reward version

`RocketLandingEnv(config, reward_version="v1")` selects the reward function; unknown versions
raise `ValueError`. See [reward-design.md](reward-design.md).

## Known simplifications

- No aerodynamic drag, wind or sensor/actuation noise.
- Point mass with rotation; no mass change as fuel burns, no centre-of-mass or gimbal effects.
- Main engine is on/off at full throttle in the discrete space.
- No landing legs or contact dynamics: touchdown is judged at the first step with `y <= 0`
  and the episode ends immediately; the rocket does not bounce or tip over.
- Because of the discrete step, the touchdown state is the first state past the ground
  (up to one step of travel below y = 0 is clamped away).

## Baseline controller

`src/rocket_landing/baseline.py` holds `HeuristicController`, a hand-written, non-learning,
deterministic cascade used to validate the environment (it shows the task is solvable and
that termination, thresholds and rewards behave sensibly):

1. Horizontal: target `vx = clip(-kx*x, +-vx_max)`; a P law on the `vx` error gives a desired
   horizontal acceleration, converted to a desired tilt `-ax/g`, clipped to a limit that
   shrinks from `max_tilt` to `min_tilt` below `tilt_fade_height` metres.
2. Attitude: PD on angle error and angular velocity; if its desired angular acceleration
   exceeds a dead band, fire LEFT or RIGHT.
3. Otherwise vertical: fire MAIN if `vy` is below an altitude-dependent target descent speed
   `-clip(sqrt(2*decel*y), v_touch, v_max)`, else NOOP.

Gains (`DEFAULT_GAINS`) were tuned by a simple random hill-climb on 40-80 episodes from seeds
1000+; the numbers below use seeds 0-99, which were not used for tuning. No env thresholds,
distributions or rewards were changed for it.

Command: `uv run python scripts/evaluate_baseline.py --episodes 100 --seed 0`

```
Heuristic baseline  seeds 0..99  reward v1
  episodes                              100
  success rate                        99.0%
  crash rate                           1.0%
  out-of-bounds rate                   0.0%
  timeout rate                         0.0%
  mean reward                        109.29
  mean fuel used                      0.365
  mean steps                          269.5
  mean touchdown |vx| (m/s)           0.404
  mean touchdown |vy| (m/s)           1.137
  mean touchdown |angle| (rad)        0.033
Failure breakdown
  crashed 1, out_of_bounds 0, timeout 0
  crash threshold violations (an episode may violate several):
    off-pad  0
    vx       1
    vy       0
    angle    0
    spin     0
  crashed seeds: 79
```

The single failure (seed 79) landed on the pad but with horizontal speed just over 1 m/s.
Success rate is an estimate from 100 episodes, not a guarantee.
