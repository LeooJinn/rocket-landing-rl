# Rocket Landing RL

A reinforcement-learning project that teaches a simulated rocket to land safely on a designated pad using only observations from the physics simulation and a reward function.

The agent is not given a scripted sequence of thrust commands. It learns by attempting repeated landings, receiving feedback from the environment, and improving its policy over time.

## Project goals

- Build or adapt a 2D rocket-landing physics environment with gravity, thrust, fuel, position, velocity, and tilt.
- Define a compact observation space containing the rocket's position, velocity, angle, angular velocity, fuel, and pad-relative state.
- Train a neural-network policy with reinforcement learning.
- Use reward shaping to encourage controllable, fuel-efficient, upright landings rather than hard-coding how to fly.
- Visualize training and replay trained episodes in a browser.
- Measure learning progress with reproducible success, crash, fuel, and reward statistics.

## Planned stack

- **Python** for the environment, training loop, evaluation, and experiment tooling
- **NumPy** for numerical operations and environment state calculations
- **PyTorch** for the policy/value networks and reinforcement-learning training
- **Gymnasium-compatible API** for reset, step, observations, actions, rewards, and episode termination
- **HTML/CSS/JavaScript** for the browser visualization and trained-episode replay
- **Matplotlib or equivalent** for offline training curves and experiment reports

TensorFlow is intentionally not part of the initial stack. Using one deep-learning framework keeps the training code focused and makes the experiment easier to reproduce. TensorFlow can be evaluated later as a separate comparison, not added only for the sake of listing another library.

## Environment model

The initial environment will be a 2D landing task with:

- Gravity and a fixed simulation timestep
- Main or vertical thrust
- Directional thrust for horizontal control and rotation
- Fuel consumption based on thrust usage
- Position, velocity, angle, and angular velocity
- A landing pad with a configurable width
- Safe-landing thresholds for speed, tilt, and horizontal position
- Terminal states for safe landing, crash, fuel exhaustion, or timeout

The first version will prioritize a correct and understandable simulator over visual complexity. The browser will render the state produced by the simulator rather than becoming a second, inconsistent physics implementation.

## Reward design

The reward function will be versioned and documented as an experiment. It may include:

- Positive reward for reducing distance to the landing pad
- Positive reward for reducing horizontal and vertical speed near the pad
- Positive reward for maintaining an upright orientation
- Small fuel-efficiency incentives
- A successful-landing bonus
- Penalties for unnecessary thrust, excessive tilt, timeout, and unsafe impact
- A large crash penalty

Reward shaping is a training signal, not a guarantee of good behavior. Each reward term will be tested for unintended incentives such as hovering indefinitely, wasting fuel, or learning to approach the pad without actually landing.

## Training and evaluation

Training will run for repeated episodes. Each experiment will record:

- Episode reward
- Success and crash rate
- Average landing speed and tilt
- Fuel consumed
- Episode length
- Evaluation performance on held-out starting conditions

A trained policy will be evaluated separately from training so that success is not measured only on the states it has already seen.

## Browser visualization

The frontend will provide:

- Live or replayed rocket trajectory
- Landing pad and environment state
- Thrust commands selected by the policy
- Current position, velocity, tilt, fuel, reward, and episode status
- Training-curve and evaluation summaries
- Controls for replay speed and selecting recorded episodes

The browser is initially a visualization and replay client. Training remains in Python so experiments can be run reproducibly and without depending on browser performance.

## Planned architecture

```text
rocket-landing-rl/
├── README.md
├── pyproject.toml
├── src/
│   └── rocket_landing/
│       ├── environment.py    # Physics state, actions, rewards, termination
│       ├── policy.py          # PyTorch policy/value networks
│       ├── train.py           # Training entry point
│       ├── evaluate.py        # Held-out evaluation
│       └── recording.py       # Episode and metric serialization
├── tests/
│   ├── test_environment.py
│   ├── test_rewards.py
│   └── test_recording.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── experiments/
│   ├── configs/
│   └── results/
└── docs/
    ├── environment.md
    ├── reward-design.md
    └── experiments.md
```

## Milestones

- [x] Create a reproducible Python project and dependency configuration.
- [x] Implement the 2D physics state and deterministic stepping tests.
- [x] Add action validation, fuel use, landing checks, crash checks, and timeout handling.
- [x] Implement and test the reward function.
- [x] Add a baseline controller for validating the environment.
- [x] Train the first PyTorch policy.
- [ ] Add evaluation on held-out starting states.
- [ ] Record episode trajectories and metrics.
- [ ] Build the HTML/CSS/JavaScript replay viewer.
- [ ] Compare reward-function versions and document the results.

## Resume-quality standard

This project will be considered complete only when it includes:

- A reproducible setup and training command
- Automated tests for physics, rewards, and terminal states
- A documented observation/action space
- At least one baseline comparison
- Training and evaluation plots
- Honest success-rate and crash-rate measurements
- A browser demo that replays a trained policy
- A discussion of limitations and failure cases

The project will describe the method as reinforcement learning with reward shaping. It will not claim to reproduce Tesla's proprietary robotics-training system.

## Status

The Python project, the deterministic 2D environment (physics, fuel, landing/crash/timeout
handling, action validation), reward v1, and a hand-written heuristic baseline controller are
implemented and tested. The baseline lands safely in about 99% of 100 evaluation episodes
(see `docs/environment.md`).

A from-scratch PyTorch PPO agent (`policy.py`, `ppo.py`, `train.py`) trains on reward v1.
For the reference run (`experiments/configs/ppo_v1_tuned.json`, seed 1, 5M env steps,
about 9 minutes on one CPU thread) the final policy lands in 95/100 episodes on seeds not used for
training (best validation checkpoint: 100/100), against 99/100 for the heuristic baseline.
It touches down more gently but is slower and uses more fuel. Training is not reliable
across seeds: of six seeds with the same config, two kept a good final policy and one never
learned. See `docs/experiments.md`. Held-out evaluation tooling, trajectory recording and
the browser replay viewer are not built yet.

## Quickstart

```bash
uv sync --extra torch        # install dependencies (PyTorch needed for training and its tests)
uv run pytest                # run the test suite
uv run python scripts/evaluate_baseline.py --episodes 100 --seed 0   # baseline eval
uv run python -m rocket_landing.train --config experiments/configs/ppo_v1_tuned.json --seed 1
uv run python scripts/evaluate_policy.py --checkpoint experiments/results/ppo_v1_tuned_s1/final.pt \
    --episodes 100 --seed 0 --compare-baseline
```
