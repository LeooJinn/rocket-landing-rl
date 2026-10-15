import math

import numpy as np

from rocket_landing import RocketLandingEnv
from rocket_landing.baseline import (
    HeuristicController,
    evaluate,
    run_episode,
)


def test_controller_returns_valid_actions():
    env = RocketLandingEnv()
    ctrl = HeuristicController()
    obs, _ = env.reset(seed=3)
    for _ in range(50):
        a = ctrl.act(obs)
        assert a in (0, 1, 2, 3)
        assert ctrl.act(obs) == a  # pure function of obs
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    assert ctrl.act(np.array([0, 50, 0, 0, 0, 0, 1], dtype=np.float32)) in (0, 1, 2, 3)


def test_run_episode_deterministic():
    env = RocketLandingEnv()
    ctrl = HeuristicController()
    assert run_episode(env, ctrl, 5) == run_episode(env, ctrl, 5)


def test_evaluate_rates_sum_to_one():
    s = evaluate(HeuristicController(), 20, seed=0)
    assert s.n == len(s.episodes) == 20
    total = s.success_rate + s.crash_rate + s.out_of_bounds_rate + s.timeout_rate
    assert math.isclose(total, 1.0)


def test_baseline_success_floor():
    # Observed 99-100% on this range; floor leaves margin for tuning drift.
    s = evaluate(HeuristicController(), 20, seed=0)
    assert s.success_rate >= 0.8


def test_all_noop_crashes():
    class Noop:
        def act(self, obs):
            return 0

    s = evaluate(Noop(), 10, seed=0)
    assert s.success_rate == 0.0
    assert s.crash_rate == 1.0
