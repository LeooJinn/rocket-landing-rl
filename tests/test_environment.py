import math

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import rocket_landing  # noqa: F401  (registers the env)
from rocket_landing.environment import (
    OBSERVATION_FIELDS,
    EnvConfig,
    RocketLandingEnv,
    action_to_controls,
    simulate_step,
    state_to_observation,
)
from rocket_landing.state import Action, Outcome, RocketState

CFG = EnvConfig()
G, DT = CFG.gravity, CFG.dt


def S(**kw) -> RocketState:
    base = dict(x=0.0, y=100.0, vx=0.0, vy=0.0, angle=0.0, angular_velocity=0.0, fuel=1.0)
    base.update(kw)
    return RocketState(**base)


def start(state: RocketState, config: EnvConfig | None = None) -> RocketLandingEnv:
    env = RocketLandingEnv(config)
    env.reset(seed=0, options={"state": state})
    return env


def test_reset_obs_and_info():
    env = RocketLandingEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == (7,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.array_equal(obs, state_to_observation(env.state))
    assert OBSERVATION_FIELDS == tuple(RocketState.__dataclass_fields__)
    for k in ("outcome", "steps", "fuel_exhausted", "state"):
        assert k in info
    assert info["outcome"] == "in_progress" and info["steps"] == 0
    assert info["fuel_exhausted"] is False


def test_state_before_reset():
    with pytest.raises(RuntimeError):
        _ = RocketLandingEnv().state


def test_seed_determinism():
    def rollout(seed):
        env = RocketLandingEnv()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for a in [1, 2, 0, 3, 1, 1, 0, 2] * 3:
            obs, r, term, trunc, _ = env.step(a)
            out.append(obs)
            if term or trunc:
                break
        return np.stack(out)

    assert np.array_equal(rollout(5), rollout(5))
    assert not np.array_equal(rollout(5)[0], rollout(6)[0])


def test_free_fall_formula():
    s = S(x=3.0, vx=2.0, vy=-4.0, angle=0.1, angular_velocity=0.2)
    env = start(s)
    env.step(Action.NOOP)
    n = env.state
    vy = -4.0 - G * DT
    assert n.vy == pytest.approx(vy)
    assert n.y == pytest.approx(100.0 + vy * DT)
    assert n.vx == 2.0 and n.x == pytest.approx(3.0 + 2.0 * DT)
    assert n.angle == pytest.approx(0.1 + 0.2 * DT)


def test_main_upright_accelerates_up():
    env = start(S())
    env.step(Action.MAIN)
    assert env.state.vy == pytest.approx((CFG.main_engine_accel - G) * DT)
    assert env.state.vx == pytest.approx(0.0)


def test_main_tilted_horizontal_sign():
    env = start(S(angle=0.3))  # CCW tilt: nose toward -x
    env.step(Action.MAIN)
    assert env.state.vx == pytest.approx(-math.sin(0.3) * CFG.main_engine_accel * DT)
    assert env.state.vx < 0
    env = start(S(angle=-0.3))
    env.step(Action.MAIN)
    assert env.state.vx > 0


def test_side_thrusters():
    env = start(S())
    env.step(Action.LEFT)
    assert env.state.angular_velocity == pytest.approx(CFG.side_engine_angular_accel * DT)
    env = start(S())
    env.step(Action.RIGHT)
    assert env.state.angular_velocity == pytest.approx(-CFG.side_engine_angular_accel * DT)


def test_action_to_controls():
    assert action_to_controls(Action.NOOP) == (0.0, 0.0)
    assert action_to_controls(Action.MAIN) == (1.0, 0.0)
    assert action_to_controls(Action.LEFT) == (0.0, 1.0)
    assert action_to_controls(Action.RIGHT) == (0.0, -1.0)


def test_fuel_use():
    env = start(S())
    _, _, _, _, info = env.step(Action.NOOP)
    assert info["fuel_used"] == 0.0 and env.state.fuel == 1.0
    _, _, _, _, info = env.step(Action.MAIN)
    assert info["fuel_used"] == pytest.approx(CFG.main_fuel_rate * DT)
    assert env.state.fuel == pytest.approx(1.0 - CFG.main_fuel_rate * DT)
    f = env.state.fuel
    _, _, _, _, info = env.step(Action.LEFT)
    assert info["fuel_used"] == pytest.approx(CFG.side_fuel_rate * DT)
    assert env.state.fuel == pytest.approx(f - CFG.side_fuel_rate * DT)


def test_fuel_never_negative_and_partial_thrust():
    tiny = CFG.main_fuel_rate * DT / 2
    s = simulate_step(S(fuel=tiny), 1.0, 0.0, CFG)
    assert s.fuel == 0.0
    assert s.vy == pytest.approx(-G * DT + 0.5 * CFG.main_engine_accel * DT)
    env = start(S(fuel=0.01))
    for _ in range(100):
        _, _, term, trunc, _ = env.step(Action.MAIN)
        assert env.state.fuel >= 0.0
        if term or trunc:
            break


def test_no_fuel_no_thrust():
    env = start(S(fuel=0.0, angle=0.1))
    _, _, _, _, info = env.step(Action.MAIN)
    assert info["fuel_exhausted"] is True and info["fuel_used"] == 0.0
    assert env.state.vy == pytest.approx(-G * DT)
    assert env.state.vx == 0.0 and env.state.fuel == 0.0
    env.step(Action.LEFT)
    assert env.state.angular_velocity == 0.0


def test_soft_landing():
    env = start(S(y=0.05, vy=-1.0, x=5.0))
    _, _, term, trunc, info = env.step(Action.NOOP)
    assert term and not trunc
    assert env.outcome is Outcome.LANDED and info["outcome"] == "landed"
    assert env.state.y == 0.0
    assert set(info["landing"]) == {"vx", "vy", "angle", "angular_velocity", "x"}


@pytest.mark.parametrize(
    "kw",
    [
        dict(vy=-5.0),
        dict(vx=1.5),
        dict(angle=0.3),
        dict(angular_velocity=0.8),
        dict(x=15.0),
    ],
)
def test_crash_each_threshold(kw):
    base = dict(y=0.05, vy=-1.0, x=0.0)
    base.update(kw)
    env = start(S(**base))
    _, _, term, _, info = env.step(Action.NOOP)
    assert term and env.outcome is Outcome.CRASHED
    assert "landing" in info


def test_out_of_bounds():
    env = start(S(x=99.9, vx=10.0))
    _, _, term, _, _ = env.step(Action.NOOP)
    assert term and env.outcome is Outcome.OUT_OF_BOUNDS
    env = start(S(y=199.9, vy=10.0))
    _, _, term, _, _ = env.step(Action.NOOP)
    assert term and env.outcome is Outcome.OUT_OF_BOUNDS


def test_timeout():
    env = start(S(y=150.0), EnvConfig(max_steps=3))
    for i in range(3):
        _, _, term, trunc, info = env.step(Action.NOOP)
        assert not term
        assert trunc == (i == 2)
    assert env.outcome is Outcome.TIMEOUT and info["steps"] == 3
    assert env.steps == 3


@pytest.mark.parametrize(
    "bad", [1.0, 1.5, True, False, "1", None, -1, 4, np.array([1, 2]), np.array(1.0), [1]]
)
def test_invalid_actions(bad):
    env = start(S())
    with pytest.raises(ValueError):
        env.step(bad)


@pytest.mark.parametrize("ok", [1, np.int64(1), np.int32(2), Action.MAIN, np.array(3)])
def test_valid_actions(ok):
    env = start(S())
    env.step(ok)


def test_step_lifecycle_errors():
    env = RocketLandingEnv()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0, options={"state": S(y=0.05, vy=-1.0)})
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    env.step(0)


def test_reset_rejects_grounded_state():
    with pytest.raises(ValueError):
        RocketLandingEnv().reset(options={"state": S(y=0.0)})


def test_unknown_reward_version():
    with pytest.raises(ValueError):
        RocketLandingEnv(reward_version="nope")


def test_reward_info():
    env = start(S())
    _, r, _, _, info = env.step(0)
    assert isinstance(r, float) and "reward_terms" in info


def test_gym_make_and_checker():
    env = gymnasium.make("RocketLanding-v0")
    env.reset(seed=0)
    check_env(env.unwrapped)


def test_angle_wrapping():
    s = simulate_step(S(angle=3.1, angular_velocity=2.0), 0.0, 0.0, CFG)
    assert -math.pi <= s.angle <= math.pi
    assert s.angle == pytest.approx(3.1 + 0.1 - 2 * math.pi)
    s = simulate_step(S(angle=-3.1, angular_velocity=-2.0), 0.0, 0.0, CFG)
    assert -math.pi <= s.angle <= math.pi
