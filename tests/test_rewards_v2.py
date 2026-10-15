import pytest

from rocket_landing.environment import RocketLandingEnv
from rocket_landing.rewards import (
    REWARD_VERSIONS,
    RewardV1Config,
    RewardV2Config,
    compute_reward,
)
from rocket_landing.state import Action, Outcome, RocketState

IP, L, C, OOB, TO = (
    Outcome.IN_PROGRESS,
    Outcome.LANDED,
    Outcome.CRASHED,
    Outcome.OUT_OF_BOUNDS,
    Outcome.TIMEOUT,
)


def S(x=0.0, y=50.0, vx=0.0, vy=0.0, angle=0.0, w=0.0, fuel=1.0):
    return RocketState(x, y, vx, vy, angle, w, fuel)


def r2(prev, st, outcome=IP):
    return compute_reward(prev, st, Action.NOOP, outcome, version="v2")


def r1(prev, st, outcome=IP):
    return compute_reward(prev, st, Action.NOOP, outcome, version="v1")


def crash_bonus(st):
    return r2(S(y=1), st, C).terms["crash_softness"]


def test_versions_and_unknown():
    assert REWARD_VERSIONS == ("v1", "v2")
    with pytest.raises(ValueError):
        compute_reward(S(), S(), Action.NOOP, IP, version="v3")


def test_config_type_checked():
    with pytest.raises(TypeError):
        compute_reward(S(), S(), Action.NOOP, IP, version="v2", config=RewardV1Config())
    with pytest.raises(TypeError):
        compute_reward(S(), S(), Action.NOOP, IP, version="v1", config=RewardV2Config())
    compute_reward(S(), S(), Action.NOOP, IP, version="v2", config=RewardV2Config())


def test_v2_equals_v1_on_non_crash():
    prev = S(x=5, y=2, vy=-3, fuel=0.5)
    st = S(x=4, y=0, vx=0.2, vy=-1.0, angle=0.02, fuel=0.49)
    for o in (IP, L, OOB, TO):
        a, b = r1(prev, st, o), r2(prev, st, o)
        assert b.terms["crash_softness"] == 0.0
        assert {k: v for k, v in b.terms.items() if k != "crash_softness"} == a.terms
        assert b.total == a.total


def test_terms_are_v1_plus_crash_softness():
    prev, st = S(y=1), S(y=0, vy=-3)
    assert set(r2(prev, st, C).terms) == set(r1(prev, st, C).terms) | {"crash_softness"}
    b = r2(prev, st, C)
    assert b.total == pytest.approx(sum(b.terms.values()))
    assert b.terms["terminal"] == -100.0


def test_softness_boundaries():
    assert crash_bonus(S(y=0, vy=-2.0001)) == pytest.approx(50.0, abs=0.01)
    assert crash_bonus(S(y=0, vy=-2.0)) == pytest.approx(50.0)
    assert crash_bonus(S(y=0, vy=-10.0)) == 0.0  # excess == 5
    assert crash_bonus(S(y=0, vy=-20.0)) == 0.0
    assert crash_bonus(S(y=0, vy=-6.0)) == pytest.approx(25.0)  # excess 3


@pytest.mark.parametrize(
    "kw,limit",
    [("x", 10.0), ("vx", 1.0), ("vy", 2.0), ("angle", 0.2), ("w", 0.5)],
)
def test_each_component_counts(kw, limit):
    assert crash_bonus(S(y=0, **{kw: 3 * limit})) == pytest.approx(25.0)
    assert crash_bonus(S(y=0, **{kw: -3 * limit})) == pytest.approx(25.0)


def test_monotone_in_impact_speed():
    vals = [crash_bonus(S(y=0, vy=-v)) for v in (0.5, 2.5, 4, 6, 8, 10, 15, 44)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] == 50.0 and vals[-1] == 0.0
    assert len(set(vals)) > 3


def test_crash_below_every_safe_landing():
    worst_landing = r2(
        S(x=10, y=2, vx=1, vy=-2, angle=0.2, fuel=0.0),
        S(x=10, y=0, vx=1, vy=-2, angle=0.2, w=0.5, fuel=0.0),
        L,
    )
    landing_terminal = worst_landing.terms["terminal"] + worst_landing.terms["precision"]
    assert landing_terminal >= 100.0
    for vy in (-0.1, -2.5, -5, -44):
        b = r2(S(y=1, vy=vy), S(y=0, vy=vy), C)
        crash_terminal = b.terms["terminal"] + b.terms["crash_softness"]
        assert -100.0 <= crash_terminal <= -50.0
        assert landing_terminal - crash_terminal >= 150.0
        assert b.total < worst_landing.total


def test_soft_crash_not_better_than_timeout_preference():
    soft = r2(S(y=1), S(y=0), C).terms
    assert soft["terminal"] + soft["crash_softness"] == pytest.approx(-50.0)
    timeout = r2(S(y=1), S(y=0), TO).terms["terminal"]
    assert soft["terminal"] + soft["crash_softness"] >= timeout


def test_free_fall_crash_full_penalty():
    b = r2(S(y=1, vy=-44), S(y=0, vy=-44), C)
    assert b.terms["crash_softness"] == 0.0
    assert b.terms["terminal"] + b.terms["crash_softness"] == -100.0


def test_env_runs_v2_episode():
    env = RocketLandingEnv(reward_version="v2")
    env.reset(seed=0)
    for _ in range(2000):
        out = env.step(env.action_space.sample())
        info = out[-1]
        assert "crash_softness" in info["reward_terms"]
        if out[2] or out[3]:
            break
    else:
        pytest.fail("episode did not terminate")
