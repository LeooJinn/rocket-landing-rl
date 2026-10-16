"""Reward v3 (v3a / v3b): efficiency terms and the absence of new exploits.

v3a = v2 + time penalty 0.03/step + landing bonus 40 * fuel left.
v3b = v3a + extra -25 when the CRASHED touchdown happens with an empty tank.
"""

import dataclasses

import pytest

from rocket_landing.baseline import HeuristicController, run_episode
from rocket_landing.environment import RocketLandingEnv
from rocket_landing.rewards import (
    REWARD_VERSIONS,
    RewardV1Config,
    RewardV2Config,
    RewardV3BConfig,
    RewardV3Config,
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
BURN = 0.0025
V3 = ("v3a", "v3b")


def S(x=0.0, y=50.0, vx=0.0, vy=0.0, angle=0.0, w=0.0, fuel=1.0):
    return RocketState(x, y, vx, vy, angle, w, fuel)


def r(version, prev, st, outcome=IP):
    return compute_reward(prev, st, Action.NOOP, outcome, version=version)


def episode_return(version, states, last_outcome):
    total = 0.0
    for i, (a, b) in enumerate(zip(states, states[1:])):
        o = last_outcome if i == len(states) - 2 else IP
        total += r(version, a, b, o).total
    return total


def traj(n, fuel_used, end, start=None):
    """Linear interpolation of n steps from ``start`` to ``end``; fuel falls linearly to 1 - fuel_used."""
    start = start or S(x=30, y=100, vx=-2, vy=-5)
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append(
            S(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
                start.vx + (end.vx - start.vx) * t,
                start.vy + (end.vy - start.vy) * t,
                end.angle * t,
                0.0,
                1.0 - fuel_used * t,
            )
        )
    pts[-1] = dataclasses.replace(pts[-1], fuel=max(0.0, 1.0 - fuel_used))
    return pts


def landing(n, fuel_used):
    return traj(n, fuel_used, S(x=0.0, y=0.0, vx=0.2, vy=-1.0, angle=0.02))


def hard_crash(n, fuel_used, vy=-44.0):
    return traj(n, fuel_used, S(x=5.0, y=0.0, vx=0.0, vy=vy))


# --------------------------------------------------------------- registry
def test_versions_registered_and_old_ones_untouched():
    assert REWARD_VERSIONS == ("v1", "v2", "v3a", "v3b")
    assert RewardV1Config().time_penalty == 0.01
    assert RewardV2Config().time_penalty == 0.01
    assert RewardV2Config().crash_softness_bonus == 50.0


def test_config_defaults_and_types():
    a, b = RewardV3Config(), RewardV3BConfig()
    assert (a.time_penalty, a.landing_fuel_bonus, a.fuel_out_crash_penalty) == (0.03, 40.0, 0.0)
    assert (b.time_penalty, b.landing_fuel_bonus, b.fuel_out_crash_penalty) == (0.03, 40.0, 25.0)
    with pytest.raises(TypeError):
        compute_reward(S(), S(), Action.NOOP, IP, version="v3a", config=RewardV2Config())
    with pytest.raises(TypeError):
        compute_reward(S(), S(), Action.NOOP, IP, version="v3b", config=RewardV3Config())
    with pytest.raises(TypeError):
        compute_reward(S(), S(), Action.NOOP, IP, version="v2", config=RewardV3Config())
    compute_reward(S(), S(), Action.NOOP, IP, version="v3a", config=RewardV3Config())
    compute_reward(S(), S(), Action.NOOP, IP, version="v3b", config=RewardV3BConfig())


@pytest.mark.parametrize("version", V3)
def test_terms_are_v2_plus_new_keys(version):
    prev, st = S(y=1, fuel=0.4), S(y=0, vy=-3, fuel=0.39)
    for o in Outcome:
        b = r(version, prev, st, o)
        assert set(b.terms) == set(r("v2", prev, st, o).terms) | {"landing_fuel", "fuel_out"}
        assert b.total == pytest.approx(sum(b.terms.values()))


def test_reduces_to_v2_with_neutral_weights():
    """With v2's time penalty and both new weights at 0, v3 is exactly v2 (+ two zero terms)."""
    prev = S(x=5, y=2, vy=-3, fuel=0.5)
    st = S(x=4, y=0, vx=0.2, vy=-1.0, angle=0.02, fuel=0.0)
    neutral = dict(time_penalty=0.01, landing_fuel_bonus=0.0, fuel_out_crash_penalty=0.0)
    for version, cfg_type in (("v3a", RewardV3Config), ("v3b", RewardV3BConfig)):
        for o in Outcome:
            b = compute_reward(prev, st, Action.NOOP, o, version=version, config=cfg_type(**neutral))
            v2 = r("v2", prev, st, o)
            assert {k: v for k, v in b.terms.items() if k not in ("landing_fuel", "fuel_out")} == v2.terms
            assert b.total == pytest.approx(v2.total)


def test_v3b_equals_v3a_except_fuel_out():
    prev, st = S(y=1, fuel=0.2), S(y=0, vy=-6, fuel=0.0)
    for o in Outcome:
        a, b = r("v3a", prev, st, o), r("v3b", prev, st, o)
        assert {k: v for k, v in a.terms.items() if k != "fuel_out"} == {
            k: v for k, v in b.terms.items() if k != "fuel_out"
        }
    assert r("v3a", prev, st, C).terms["fuel_out"] == 0.0
    assert r("v3b", prev, st, C).terms["fuel_out"] == -25.0


# ------------------------------------------------------------ the new terms
@pytest.mark.parametrize("version", V3)
def test_time_penalty_is_three_times_v2(version):
    a, b = S(), S()
    assert r(version, a, b).terms["time"] == pytest.approx(-0.03)
    assert r("v2", a, b).terms["time"] == pytest.approx(-0.01)


@pytest.mark.parametrize("version", V3)
def test_landing_fuel_bonus_linear_in_fuel_left_and_landed_only(version):
    prev = S(y=1, fuel=0.9)
    for fuel in (0.0, 0.1, 0.35, 0.65, 1.0):
        st = S(y=0, vy=-1.0, fuel=fuel)
        assert r(version, prev, st, L).terms["landing_fuel"] == pytest.approx(40.0 * fuel)
        for o in (IP, C, OOB, TO):
            assert r(version, prev, st, o).terms["landing_fuel"] == 0.0


def test_fuel_out_penalty_only_for_empty_tank_crash_in_v3b():
    prev = S(y=1, fuel=0.001)
    empty, some = S(y=0, vy=-5, fuel=0.0), S(y=0, vy=-5, fuel=0.001)
    assert r("v3b", prev, empty, C).terms["fuel_out"] == -25.0
    assert r("v3b", prev, some, C).terms["fuel_out"] == 0.0
    for o in (IP, L, OOB, TO):
        assert r("v3b", prev, empty, o).terms["fuel_out"] == 0.0
    # A landing that uses the last drop of fuel is not a fuel-out crash.
    assert r("v3b", prev, S(y=0, vy=-1.0, fuel=0.0), L).terms["fuel_out"] == 0.0
    # An empty-tank crash is strictly worse than the same impact with fuel left.
    assert r("v3b", prev, empty, C).total < r("v3b", prev, some, C).total


# ------------------------------------------------- efficiency ordering (goal)
@pytest.mark.parametrize("version", V3)
def test_fast_frugal_landing_beats_slow_wasteful_landing_by_tens_of_points(version):
    fast = episode_return(version, landing(250, 0.35), L)  # ~250 steps, 65% fuel left
    slow = episode_return(version, landing(700, 1.00), L)  # ~700 steps, empty tank at touchdown
    gap_v3 = fast - slow
    gap_v2 = episode_return("v2", landing(250, 0.35), L) - episode_return("v2", landing(700, 1.00), L)
    assert gap_v3 >= 40.0
    assert gap_v3 >= 3 * gap_v2 > 0  # v2 already preferred the fast one, but only by ~12 (v3: ~47)
    assert gap_v2 < 15.0


@pytest.mark.parametrize("version", V3)
def test_return_monotone_in_steps_and_fuel_used(version):
    by_steps = [episode_return(version, landing(n, 0.5), L) for n in (200, 300, 450, 700)]
    by_fuel = [episode_return(version, landing(300, f), L) for f in (0.2, 0.4, 0.6, 0.9)]
    assert all(a > b for a, b in zip(by_steps, by_steps[1:]))
    assert all(a > b for a, b in zip(by_fuel, by_fuel[1:]))


# ------------------------------------------------------- no new exploits
@pytest.mark.parametrize("version", V3)
def test_every_safe_landing_still_beats_every_crash_by_a_wide_margin(version):
    # Worst conceivable safe landing: pad edge, all limits, slowest and fuel-less (700 steps, empty).
    worst = traj(
        1000, 1.0, S(x=10.0, y=0.0, vx=1.0, vy=-2.0, angle=0.2), start=S(x=40, y=120, vx=5, vy=-10)
    )
    worst[-1] = dataclasses.replace(worst[-1], angular_velocity=0.5)
    worst_land = episode_return(version, worst, L)
    # Best crash: soft touchdown just beyond the limits, fast and with a full tank (no time cost).
    best = traj(100, 0.0, S(x=0.0, y=0.0, vx=0.0, vy=-2.01))
    best_crash = episode_return(version, best, C)
    assert worst_land > 40.0
    assert best_crash < -30.0
    assert worst_land - best_crash >= 100.0
    # Same hard guarantee at the terminal level (excl. dense terms): >= 150 as in v2.
    land_t = r(version, S(y=1, fuel=0.0), S(y=0, vy=-1.0, fuel=0.0), L).terms
    crash_t = r(version, S(y=1), S(y=0, vy=-2.01), C).terms
    gap = land_t["terminal"] + land_t["precision"] - (crash_t["terminal"] + crash_t["crash_softness"])
    assert gap >= 149.0


@pytest.mark.parametrize("version", V3)
def test_early_crash_does_not_beat_careful_landing(version):
    careful = episode_return(version, landing(700, 0.95), L)  # the most expensive way to land
    for steps in (20, 85, 150):
        early = episode_return(version, hard_crash(steps, 0.0), C)
        assert early < careful - 150.0


@pytest.mark.parametrize("version", V3)
def test_diving_to_save_fuel_does_not_pay(version):
    """Free-fall touchdown (no burn, ~85 steps) vs the efficient landing."""
    dive = episode_return(version, hard_crash(85, 0.0), C)
    good = episode_return(version, landing(250, 0.35), L)
    assert dive < 0 < good
    assert good - dive >= 200.0
    # Even a perfectly gentle but just-too-fast touchdown with the whole tank left loses.
    almost = episode_return(version, traj(250, 0.35, S(x=0.0, y=0.0, vy=-2.5)), C)
    assert almost < good - 150.0


@pytest.mark.parametrize("version", V3)
def test_hover_to_timeout_and_hover_until_empty_worse_than_landing(version):
    start = S(x=30, y=100, vx=-2, vy=-5)
    good = episode_return(version, landing(250, 0.35), L)
    free = [start] * 1001  # hover without burning anything (impossible, but the cheapest hover)
    assert episode_return(version, free, TO) < good - 150.0
    burn = [dataclasses.replace(start, fuel=max(0.0, 1.0 - BURN * i)) for i in range(1001)]
    assert episode_return(version, burn, TO) < good - 150.0
    # Hover until the tank is empty, then a hard crash from altitude ~800 steps later.
    hover_crash = burn[:800] + [S(x=30, y=0, vx=-2, vy=-44, fuel=0.0)]
    assert episode_return(version, hover_crash, C) < episode_return(version, hard_crash(85, 0.0), C)


def test_fuel_out_crash_ranked_below_soft_crash_below_landing():
    """The ordering v3b adds: landing > soft crash with fuel > fuel-out crash at the same impact."""
    landed = episode_return("v3b", landing(300, 0.5), L)
    soft_fuel = episode_return("v3b", traj(300, 0.5, S(x=0.0, y=0.0, vy=-2.5)), C)
    soft_empty = episode_return("v3b", traj(300, 1.0, S(x=0.0, y=0.0, vy=-2.5)), C)
    assert landed > soft_fuel > soft_empty
    # v3a does not add the explicit terminal penalty: the gap there is only the dense fuel cost.
    gap_a = episode_return("v3a", traj(300, 0.5, S(vy=-2.5, y=0.0)), C) - episode_return(
        "v3a", traj(300, 1.0, S(vy=-2.5, y=0.0)), C
    )
    gap_b = soft_fuel - soft_empty
    assert gap_b == pytest.approx(gap_a + 25.0)


@pytest.mark.parametrize("version", V3)
def test_return_is_bounded(version):
    """Largest conceivable landing return stays near 100 + 20 + 40 (no unbounded term)."""
    best = traj(60, 0.0, S(x=0.0, y=0.0, vx=0.0, vy=0.0), start=S(x=0, y=1.0, vy=-0.5))
    assert episode_return(version, best, L) < 100.0 + 20.0 + 40.0 + 2.0


# ----------------------------------------------------------- env + baseline
@pytest.mark.parametrize("version", V3)
def test_env_runs_episode_with_new_terms(version):
    env = RocketLandingEnv(reward_version=version)
    env.reset(seed=0)
    for _ in range(2000):
        out = env.step(env.action_space.sample())
        assert {"landing_fuel", "fuel_out", "crash_softness"} <= set(out[-1]["reward_terms"])
        if out[2] or out[3]:
            break
    else:
        pytest.fail("episode did not terminate")


def _heuristic_episodes(version, gains, n=40, seed=50000):
    env = RocketLandingEnv(reward_version=version)
    return [run_episode(env, HeuristicController(**gains), seed + i) for i in range(n)]


def test_reward_does_not_change_dynamics_and_heuristic_lands_under_v3():
    base = _heuristic_episodes("v2", {})
    key = lambda e: (e.seed, e.outcome, e.steps, round(e.fuel_used, 9))  # noqa: E731
    for version in V3:
        eps = _heuristic_episodes(version, {})
        assert [key(e) for e in eps] == [key(e) for e in base]
        assert sum(e.outcome == "landed" for e in eps) / len(eps) >= 0.95
        # return = v2 return + 0.02/step extra time cost, - nothing else, + 40 * fuel left
        for e2, e3 in zip(base, eps):
            expected = e2.total_reward - 0.02 * e2.steps + 40.0 * (1.0 - e2.fuel_used)
            assert e3.total_reward == pytest.approx(expected, abs=1e-6)


def test_heuristic_ladder_ranks_fast_frugal_above_slow_wasteful():
    """Heuristic controllers with slower descent profiles: v3 returns must rank by efficiency."""
    gains = [{}, {"descent_decel": 0.5}, {"descent_decel": 0.2}]
    means = {}
    for version in ("v2", "v3a", "v3b"):
        means[version] = []
        for g in gains:
            eps = _heuristic_episodes(version, g, n=25)
            assert all(e.outcome == "landed" for e in eps)
            means[version].append(sum(e.total_reward for e in eps) / len(eps))
    for version in V3:
        m = means[version]
        assert m[0] > m[1] > m[2]
        assert m[0] - m[2] >= 15.0  # clear separation (v2: ~7 points)
    assert means["v2"][0] - means["v2"][2] < 12.0
