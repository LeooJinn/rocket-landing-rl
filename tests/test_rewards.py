import math
import random

import pytest

from rocket_landing.rewards import (
    REWARD_VERSIONS,
    RewardBreakdown,
    compute_reward,
    potential,
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


def S(x=0.0, y=50.0, vx=0.0, vy=0.0, angle=0.0, w=0.0, fuel=1.0):
    return RocketState(x, y, vx, vy, angle, w, fuel)


def r(prev, st, outcome=IP, action=Action.NOOP):
    return compute_reward(prev, st, action, outcome)


def test_versions_and_unknown():
    assert "v1" in REWARD_VERSIONS
    with pytest.raises(ValueError):
        compute_reward(S(), S(), Action.NOOP, IP, version="nope")


def test_total_is_sum_and_float():
    for o in Outcome:
        b = r(S(x=5, fuel=0.5), S(x=4, y=40, vy=-3, angle=0.1, fuel=0.49), o)
        assert isinstance(b, RewardBreakdown)
        assert type(b.total) is float and math.isfinite(b.total)
        assert abs(b.total - sum(b.terms.values())) < 1e-9
        for k in ("shaping", "fuel", "time", "terminal"):
            assert k in b.terms


def test_shaping_signs():
    assert r(S(x=20), S(x=10)).terms["shaping"] > 0
    assert r(S(x=10), S(x=20)).terms["shaping"] < 0
    assert r(S(vy=-8), S(vy=-2)).terms["shaping"] > 0
    assert r(S(vy=-2), S(vy=-8)).terms["shaping"] < 0
    assert r(S(angle=0.4), S(angle=0.1)).terms["shaping"] > 0
    assert r(S(angle=0.1), S(angle=-0.4)).terms["shaping"] < 0


def test_shaping_telescopes():
    rng = random.Random(1)
    states = [
        S(rng.uniform(-40, 40), rng.uniform(0, 120), rng.uniform(-5, 5),
          rng.uniform(-10, 0), rng.uniform(-1, 1), 0.0, 1.0)
        for _ in range(200)
    ]
    total = sum(r(a, b).terms["shaping"] for a, b in zip(states, states[1:]))
    assert total == pytest.approx(potential(states[-1]) - potential(states[0]))
    loop = states + [states[0]]
    assert sum(r(a, b).terms["shaping"] for a, b in zip(loop, loop[1:])) == pytest.approx(0, abs=1e-9)


def episode_return(states, outcomes_last, burn=0.0):
    total = 0.0
    for i, (a, b) in enumerate(zip(states, states[1:])):
        o = outcomes_last if i == len(states) - 2 else IP
        total += r(a, b, o).total
    return total


def landing_traj(x_end=0.0, n=400, fuel_used=0.3, start=None):
    """Smooth descent from a start state to a touchdown at x_end."""
    start = start or S(x=30, y=100, vx=-2, vy=-5)
    end = S(x=x_end, y=0.0, vx=0.2, vy=-1.0, angle=0.02, w=0.0, fuel=1 - fuel_used)
    out = []
    for i in range(n + 1):
        t = i / n
        out.append(S(
            start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t,
            start.vx + (end.vx - start.vx) * t, start.vy + (end.vy - start.vy) * t,
            end.angle * t, 0.0, 1 - fuel_used * t,
        ))
    return out


def test_hover_worse_than_landing():
    start = S(x=30, y=100, vx=-2, vy=-5)
    land = episode_return(landing_traj(start=start), L)
    # Hover 1000 steps at the same pose, burning main fuel until empty, then timeout.
    hover = [S(30, 100, -2, -5, 0, 0, max(0.0, 1 - BURN * i)) for i in range(1001)]
    hov = episode_return(hover, TO)
    assert hov < land
    assert hov < 0 < land
    # Even a hover that burns nothing is strictly worse (time + timeout).
    free = [start] * 1001
    assert episode_return(free, TO) < land


def test_timeout_worse_than_landing():
    st = S(x=3, y=1, vy=-0.5)
    assert r(st, st, TO).total < r(st, st, IP).total
    assert r(st, st, TO).total < r(st, S(x=3, y=0, vy=-0.5), L).total


def test_crash_near_pad_worse_than_any_safe_landing():
    prev = S(x=0.5, y=2.0, vy=-8)
    crash = r(prev, S(x=0.5, y=0, vy=-8), C).total
    # Worst safe landing: edge of pad, at all safety limits, from a bad approach.
    worst = r(S(x=10, y=2, vx=1, vy=-2, angle=0.2, fuel=0.0),
              S(x=10, y=0, vx=1, vy=-2, angle=0.2, w=0.5, fuel=0.0), L).total
    best_crash = r(S(x=0, y=0.1, vy=-2.1), S(x=0, y=0, vy=-2.1), C).total
    assert crash < worst and best_crash < worst
    assert worst > 0


def test_fuel_waste_penalised():
    a, b = S(fuel=0.8), S(fuel=0.8 - BURN)
    same = S(fuel=0.8)
    assert r(a, b, action=Action.MAIN).total < r(a, same).total
    assert r(a, b).terms["fuel"] == pytest.approx(-0.03)
    assert r(a, same).terms["fuel"] == 0.0
    # Same positions across an episode, more burning => lower return.
    base = landing_traj(fuel_used=0.1)
    wasteful = landing_traj(fuel_used=0.9)
    assert episode_return(wasteful, L) < episode_return(base, L)


def test_approach_then_leave_worse_than_landing():
    start = S(x=30, y=100, vx=-2, vy=-5)
    land = episode_return(landing_traj(start=start), L)
    approach = landing_traj(start=start, n=200)[:-1]
    near = approach[-1]
    away = [near] + [S(near.x + 2 * i, near.y + 2 * i, 2, 2, 0, 0, near.fuel) for i in range(1, 60)]
    out = episode_return(approach + away, OOB)
    assert out < land
    assert out < 0


def test_terminal_only_on_terminal():
    a, b = S(x=5), S(x=4)
    assert r(a, b, IP).terms["terminal"] == 0.0
    assert r(a, b, L).terms["terminal"] == 100.0
    assert r(a, b, C).terms["terminal"] == -100.0
    assert r(a, b, OOB).terms["terminal"] == -100.0
    assert r(a, b, TO).terms["terminal"] == -50.0
    for o in (IP, C, OOB, TO):
        assert r(a, b, o).terms["precision"] == 0.0


def test_precision_bonus():
    prev = S(y=1)
    perfect = r(prev, S(y=0), L).terms["precision"]
    off = r(prev, S(x=8, y=0), L).terms["precision"]
    fast = r(prev, S(y=0, vy=-1.9), L).terms["precision"]
    assert perfect == pytest.approx(20.0)
    assert 0 <= off < perfect and 0 <= fast < perfect
    assert r(prev, S(x=10, y=0), L).terms["precision"] == 0.0


def test_determinism():
    a, b = S(x=3, y=9, vy=-2, fuel=0.5), S(x=2, y=8, vy=-1.5, fuel=0.49)
    assert r(a, b, L) == r(a, b, L)
