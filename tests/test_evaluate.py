import json

import pytest
import torch

from rocket_landing import evaluate as ev
from rocket_landing.policy import ActorCritic, RunningMeanStd, save_checkpoint
from rocket_landing.train import DEFAULT_CONFIG, TRAIN_SEED_BASE, training_env_seed


def make_run(base, name, train_seed=0, torch_seed=0, reward_version="v1"):
    d = base / name
    d.mkdir(parents=True)
    cfg = dict(DEFAULT_CONFIG, reward_version=reward_version)
    (d / "config.json").write_text(json.dumps({"config": cfg, "seed": train_seed}))
    torch.manual_seed(torch_seed)
    model = ActorCritic(hidden_sizes=(8,))
    save_checkpoint(d / "final.pt", model, RunningMeanStd(), cfg, extra={"update": 3})
    return d


def test_wilson_known_values():
    lo, hi = ev.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = ev.wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = ev.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-3)
    lo, hi = ev.wilson_interval(81, 100)
    assert (lo, hi) == pytest.approx((0.7225, 0.8755), abs=2e-3)
    assert all(map(lambda v: v != v, ev.wilson_interval(0, 0)))
    with pytest.raises(ValueError):
        ev.wilson_interval(3, 2)


def test_train_seed_constants_match_train():
    assert ev.TRAIN_SEED_BASE == TRAIN_SEED_BASE
    assert ev.train_seed_set(3, 16) == {training_env_seed(3, i) for i in range(16)}


def test_seed_disjointness_assertion():
    ok = ev.assert_heldout(50_000, 500, 0, 16, 20_000, 50)
    assert ok["heldout"] == [50_000, 50_499]
    ev.assert_heldout(20_050, 10, 0, 16, 20_000, 50)  # adjacent, fine
    with pytest.raises(AssertionError):  # overlaps in-training eval
        ev.assert_heldout(20_040, 100, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # reaches training seed region
        ev.assert_heldout(999_900, 200, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # exactly the training seeds
        ev.assert_heldout(TRAIN_SEED_BASE, 5, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):
        ev.assert_heldout(-5, 10, 0, 16, 20_000, 50)


def test_evaluate_run_rejects_overlap(tmp_path):
    d = make_run(tmp_path, "r")
    with pytest.raises(AssertionError):
        ev.evaluate_run(d, "final", episodes=5, seed_start=20_010, write=False)


def test_run_includes_baselines_and_is_deterministic(tmp_path):
    d = make_run(tmp_path, "r")
    kw = dict(checkpoint="final", episodes=4, seed_start=50_000, stochastic=True)
    a = ev.evaluate_run(d, write=True, **kw)
    b = ev.evaluate_run(d, write=False, **kw)
    assert a == b
    assert set(a["policies"]) == {"greedy", "stochastic", "heuristic", "random"}
    assert all(p["n"] == 4 for p in a["policies"].values())
    on_disk = json.loads((d / "heldout_eval.json").read_text())
    assert on_disk["policies"]["greedy"] == a["policies"]["greedy"]
    s = a["policies"]["greedy"]["success"]
    assert s["count"] / 4 == s["rate"] and 0 <= s["ci95"][0] <= s["rate"] <= s["ci95"][1] <= 1
    total = sum(a["policies"]["random"][k]["count"] for k in ("success", "crash", "out_of_bounds", "timeout"))
    assert total == 4
    assert "random" in ev.format_run(a)


def test_checkpoint_resolution(tmp_path):
    d = make_run(tmp_path, "r")
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "ckpt_00050.pt").write_bytes(b"")
    assert ev.resolve_checkpoint(d, "50").name == "ckpt_00050.pt"
    assert ev.resolve_checkpoint(d, "ckpt_00050").name == "ckpt_00050.pt"
    with pytest.raises(FileNotFoundError):
        ev.resolve_checkpoint(d, "best")
    with pytest.raises(ValueError):
        ev.resolve_checkpoint(d, "nonsense")


def _fake(run, rate_succ, n=10):
    def pol(k):
        c = round(rate_succ * n)
        return {
            "success": {"count": c, "rate": c / n, "ci95": [0, 1]},
            "crash": {"count": n - c, "rate": 1 - c / n, "ci95": [0, 1]},
            "out_of_bounds": {"count": 0, "rate": 0.0, "ci95": [0, 1]},
            "timeout": {"count": 0, "rate": 0.0, "ci95": [0, 1]},
            "reward": {"mean": 10.0 * k, "std": 1, "n": n},
            "fuel_used": {"mean": 0.5, "std": 0, "n": n},
            "episode_length": {"mean": 100.0, "std": 0, "n": n},
        }
    return {"run": run, "checkpoint": "final", "policies": {"greedy": pol(1), "heuristic": pol(2), "random": pol(0)}}


def test_aggregation_keeps_failed_seeds():
    rs = [_fake("a", 1.0), _fake("b", 0.0), _fake("c", 0.5)]
    agg = ev.aggregate(rs)
    g = agg["policies"]["greedy"]
    assert g["success_rate"] == pytest.approx({"mean": 0.5, "min": 0.0, "max": 1.0})
    assert g["crash_rate"]["mean"] == pytest.approx(0.5)
    assert g["n_failed_seeds"] == 1
    assert [p["run"] for p in g["per_run"]] == ["a", "b", "c"]
    assert agg["policies"]["random"]["mean_reward"]["mean"] == 0.0
    assert "stochastic" not in agg["policies"]
    text = ev.format_aggregate(rs, agg)
    assert "b" in text and "1/3" in text


def test_cli_multi_run(tmp_path, capsys):
    make_run(tmp_path, "a", train_seed=0, torch_seed=0)
    make_run(tmp_path, "b", train_seed=1, torch_seed=1, reward_version="v2")
    out = tmp_path / "agg.json"
    rc = ev.main(["--runs", str(tmp_path / "a"), str(tmp_path / "b"), "--episodes", "3",
                  "--aggregate-out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["aggregate"]["n_runs"] == 2 and len(data["runs"]) == 2
    assert (tmp_path / "b" / "heldout_eval.json").exists()
    assert "Aggregate over 2 runs" in capsys.readouterr().out
