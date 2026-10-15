import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rocket_landing.policy import PolicyController
from rocket_landing.train import load_config, train, training_env_seed

SMOKE = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "smoke.json"


def _metrics(run_dir):
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("time_s")
    return rows


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("runs")
    cfg = load_config(SMOKE)
    out = {}
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        out[name] = (train(cfg, seed, base / name), base / name)
    return out


def test_writes_all_files(runs):
    summary, d = runs["a"]
    for f in ("config.json", "metrics.jsonl", "eval.jsonl", "final.pt", "best.pt", "summary.json"):
        assert (d / f).exists(), f
    assert (d / "checkpoints" / "ckpt_00001.pt").exists()
    rows = _metrics(d)
    assert len(rows) == summary["num_updates"] == 4
    assert rows[-1]["global_step"] == 512
    assert json.loads((d / "config.json").read_text())["seed"] == 0


def test_deterministic_same_seed(runs):
    (_, a), (_, b) = runs["a"], runs["b"]
    assert _metrics(a) == _metrics(b)
    pa = torch.load(a / "final.pt", weights_only=False)["model_state"]
    pb = torch.load(b / "final.pt", weights_only=False)["model_state"]
    assert pa.keys() == pb.keys()
    for k in pa:
        assert torch.equal(pa[k], pb[k])


def test_different_seed_differs(runs):
    (_, a), (_, c) = runs["a"], runs["c"]
    pa = torch.load(a / "final.pt", weights_only=False)["model_state"]
    pc = torch.load(c / "final.pt", weights_only=False)["model_state"]
    assert any(not torch.equal(pa[k], pc[k]) for k in pa)


def test_final_checkpoint_loads_as_controller(runs):
    _, d = runs["a"]
    ctrl = PolicyController.from_checkpoint(d / "final.pt")
    assert ctrl.act(np.zeros(7, dtype=np.float32)) in range(4)


def test_training_env_seeds_avoid_eval_range():
    for seed in (0, 1, 99):
        for i in range(64):
            assert training_env_seed(seed, i) >= 1_000_000


def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"bogus": 1}))
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text(json.dumps({"ppo": {"bogus": 1}}))
    with pytest.raises(ValueError):
        load_config(p)
    assert load_config(SMOKE)["num_envs"] == 2
