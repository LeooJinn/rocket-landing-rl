import importlib.util
import json
from pathlib import Path

import pytest
import torch

from rocket_landing.policy import ActorCritic, RunningMeanStd, save_checkpoint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load("export_static_demo")
sv = ex.sv

FRAME_KEYS = {"t", "x", "y", "vx", "vy", "angle", "av", "fuel", "action", "reward", "cum_reward"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("results") / "tiny_run"
    (d / "checkpoints").mkdir(parents=True)
    torch.manual_seed(0)
    for u in (10, 40, 50, 100):
        save_checkpoint(d / "checkpoints" / f"ckpt_{u:05d}.pt", ActorCritic(), RunningMeanStd(),
                        {"reward_version": "v1"}, extra={"update": u})
    save_checkpoint(d / "final.pt", ActorCritic(), RunningMeanStd(), {"reward_version": "v1"},
                    extra={"update": 100})
    (d / "config.json").write_text(json.dumps({"seed": 0, "config": {"total_timesteps": 1000, "reward_version": "v1"}}))
    (d / "metrics.jsonl").write_text(
        "".join(json.dumps({"update": i, "global_step": i * 10, "entropy": 1.0 / i}) + "\n" for i in range(1, 401)))
    (d / "eval.jsonl").write_text(json.dumps({"update": 100, "success_rate": 0.0, "crash_rate": 1.0}) + "\n")
    (d / "summary.json").write_text("{}")
    return d


def test_export_files_and_contract(run_dir, tmp_path):
    out = tmp_path / "demo"
    m = ex.export(run_dir, out, n_seeds=3, first_seed=7, fractions=[0.1, 0.4])
    assert m["seeds"] == [7, 8, 9]
    assert [c["id"] for c in m["checkpoints"]] == ["ckpt_00010", "ckpt_00040", "final"]
    for name in ("manifest.json", "runs.json", "metrics_tiny_run.json", "checkpoints_tiny_run.json"):
        assert (out / name).is_file()
    for ck in ("ckpt_00010", "ckpt_00040", "final"):
        for s in (7, 8, 9):
            assert (out / f"episode_tiny_run__{ck}__{s}.json").is_file()
    for s in (7, 8, 9):
        assert (out / f"baseline_{s}.json").is_file()

    runs = json.loads((out / "runs.json").read_text())
    assert len(runs) == 1 and runs[0]["name"] == "tiny_run" and runs[0]["finished"] is True
    assert {"name", "reward_version", "seed", "updates", "total_timesteps", "global_step",
            "finished", "last_eval", "mtime"} <= set(runs[0])
    metrics = json.loads((out / "metrics_tiny_run.json").read_text())
    assert set(metrics) == {"train", "eval", "config"}
    assert len(metrics["train"]) <= ex.TRAIN_MAX_ROWS + 1
    assert metrics["train"][-1]["update"] == 400  # last row kept
    cks = json.loads((out / "checkpoints_tiny_run.json").read_text())
    assert all({"id", "update", "mtime"} == set(c) for c in cks)

    ep = json.loads((out / "episode_tiny_run__final__8.json").read_text())
    assert set(ep) == {"meta", "outcome", "total_reward", "frames"}
    assert ep["meta"]["seed"] == 8 and ep["meta"]["checkpoint"] == "final"
    assert {"run", "checkpoint", "update", "seed", "reward_version", "dt", "world", "limits"} <= set(ep["meta"])
    assert set(ep["frames"][0]) == FRAME_KEYS and ep["frames"][0]["action"] == -1
    base = json.loads((out / "baseline_8.json").read_text())
    assert base["meta"]["checkpoint"] == "baseline"


def test_deterministic_and_small(run_dir, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ex.export(run_dir, a, n_seeds=2, first_seed=1)
    ex.export(run_dir, b, n_seeds=2, first_seed=1)
    fa = sorted(p.name for p in a.iterdir())
    assert fa == sorted(p.name for p in b.iterdir())
    for n in fa:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
    assert sum(p.stat().st_size for p in a.iterdir()) < 3_000_000


def test_pick_checkpoints_and_thin():
    cks = [{"id": f"ckpt_{u:05d}", "update": u} for u in range(50, 1250, 50)] + [{"id": "final", "update": 1220}]
    assert ex.pick_checkpoints(cks, [0.1, 0.4]) == ["ckpt_00100", "ckpt_00500", "final"]
    rows = [{"update": i} for i in range(1, 1221)]
    t = ex.thin(rows, 150)
    assert len(t) <= 152 and t[0]["update"] == 1 and t[-1]["update"] == 1220
