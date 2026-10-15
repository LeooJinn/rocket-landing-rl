import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

from rocket_landing.policy import ActorCritic, RunningMeanStd, save_checkpoint
from rocket_landing.recording import (
    load_policy_from_checkpoint,
    record_baseline_episode,
    record_episode,
)

_spec = importlib.util.spec_from_file_location(
    "serve_viewer", Path(__file__).resolve().parents[1] / "scripts" / "serve_viewer.py"
)
sv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sv)

FRAME_KEYS = {"t", "x", "y", "vx", "vy", "angle", "av", "fuel", "action", "reward", "cum_reward"}


def _ckpt(path, update, seed=0):
    torch.manual_seed(seed)
    save_checkpoint(path, ActorCritic(), RunningMeanStd(), {"reward_version": "v1"}, extra={"update": update})


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    run = root / "run_a"
    (run / "checkpoints").mkdir(parents=True)
    _ckpt(run / "checkpoints" / "ckpt_00050.pt", 50)
    _ckpt(run / "checkpoints" / "ckpt_00100.pt", 100, seed=1)
    _ckpt(run / "best.pt", 50)
    (run / "config.json").write_text(
        json.dumps({"config": {"reward_version": "v1", "total_timesteps": 1000}, "seed": 3})
    )
    rows = [{"update": i, "global_step": i * 10} for i in (1, 2)]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, rows)) + '\n{"update": 3, "glo')
    (run / "eval.jsonl").write_text(
        json.dumps({"update": 2, "success_rate": 0.5, "crash_rate": 0.5, "out_of_bounds_rate": 0.0,
                    "timeout_rate": 0.0, "mean_reward": -1.0}) + "\n"
    )
    (root / "empty_run").mkdir()
    return root


def _check_schema(t):
    assert set(t) == {"meta", "outcome", "total_reward", "frames"}
    assert t["outcome"] in {"landed", "crashed", "out_of_bounds", "timeout"}
    f0 = t["frames"][0]
    assert set(f0) == FRAME_KEYS and f0["action"] == -1 and f0["reward"] == 0 and f0["t"] == 0
    assert all(set(f) == FRAME_KEYS for f in t["frames"])
    assert t["frames"][-1]["cum_reward"] == pytest.approx(t["total_reward"])
    m = t["meta"]
    assert m["world"].keys() == {"half_width", "height", "pad_half_width"}
    assert m["limits"].keys() == {"safe_vx", "safe_vy", "safe_angle", "safe_angular_velocity"}
    assert m["dt"] == 0.05
    json.dumps(t)


def test_policy_episode_schema_and_determinism(results):
    ctrl = load_policy_from_checkpoint(results / "run_a" / "checkpoints" / "ckpt_00050.pt")
    a = record_episode(ctrl, 5, checkpoint="ckpt_00050", run="run_a", update=50)
    b = record_episode(ctrl, 5)
    _check_schema(a)
    assert a["meta"]["run"] == "run_a" and a["meta"]["seed"] == 5
    assert a["frames"] == b["frames"]
    assert record_episode(ctrl, 6)["frames"][0] != a["frames"][0]


def test_baseline_episode():
    t = record_baseline_episode(1)
    _check_schema(t)
    assert t["meta"]["checkpoint"] == "baseline" and t["meta"]["run"] is None
    assert t["outcome"] == "landed"


def test_list_runs(results):
    runs = sv.list_runs(results)
    assert {r["name"] for r in runs} == {"run_a", "empty_run"}
    r = next(r for r in runs if r["name"] == "run_a")
    assert r["updates"] == 2 and r["global_step"] == 20 and r["seed"] == 3
    assert r["reward_version"] == "v1" and r["total_timesteps"] == 1000
    assert r["finished"] is False and r["last_eval"]["success_rate"] == 0.5
    assert [x["mtime"] for x in runs] == sorted((x["mtime"] for x in runs), reverse=True)


def test_metrics_tolerates_partial_line(results):
    m = sv.run_metrics(results, "run_a")
    assert len(m["train"]) == 2 and len(m["eval"]) == 1 and m["config"]["seed"] == 3


def test_checkpoints_and_latest(results):
    ids = [c["id"] for c in sv.run_checkpoints(results, "run_a")]
    assert ids == ["latest", "best", "ckpt_00050", "ckpt_00100"]
    assert sv.run_checkpoints(results, "run_a")[0]["update"] == 100
    t = sv.run_episode(results, "run_a", 20000, "latest")
    assert t["meta"]["update"] == 100 and t["meta"]["checkpoint"] == "latest"
    assert sv.run_episode(results, "run_a", 20000, "latest") is t  # cached


def test_no_checkpoint_404(results):
    with pytest.raises(sv.ApiError) as e:
        sv.route(results, "/api/runs/empty_run/episode", {})
    assert e.value.status == 404


@pytest.mark.parametrize("path", ["/api/runs/..%2F/metrics", "/api/runs/../metrics",
                                  "/api/runs/%2e%2e/checkpoints", "/api/runs/nope/metrics"])
def test_traversal_rejected(results, path):
    with pytest.raises(sv.ApiError) as e:
        sv.route(results, path, {})
    assert e.value.status == 404


def test_bad_seed_and_checkpoint(results):
    with pytest.raises(sv.ApiError) as e:
        sv.route(results, "/api/runs/run_a/episode", {"seed": ["1;rm"]})
    assert e.value.status == 400
    with pytest.raises(sv.ApiError) as e:
        sv.route(results, "/api/runs/run_a/episode", {"checkpoint": ["../../x"]})
    assert e.value.status == 400


def test_http_server(results, tmp_path):
    fe = tmp_path / "fe"
    fe.mkdir()
    (fe / "index.html").write_text("<h1>hi</h1>")
    (tmp_path / "secret.txt").write_text("s")
    srv = sv.make_server(results, 0, fe)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert urllib.request.urlopen(base + "/").read() == b"<h1>hi</h1>"
        runs = json.load(urllib.request.urlopen(base + "/api/runs"))
        assert len(runs) == 2
        t = json.load(urllib.request.urlopen(base + "/api/baseline/episode?seed=2"))
        assert t["meta"]["checkpoint"] == "baseline"
        for bad in ("/../secret.txt", "/%2e%2e/secret.txt", "/api/runs/nope/metrics"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + bad)
            assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/runs/empty_run/episode")
        assert e.value.code == 404 and "error" in json.load(e.value)
    finally:
        srv.shutdown()
        srv.server_close()
