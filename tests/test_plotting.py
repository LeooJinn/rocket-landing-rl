import json

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="needs the 'plots' extra: uv sync --extra plots")
matplotlib.use("Agg")

from rocket_landing import plotting as P  # noqa: E402


def _cfg(version="v1", seed=0, gamma=0.999):
    return {
        "config": {
            "total_timesteps": 1000,
            "reward_version": version,
            "ppo": {"gamma": gamma, "ent_coef": 0.03},
            "policy": {"hidden_sizes": [64, 64]},
        },
        "seed": seed,
    }


def make_run(root, name, version="v1", seed=0, n_evals=6, succ=None, finished=True, truncate=False):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(_cfg(version, seed)))
    metrics = [
        {"update": i, "global_step": i * 100, "entropy": 1.3 - 0.05 * i, "value_loss": 1.0 / i,
         "ep_reward_mean": None if i == 1 else -50.0 + i}
        for i in range(1, 11)
    ]
    succ = succ if succ is not None else [i / n_evals for i in range(n_evals)]
    evals = [
        {"update": (i + 1) * 2, "global_step": (i + 1) * 200, "success_rate": s,
         "crash_rate": 1 - s, "out_of_bounds_rate": 0.0, "timeout_rate": 0.0, "mean_reward": -100 + 200 * s}
        for i, s in enumerate(succ)
    ]
    (d / "metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    text = "".join(json.dumps(e) + "\n" for e in evals)
    if truncate:
        text += '{"update": 99, "global_st'  # partially written last line
    (d / "eval.jsonl").write_text(text)
    if finished:
        (d / "summary.json").write_text(json.dumps({"seed": seed}))
    return d


def test_load_jsonl_skips_partial_and_bad_lines(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n\nnot json\n[1,2]\n{"a": 2}\n{"a": 3')
    assert P.load_jsonl(f) == [{"a": 1}, {"a": 2}]
    assert P.load_jsonl(tmp_path / "missing.jsonl") == []


def test_load_run_fields_and_progress(tmp_path):
    make_run(tmp_path, "done", seed=3, succ=[0.0, 0.5, 1.0, 0.2], finished=True)
    make_run(tmp_path, "live", seed=4, succ=[0.1, 0.9], finished=False, truncate=True)
    done, live = P.load_run(tmp_path / "done"), P.load_run(tmp_path / "live")
    assert done.seed == 3 and not done.in_progress
    assert done.config["ppo.gamma"] == 0.999 and done.config["reward_version"] == "v1"
    assert done.final_success == 0.2 and done.best_success == 1.0  # final != best
    assert live.in_progress and len(live.evals) == 2  # truncated line dropped
    x, y = done.series("metrics", "ep_reward_mean")
    assert len(x) == 9  # None dropped


def test_load_run_without_files(tmp_path):
    (tmp_path / "empty").mkdir()
    r = P.load_run(tmp_path / "empty")
    assert r.in_progress and r.final_success is None and r.best_success is None
    assert P.load_runs(tmp_path) == []  # directory without run files is not a run


def test_grouping_and_common_mean(tmp_path):
    make_run(tmp_path, "a1", "v1", seed=1, succ=[0.0, 1.0, 1.0])
    make_run(tmp_path, "a0", "v1", seed=0, succ=[0.0, 0.0, 0.0, 0.0], finished=False)
    make_run(tmp_path, "b0", "v2", seed=0, succ=[1.0, 1.0, 1.0])
    groups = P.group_runs(P.load_runs(tmp_path), ["reward_version"])
    assert list(groups) == ["reward_version=v1", "reward_version=v2"]
    assert [r.seed for r in groups["reward_version=v1"]] == [0, 1]
    steps, mean = P.common_mean(groups["reward_version=v1"], "evals", "success_rate")
    assert steps.tolist() == [200, 400, 600]  # truncated to what all seeds reached
    assert mean.tolist() == [0.0, 0.5, 0.5]
    rows = P.final_table(groups)
    assert {r["run"]: r["final_success"] for r in rows}["a1"] == 1.0


def test_plot_run_creates_png(tmp_path):
    make_run(tmp_path, "r", finished=False, truncate=True)
    out = P.plot_run(P.load_run(tmp_path / "r"), tmp_path / "figs" / "r.png")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_run_empty_run_does_not_crash(tmp_path):
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "metrics.jsonl").write_text('{"update": 1, "glo')
    assert P.plot_run(P.load_run(tmp_path / "e"), tmp_path / "e.png").exists()


def test_plot_comparison_creates_png(tmp_path):
    for s in range(3):
        make_run(tmp_path, f"v1_s{s}", "v1", seed=s, succ=[0, 0.2 * s, 0.1, 0.0])
        make_run(tmp_path, f"v2_s{s}", "v2", seed=s, succ=[0, 0.5, 1.0, 0.9 - 0.4 * s], finished=s != 2)
    groups = P.group_runs(P.load_runs(tmp_path), ["reward_version"])
    out = P.plot_comparison(groups, tmp_path / "cmp.png", title="t")
    assert out.exists() and out.stat().st_size > 1000


def test_cli_script(tmp_path):
    import runpy
    import sys

    make_run(tmp_path / "res", "x_s0")
    argv = ["plot_training.py", "--results", str(tmp_path / "res"), "--out-dir", str(tmp_path / "o"),
            "--run", "x_s0", "--compare", "x_*"]
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path("scripts/plot_training.py", run_name="__main__")
    finally:
        sys.argv = old
    assert (tmp_path / "o" / "run_x_s0.png").exists() and (tmp_path / "o" / "comparison.png").exists()
