import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rocket_landing import evaluate as ev
from rocket_landing import selection as sl
from rocket_landing.policy import (
    ActorCritic,
    PolicyController,
    RunningMeanStd,
    load_checkpoint,
    save_checkpoint,
)
from rocket_landing.selection import Score
from rocket_landing.train import DEFAULT_CONFIG, TRAIN_SEED_BASE


# ------------------------------------------------------------------ helpers
def curve_scores(successes, n=200, margins=None):
    """Scores with updates 50, 100, ... from a list of success counts."""
    margins = margins or [0.0] * len(successes)
    return [Score(f"ckpt_{50 * (i + 1):05d}", 50 * (i + 1), k, n, m)
            for i, (k, m) in enumerate(zip(successes, margins))]


def pick(rule, successes, **kw):
    scores = curve_scores(successes, **kw)
    return sl.apply_rule(scores, rule).update // 50 - 1  # index in the original list


def make_run(base, name="r", n_ckpts=3, train_seed=0, eval_seed_start=20_000, reward_version="v2"):
    """A tiny run dir: random-init 8-unit policies as ckpt_00050.., plus final.pt (update 999) and best.pt."""
    d = base / name
    (d / "checkpoints").mkdir(parents=True)
    cfg = dict(DEFAULT_CONFIG, reward_version=reward_version, eval_seed_start=eval_seed_start)
    (d / "config.json").write_text(json.dumps({"config": cfg, "seed": train_seed}))

    def save(path, seed, update):
        torch.manual_seed(seed)
        save_checkpoint(path, ActorCritic(hidden_sizes=(8,)), RunningMeanStd(), cfg, extra={"update": update})

    for i in range(n_ckpts):
        save(d / "checkpoints" / f"ckpt_{50 * (i + 1):05d}.pt", i, 50 * (i + 1))
    save(d / "final.pt", 100, 999)
    save(d / "best.pt", 101, 75)
    return d


def load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "select_checkpoint.py"
    spec = importlib.util.spec_from_file_location("select_checkpoint_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- rule logic
def test_best_val_ties_go_to_latest():
    assert pick("best_val", [10, 200, 50, 200, 0]) == 3
    assert pick("best_val", [0, 0, 0]) == 2  # nothing learned -> latest
    assert pick("best_val", [200]) == 0


def test_rules_ignore_input_order():
    scores = curve_scores([0, 200, 0, 200, 0])
    shuffled = [scores[i] for i in (3, 0, 4, 1, 2)]
    for rule in sl.RULES:
        assert sl.apply_rule(shuffled, rule) == sl.apply_rule(scores, rule)


def test_latest_near_best_uses_wilson_overlap():
    # 200/200 has Wilson lower bound ~0.981. 196/200 (98%) upper ~0.992 overlaps -> eligible;
    # 190/200 (95%) upper ~0.974 does not.
    lo_best = sl.wilson_interval(200, 200)[0]
    assert sl.wilson_interval(196, 200)[1] >= lo_best > sl.wilson_interval(190, 200)[1]
    assert pick("latest_near_best", [200, 196, 190]) == 1
    assert pick("latest_near_best", [190, 200, 196, 150]) == 2
    assert pick("latest_near_best", [200, 190]) == 0  # the later one is distinguishable: not eligible
    assert pick("latest_near_best", [0, 0, 0, 0]) == 3  # all tied -> final


def test_stable3_rejects_lone_spike_and_prefers_plateau():
    # a spike at index 1 has bad neighbours; the plateau 4..6 wins although its max is lower
    curve = [0, 200, 0, 0, 190, 192, 188, 0]
    assert pick("best_val", curve) == 1
    assert pick("stable3", curve) == 5  # window (4,5,6) mean 190; best inside: 192


def test_stable3_hl_v2_s8_shape():
    # good for four checkpoints, collapsed for the last two: the collapsed tail must not win
    curve = [0, 100, 200, 200, 200, 200, 0, 0]
    assert pick("stable3", curve) == 5  # window (3,4,5) all 200, latest such window; latest inside
    assert pick("stable3", [0, 0, 0, 0]) == 3


def test_stable3_window_clipped_and_tie_to_latest_window():
    assert pick("stable3", [50, 150]) == 1  # fewer checkpoints than the window
    assert pick("stable3", [200]) == 0
    # equal window means: the later window wins
    assert pick("stable3", [200, 200, 200, 200]) == 3


def test_stable3_uses_pooled_rate_not_floats():
    # windows (0,1,2): 100+100+100, (1,2,3): 100+100+100 (exact tie) -> later window, latest inside
    assert pick("stable3", [100, 100, 100, 100]) == 3


def test_max_margin_picks_largest_margin_among_near_best():
    s = curve_scores([200, 200, 196, 120], margins=[0.10, 0.30, 0.50, 0.90])
    # eligible: 0, 1, 2 (120/200 is distinguishable from 200/200); margin 0.50 wins
    assert sl.apply_rule(s, "max_margin").update == 150
    s = curve_scores([200, 200], margins=[0.2, 0.2])
    assert sl.apply_rule(s, "max_margin").update == 100  # tie -> latest


def test_fuel_margin_is_p10_of_fuel_left():
    fuel_used = [0.1] * 90 + [0.9] * 10  # 10th percentile of fuel left lies between 0.1 and 0.9
    m = sl.fuel_margin(fuel_used)
    assert 0.1 - 1e-9 <= m <= 0.9 + 1e-9
    assert sl.fuel_margin([0.2] * 20) == pytest.approx(0.8)
    assert sl.fuel_margin([]) == 0.0


def test_rule_errors():
    with pytest.raises(ValueError):
        sl.apply_rule([], "best_val")
    with pytest.raises(ValueError):
        sl.apply_rule(curve_scores([1]), "nope")
    assert sl.DEFAULT_RULE in sl.RULES


def test_split_half_scores_selected_vs_other_half():
    # checkpoint A is 100% on the first half and 0% on the second, B is 50% on both:
    # selecting on one half and scoring on the other exposes the spike
    a = "L" * 100 + "C" * 100
    b = ("LC" * 50) * 2
    curve = {"episodes": 200, "checkpoints": [
        {"name": "ckpt_00050", "update": 50, "outcomes": a},
        {"name": "ckpt_00100", "update": 100, "outcomes": b},
    ]}
    # half 1: A (100%) beats B (50%) -> scored 1.0 here, 0.0 on half 2 (spike does not transfer)
    # half 2: B (50%) beats A (0%)    -> scored 0.5 here, 0.5 on half 1
    r = sl.split_half_score(curve, "best_val")
    assert r["selected_half"] == pytest.approx(0.75)
    assert r["other_half"] == pytest.approx(0.25)


# ---------------------------------------------------------------- seeds
def test_default_ranges_are_disjoint_from_everything():
    v = (sl.VALIDATION_SEED_START, sl.VALIDATION_SEED_START + sl.VALIDATION_EPISODES)
    test = (sl.TEST_SEED_START, sl.TEST_SEED_START + sl.TEST_EPISODES)
    assert sl.TEST_SEED_START == ev.DEFAULT_SEED_START
    assert test == (50_000, 50_500)
    assert v[1] <= test[0] or test[1] <= v[0]
    for other in (sl.BASELINE_TEST_SEEDS, sl.BASELINE_TUNING_SEEDS, sl.PILOT_SELECTION_SEEDS, (20_000, 20_050)):
        assert not sl._overlaps(v, other)
    assert v[1] <= TRAIN_SEED_BASE
    info = sl.assert_validation_disjoint(v[0], v[1] - v[0], 0, 16, 20_000, 50)
    assert info["validation"] == [70_000, 70_199] and info["test"] == [50_000, 50_499]


def test_validation_overlaps_are_rejected():
    ok = (70_000, 10, 0, 16, 20_000, 50)
    sl.assert_validation_disjoint(*ok)
    sl.assert_validation_disjoint(50_500, 10, 0, 16, 20_000, 50)  # adjacent to TEST on the right
    sl.assert_validation_disjoint(49_990, 10, 0, 16, 20_000, 50)  # adjacent to TEST on the left
    with pytest.raises(AssertionError):  # inside TEST
        sl.assert_validation_disjoint(50_100, 10, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # straddles the TEST start
        sl.assert_validation_disjoint(49_995, 10, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # in-training eval seeds
        sl.assert_validation_disjoint(20_040, 100, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # baseline test seeds 0..99
        sl.assert_validation_disjoint(50, 10, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # heuristic tuning seeds
        sl.assert_validation_disjoint(1_500, 10, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # stabiliser pilot-selection seeds
        sl.assert_validation_disjoint(60_100, 10, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):  # training region
        sl.assert_validation_disjoint(TRAIN_SEED_BASE, 10, 0, 16, 20_000, 50)
    with pytest.raises(AssertionError):
        sl.assert_validation_disjoint(70_000, 0, 0, 16, 20_000, 50)


def test_build_curve_rejects_run_whose_eval_seeds_overlap_validation(tmp_path):
    d = make_run(tmp_path, eval_seed_start=70_050)  # in-training eval seeds 70050..70099
    with pytest.raises(AssertionError):
        sl.build_validation_curve(d, episodes=60, procs=1)  # validation 70000..70059 overlaps
    sl.build_validation_curve(d, episodes=2, procs=1)  # 70000..70001 does not


# ------------------------------------------------------ checkpoint listing
def test_list_checkpoints_sorted_with_final_and_without_best(tmp_path):
    d = make_run(tmp_path, n_ckpts=3)
    lst = sl.list_checkpoints(d)
    assert [(u, n) for u, n, _ in lst] == [(50, "ckpt_00050"), (100, "ckpt_00100"), (150, "ckpt_00150"),
                                           (999, "final")]
    assert all(p.exists() for _, _, p in lst)
    assert "best" not in [n for _, n, _ in lst]
    with pytest.raises(FileNotFoundError):
        sl.list_checkpoints(tmp_path / "empty")


# ----------------------------------------------------------- end to end
def test_select_run_end_to_end_schema_and_export(tmp_path):
    d = make_run(tmp_path, n_ckpts=3)
    out = tmp_path / "out"
    sel = sl.select_run(d, out, rule="stable3", episodes=3, procs=1)
    on_disk = json.loads((out / "selected.json").read_text())
    assert on_disk == sel
    assert sl.SELECTED_KEYS <= set(on_disk)
    assert on_disk["schema_version"] == sl.SCHEMA_VERSION and on_disk["rule"] == "stable3"
    v = on_disk["validation"]
    assert (v["seed_start"], v["seed_end"], v["episodes"]) == (70_000, 70_002, 3)
    assert v["n_checkpoints"] == len(v["curve"]) == 4
    assert v["seed_check"]["validation"] == [70_000, 70_002]
    assert v["seed_check"]["test"] == [50_000, 50_499]
    ch = on_disk["chosen"]
    assert ch["name"] in [c["name"] for c in v["curve"]]
    assert ch["successes"] / ch["n"] == ch["rate"] and ch["ci95"][0] - 1e-9 <= ch["rate"] <= ch["ci95"][1] + 1e-9
    row = next(c for c in v["curve"] if c["name"] == ch["name"])
    assert (row["successes"], row["n"]) == (ch["successes"], ch["n"])
    assert set(on_disk["alternatives"]) == set(sl.RULES)
    assert on_disk["final"]["name"] == "final"
    # the chosen checkpoint is what the rule picks from the stored curve
    curve = json.loads((out / "validation_curve.json").read_text())
    assert sl.select(curve, "stable3")["chosen"]["name"] == ch["name"]
    for rule, alt in on_disk["alternatives"].items():
        assert alt["name"] == sl.apply_rule(sl.scores_from_curve(curve), rule).name
    # per-checkpoint rows agree with the stored per-episode outcomes
    for c in curve["checkpoints"]:
        assert len(c["outcomes"]) == len(c["fuel_used"]) == 3
        assert c["success"] == c["outcomes"].count("L") and set(c["outcomes"]) <= set("LCOT")

    # exported checkpoint is a normal, loadable checkpoint
    exported = out / "selected.pt"
    assert on_disk["exported_checkpoint"] == "selected.pt" and len(on_disk["exported_sha256"]) == 64
    ctrl = PolicyController.from_checkpoint(exported)
    obs = np.array([0.0, 100.0, 0.0, -5.0, 0.0, 0.0, 1.0], dtype=np.float32)
    assert ctrl.act(obs) in (0, 1, 2, 3)
    src_model, src_norm, src_payload = load_checkpoint(ch["path"])
    exp_model, exp_norm, exp_payload = load_checkpoint(exported)
    for k, w in src_model.state_dict().items():
        assert torch.equal(w, exp_model.state_dict()[k])
    assert PolicyController(src_model, src_norm).act(obs) == ctrl.act(obs)
    assert exp_payload["extra"]["update"] == src_payload["extra"]["update"]
    assert exp_payload["extra"]["selection"]["source_checkpoint"] == ch["name"]
    assert exp_payload["extra"]["selection"]["rule"] == "stable3"
    assert exp_payload["policy_kwargs"] == src_payload["policy_kwargs"]


def test_select_run_is_deterministic_and_reuses_cache(tmp_path, monkeypatch):
    d = make_run(tmp_path, n_ckpts=2)
    a = sl.select_run(d, tmp_path / "o1", episodes=3, procs=1)
    b = sl.select_run(d, tmp_path / "o2", episodes=3, procs=1)
    assert a["validation"]["curve"] == b["validation"]["curve"]
    assert a["chosen"]["name"] == b["chosen"]["name"]

    def boom(*args, **kw):
        raise AssertionError("re-evaluated despite a complete cache")

    monkeypatch.setattr(sl, "evaluate_many", boom)
    c = sl.select_run(d, tmp_path / "o1", rule="best_val", episodes=3, procs=1)  # cache hit, other rule
    assert c["rule"] == "best_val" and c["validation"]["curve"] == a["validation"]["curve"]
    with pytest.raises(AssertionError):  # different episode count invalidates the cache
        sl.select_run(d, tmp_path / "o1", episodes=4, procs=1)


def test_select_run_refuses_to_write_into_the_run_dir(tmp_path):
    d = make_run(tmp_path)
    with pytest.raises(ValueError):
        sl.select_run(d, d, episodes=2, procs=1)
    assert not (d / "selected.json").exists()


def test_process_pool_matches_serial(tmp_path):
    d = make_run(tmp_path, n_ckpts=2)
    paths = [p for _, _, p in sl.list_checkpoints(d)]
    serial = sl.evaluate_many(paths, 70_000, 3, "v2", procs=1)
    pooled = sl.evaluate_many(paths, 70_000, 3, "v2", procs=2)
    assert serial == pooled


def test_validation_evaluation_matches_evaluate_py(tmp_path):
    """The validation rollout is the same greedy rollout as evaluate.rollout/summarize."""
    d = make_run(tmp_path, n_ckpts=1)
    res = sl.evaluate_checkpoint(d / "final.pt", 70_000, 4, "v2")
    model, norm, _ = load_checkpoint(d / "final.pt")
    ref = ev.summarize(ev.rollout(lambda s: PolicyController(model, norm), range(70_000, 70_004), None, "v2"))
    c = sl.counts(res["outcomes"], res["fuel_used"])
    assert c["success"] == ref["success"]["count"] and c["crash"] == ref["crash"]["count"]
    assert c["ci95"] == pytest.approx(ref["success"]["ci95"])
    assert c["fuel_used_mean"] == pytest.approx(ref["fuel_used"]["mean"], abs=1e-3)


def test_cli(tmp_path, capsys):
    d = make_run(tmp_path, n_ckpts=2)
    out = tmp_path / "cli_out"
    rc = load_cli().main(["--run", str(d), "--out-dir", str(out), "--episodes", "3", "--procs", "1",
                          "--rule", "latest_near_best"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "validation seeds 70000..70002" in text and "<- chosen" in text
    sel = json.loads((out / "selected.json").read_text())
    assert sel["rule"] == "latest_near_best"
    assert (out / "selected.pt").exists() and (out / "validation_curve.json").exists()
    assert not (d / "selected.json").exists()


# ------------------------------------------------- best.pt in the candidate pool
def test_list_checkpoints_include_best_adds_it_by_update_and_dedups(tmp_path):
    d = make_run(tmp_path, n_ckpts=3)  # best.pt is update 75: between ckpt_00050 and ckpt_00100
    lst = sl.list_checkpoints(d, include_best=True)
    assert [(u, n) for u, n, _ in lst] == [(50, "ckpt_00050"), (75, "best"), (100, "ckpt_00100"),
                                           (150, "ckpt_00150"), (999, "final")]
    # a best.pt that is the same update as a saved checkpoint is not a second candidate
    torch.manual_seed(5)
    save_checkpoint(d / "best.pt", ActorCritic(hidden_sizes=(8,)), RunningMeanStd(), {}, extra={"update": 100})
    assert "best" not in [n for _, n, _ in sl.list_checkpoints(d, include_best=True)]
    (d / "best.pt").unlink()
    assert [n for _, n, _ in sl.list_checkpoints(d, include_best=True)] == [n for _, n, _ in sl.list_checkpoints(d)]


def test_validation_cache_is_incremental_when_best_is_added(tmp_path, monkeypatch):
    d = make_run(tmp_path, n_ckpts=2)
    out = tmp_path / "o"
    a = sl.select_run(d, out, episodes=3, procs=1)
    assert a["pool"] == "saved" and a["validation"]["n_checkpoints"] == 3
    evaluated = []
    real = sl.evaluate_many

    def spy(paths, *args, **kw):
        evaluated.extend(Path(p).name for p in paths)
        return real(paths, *args, **kw)

    monkeypatch.setattr(sl, "evaluate_many", spy)
    b = sl.select_run(d, out, episodes=3, procs=1, include_best=True)
    assert evaluated == ["best.pt"]  # only the new candidate was rolled out
    assert b["pool"] == "saved+best" and b["validation"]["n_checkpoints"] == 4
    names = [c["name"] for c in b["validation"]["curve"]]
    assert names == ["ckpt_00050", "best", "ckpt_00100", "final"]  # sorted by update (best is 75)
    # old rows are unchanged
    old = {c["name"]: c for c in a["validation"]["curve"]}
    for c in b["validation"]["curve"]:
        if c["name"] in old:
            assert c == old[c["name"]]
    evaluated.clear()
    sl.select_run(d, out, episodes=3, procs=1, include_best=True)
    assert evaluated == []  # complete cache


def test_fresh_test_range_assertion():
    ok = (sl.TEST2_SEED_START, 500, 0, 16, 20_000, 50)
    info = sl.assert_test_range_fresh(*ok)
    assert info["test_range"] == [80_000, 80_499]
    for bad_start in (sl.VALIDATION_SEED_START, sl.TEST_SEED_START, 20_010, 60_100, 50, 1_500):
        with pytest.raises(AssertionError):
            sl.assert_test_range_fresh(bad_start, 500 if bad_start != 20_010 else 100, 0, 16, 20_000, 50)
    # validation and the two test ranges are pairwise disjoint
    ranges = [(sl.VALIDATION_SEED_START, sl.VALIDATION_SEED_START + sl.VALIDATION_EPISODES),
              (sl.TEST_SEED_START, sl.TEST_SEED_START + sl.TEST_EPISODES),
              (sl.TEST2_SEED_START, sl.TEST2_SEED_START + sl.TEST_EPISODES)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not sl._overlaps(ranges[i], ranges[j])
