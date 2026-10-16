"""Entropy / learning-rate schedules and stochastic-eval logging (stabilisation options).

All new options default to the previous behaviour; these tests pin that and the new maths.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from rocket_landing.policy import ActorCritic
from rocket_landing.ppo import PPOConfig, ent_coef_at, lr_at, ppo_update
from rocket_landing.train import load_config, resolve_config, train

SMOKE = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "smoke.json"


# ------------------------------------------------------------------ learning rate
def test_default_lr_is_original_linear_anneal_exactly():
    cfg = PPOConfig(learning_rate=3e-4)
    n = 1220
    for u in (1, 2, 100, 610, 1219, 1220):
        assert lr_at(cfg, u, n) == 3e-4 * (1.0 - (u - 1) / n)  # bit-identical to the old formula


def test_constant_lr_when_not_annealed():
    cfg = PPOConfig(learning_rate=1e-3, anneal_lr=False, lr_final_frac=0.1, lr_schedule="cosine")
    assert {lr_at(cfg, u, 50) for u in range(1, 51)} == {1e-3}


def test_lr_final_frac_linear_interpolates_to_floor():
    cfg = PPOConfig(learning_rate=1e-3, lr_final_frac=0.2)
    n = 100
    assert lr_at(cfg, 1, n) == pytest.approx(1e-3)
    # progress is (u-1)/n so the last update sits one step short of the floor
    assert lr_at(cfg, n, n) == pytest.approx(1e-3 * (0.2 + 0.8 * (1 / n)))
    lrs = [lr_at(cfg, u, n) for u in range(1, n + 1)]
    assert all(a > b for a, b in zip(lrs, lrs[1:]))  # strictly decreasing
    assert min(lrs) >= 0.2e-3 - 1e-12  # never below the floor


def test_cosine_lr_shape():
    cfg = PPOConfig(learning_rate=1.0, lr_schedule="cosine", lr_final_frac=0.1)
    n = 100
    assert lr_at(cfg, 1, n) == pytest.approx(1.0)
    # halfway (progress 0.5): cos factor 0.5 -> 0.1 + 0.9 * 0.5
    assert lr_at(cfg, 51, n) == pytest.approx(0.1 + 0.9 * 0.5)
    lrs = [lr_at(cfg, u, n) for u in range(1, n + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    assert lrs[-1] >= 0.1


# --------------------------------------------------------------------- entropy
def test_entropy_constant_by_default():
    cfg = PPOConfig(ent_coef=0.03)
    assert {ent_coef_at(cfg, u, 1220) for u in (1, 500, 1220)} == {0.03}


def test_entropy_linear_decay_endpoints_and_monotone():
    cfg = PPOConfig(ent_coef=0.03, ent_coef_final=0.0)
    n = 101
    assert ent_coef_at(cfg, 1, n) == 0.03
    assert ent_coef_at(cfg, n, n) == 0.0
    assert ent_coef_at(cfg, 51, n) == pytest.approx(0.015)
    vals = [ent_coef_at(cfg, u, n) for u in range(1, n + 1)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_entropy_hold_then_decay_then_hold_final():
    # hold 0.03 until 50% of training, decay to 0.003 by 80%, then stay at 0.003
    cfg = PPOConfig(ent_coef=0.03, ent_coef_final=0.003, ent_decay_start=0.5, ent_decay_end=0.8)
    n = 101  # progress = (u - 1) / 100
    assert ent_coef_at(cfg, 1, n) == 0.03
    assert ent_coef_at(cfg, 51, n) == 0.03  # progress 0.5: decay not started
    assert ent_coef_at(cfg, 66, n) == pytest.approx(0.03 + (0.003 - 0.03) * 0.5)  # progress 0.65
    assert ent_coef_at(cfg, 81, n) == pytest.approx(0.003)  # progress 0.8
    assert ent_coef_at(cfg, 101, n) == 0.003


def test_entropy_single_update_is_safe():
    cfg = PPOConfig(ent_coef=0.03, ent_coef_final=0.0)
    assert ent_coef_at(cfg, 1, 1) == 0.03


def test_invalid_schedule_options_rejected():
    for bad in (
        {"lr_schedule": "step"},
        {"lr_final_frac": -0.1},
        {"lr_final_frac": 1.5},
        {"ent_decay_start": 0.9, "ent_decay_end": 0.1},
        {"ent_decay_end": 1.5},
        {"ent_coef_final": -1.0},
    ):
        with pytest.raises(ValueError):
            PPOConfig.from_dict(bad)
    cfg = PPOConfig.from_dict({"ent_coef_final": 0.0, "lr_schedule": "cosine"})
    assert cfg.ent_coef_final == 0.0 and cfg.lr_schedule == "cosine"


def _step(ent_coef):
    torch.manual_seed(0)
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(7)
    obs = torch.randn(64, 7, generator=g)
    with torch.no_grad():
        actions, logp, _, values = model.get_action_and_value(obs, generator=g)
    batch = {
        "obs": obs, "actions": actions, "logprobs": logp, "values": values,
        "advantages": torch.zeros(64), "returns": values.clone(),
    }  # fmt: skip
    cfg = PPOConfig(update_epochs=2, num_minibatches=2, ent_coef=0.5)
    ppo_update(model, opt, batch, cfg, torch.Generator().manual_seed(1), ent_coef=ent_coef)
    return [p.detach().clone() for p in model.parameters()]


def test_ppo_update_ent_coef_override():
    default = _step(None)  # falls back to cfg.ent_coef = 0.5
    same = _step(0.5)
    other = _step(0.0)
    assert all(torch.equal(a, b) for a, b in zip(default, same))
    assert any(not torch.equal(a, b) for a, b in zip(default, other))


# ------------------------------------------------------------------ train.py wiring
def test_resolve_config_defaults_and_unknown_keys():
    cfg = resolve_config({})
    assert cfg["eval_stochastic"] is False
    assert cfg["ppo"]["ent_coef_final"] is None and cfg["ppo"]["lr_final_frac"] == 0.0
    with pytest.raises(ValueError):
        resolve_config({"ppo": {"ent_decay": 1}})  # typo of a real key


def test_train_logs_schedule_and_stochastic_eval(tmp_path):
    base = load_config(SMOKE)
    base["total_timesteps"] = 128 * 4
    base["ppo"]["ent_coef"] = 0.04
    base["ppo"]["ent_coef_final"] = 0.0
    base["ppo"]["lr_final_frac"] = 0.5
    base["eval_stochastic"] = True
    train(base, 0, tmp_path / "on")
    rows = [json.loads(line) for line in (tmp_path / "on" / "metrics.jsonl").read_text().splitlines()]
    assert [r["ent_coef"] for r in rows][0] == 0.04
    assert rows[-1]["ent_coef"] == 0.0
    assert rows[0]["lr"] == pytest.approx(3e-4)
    assert rows[-1]["lr"] >= 0.5 * 3e-4
    ev = [json.loads(line) for line in (tmp_path / "on" / "eval.jsonl").read_text().splitlines()]
    assert len(ev) == 4
    for e in ev:
        for k in ("stoch_success_rate", "stoch_crash_rate", "stoch_out_of_bounds_rate",
                  "stoch_timeout_rate", "stoch_mean_reward"):  # fmt: skip
            assert k in e
        assert math.isclose(
            e["stoch_success_rate"] + e["stoch_crash_rate"]
            + e["stoch_out_of_bounds_rate"] + e["stoch_timeout_rate"], 1.0)  # fmt: skip

    # off by default: eval.jsonl keeps exactly the old keys
    base["eval_stochastic"] = False
    train(base, 0, tmp_path / "off")
    ev_off = json.loads((tmp_path / "off" / "eval.jsonl").read_text().splitlines()[0])
    assert not any(k.startswith("stoch_") for k in ev_off)
    assert {"update", "global_step", "success_rate", "crash_rate", "mean_reward"} <= set(ev_off)
    # greedy results are identical with or without the extra stochastic eval
    ev_off_all = [json.loads(l) for l in (tmp_path / "off" / "eval.jsonl").read_text().splitlines()]
    for a, b in zip(ev, ev_off_all):
        assert {k: v for k, v in a.items() if not k.startswith("stoch_")} == b
