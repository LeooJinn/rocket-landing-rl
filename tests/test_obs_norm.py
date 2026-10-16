import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from rocket_landing.policy import (
    FIXED_OBS_SCALES,
    ActorCritic,
    FixedScaleNormalizer,
    PolicyController,
    RunningMeanStd,
    load_checkpoint,
    make_obs_normalizer,
    save_checkpoint,
)
from rocket_landing.train import load_config, resolve_config, train

SMOKE = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "smoke.json"


def test_fixed_is_deterministic_and_never_updated():
    n = FixedScaleNormalizer()
    x = np.array([50, 100, 10, -10, 0.3, -0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(n.normalize(x), [1, 1, 1, -1, 1, -1, 1], rtol=1e-6)
    before = n.state_dict()
    n.update(np.random.default_rng(0).normal(size=(100, 7)) * 1000)
    assert n.state_dict() == before
    np.testing.assert_array_equal(n.normalize(x), n.normalize(x))
    batch = np.stack([x, x])
    assert n.normalize(batch).shape == (2, 7)
    assert n.normalize(np.full(7, 1e9)).max() == 10.0  # clipped


def test_factory_and_default_config():
    assert isinstance(make_obs_normalizer("running"), RunningMeanStd)
    assert isinstance(make_obs_normalizer("fixed"), FixedScaleNormalizer)
    with pytest.raises(ValueError):
        make_obs_normalizer("bogus")
    assert resolve_config({})["obs_norm"] == "fixed"
    assert resolve_config({"obs_norm": "running"})["obs_norm"] == "running"


def test_fixed_checkpoint_round_trip_and_controller(tmp_path):
    torch.manual_seed(0)
    model = ActorCritic()
    save_checkpoint(tmp_path / "c.pt", model, FixedScaleNormalizer(), {"obs_norm": "fixed"})
    _, norm, _ = load_checkpoint(tmp_path / "c.pt")
    assert isinstance(norm, FixedScaleNormalizer)
    np.testing.assert_array_equal(norm.scales, FIXED_OBS_SCALES)
    ctrl = PolicyController.from_checkpoint(tmp_path / "c.pt")
    assert 0 <= ctrl.act(np.zeros(7, dtype=np.float32)) <= 3


def test_legacy_running_checkpoint_still_loads(tmp_path):
    # A pre-change checkpoint stores the bare RunningMeanStd dict (no "type" key).
    rms = RunningMeanStd()
    rms.update(np.random.default_rng(1).normal(size=(10, 7)))
    save_checkpoint(tmp_path / "old.pt", ActorCritic(), rms, {})
    payload = torch.load(tmp_path / "old.pt", weights_only=True)
    assert "type" not in payload["obs_norm"]
    _, norm, _ = load_checkpoint(tmp_path / "old.pt")
    assert isinstance(norm, RunningMeanStd)
    np.testing.assert_array_equal(norm.mean, rms.mean)


def test_training_with_fixed_norm_saves_fixed_and_does_not_adapt(tmp_path):
    cfg = copy.deepcopy(load_config(SMOKE))
    cfg["obs_norm"] = "fixed"
    train(cfg, 0, tmp_path / "r")
    _, norm, payload = load_checkpoint(tmp_path / "r" / "final.pt")
    assert isinstance(norm, FixedScaleNormalizer)
    assert norm.state_dict() == FixedScaleNormalizer().state_dict()
    assert payload["config"]["obs_norm"] == "fixed"
