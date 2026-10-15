import numpy as np
import pytest
import torch

from rocket_landing.baseline import evaluate
from rocket_landing.policy import (
    ActorCritic,
    PolicyController,
    RunningMeanStd,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("batch", [1, 32])
def test_actor_critic_shapes(batch):
    model = ActorCritic()
    obs = torch.randn(batch, 7)
    logits, value = model(obs)
    assert logits.shape == (batch, 4)
    assert value.shape == (batch,)
    action, logp, entropy, value = model.get_action_and_value(obs)
    assert action.shape == logp.shape == entropy.shape == value.shape == (batch,)
    assert action.dtype == torch.long
    assert model.greedy_action(obs).shape == (batch,)


def test_sampling_reproducible_from_generator():
    model = ActorCritic()
    obs = torch.randn(64, 7)
    a1 = model.get_action_and_value(obs, generator=torch.Generator().manual_seed(3))[0]
    a2 = model.get_action_and_value(obs, generator=torch.Generator().manual_seed(3))[0]
    assert torch.equal(a1, a2)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(loc=i, scale=1 + i, size=(n, 7)) for i, n in enumerate([5, 17, 1, 40])]
    rms = RunningMeanStd(epsilon=1e-12)
    for c in chunks:
        rms.update(c)
    allx = np.concatenate(chunks)
    np.testing.assert_allclose(rms.mean, allx.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(rms.var, allx.var(axis=0), rtol=1e-6)


def test_normalize_shapes_dtype_and_clip():
    rms = RunningMeanStd()
    rms.update(np.random.default_rng(1).normal(size=(100, 7)))
    assert rms.normalize(np.zeros(7)).shape == (7,)
    out = rms.normalize(np.full((3, 7), 1e6))
    assert out.shape == (3, 7) and out.dtype == np.float32
    assert np.all(out == 10.0)
    assert np.all(rms.normalize(np.full(7, -1e6)) == -10.0)


def test_running_mean_std_state_round_trip():
    rms = RunningMeanStd(clip=5.0)
    rms.update(np.random.default_rng(2).normal(size=(50, 7)))
    other = RunningMeanStd.from_state_dict(rms.state_dict())
    np.testing.assert_array_equal(other.mean, rms.mean)
    np.testing.assert_array_equal(other.var, rms.var)
    assert (other.count, other.clip, other.epsilon) == (rms.count, rms.clip, rms.epsilon)


def _trained_bits():
    torch.manual_seed(0)
    model = ActorCritic(hidden_sizes=(16, 16), activation="relu")
    rms = RunningMeanStd()
    rms.update(np.random.default_rng(3).normal(size=(64, 7)))
    return model, rms


def test_checkpoint_round_trip(tmp_path):
    model, rms = _trained_bits()
    path = tmp_path / "nested" / "ckpt.pt"
    save_checkpoint(path, model, rms, {"a": 1}, extra={"update": 4})
    loaded, loaded_rms, payload = load_checkpoint(path)
    obs = torch.randn(8, 7)
    assert torch.equal(model(obs)[0], loaded(obs)[0])
    np.testing.assert_array_equal(loaded_rms.mean, rms.mean)
    np.testing.assert_array_equal(loaded_rms.var, rms.var)
    assert not loaded.training
    assert payload["config"] == {"a": 1} and payload["extra"] == {"update": 4}
    assert payload["format_version"] == 1


def test_policy_controller_works_with_evaluate(tmp_path):
    model, rms = _trained_bits()
    save_checkpoint(tmp_path / "c.pt", model, rms, {})
    for deterministic in (True, False):
        ctrl = PolicyController.from_checkpoint(tmp_path / "c.pt", deterministic=deterministic)
        action = ctrl.act(np.zeros(7, dtype=np.float32))
        assert type(action) is int and 0 <= action <= 3
    count_before = rms.count
    ctrl = PolicyController(model, rms)
    summary = evaluate(ctrl, 2, seed=0)
    assert summary.n == 2
    assert rms.count == count_before
