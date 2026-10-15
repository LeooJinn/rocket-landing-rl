import math

import pytest
import torch

from rocket_landing.policy import ActorCritic
from rocket_landing.ppo import PPOConfig, compute_gae, ppo_update


def test_gae_hand_computed():
    # gamma = 0.5, lam = 0.5 so gamma * lam = 0.25. T = 3, N = 2.
    # Env 0: r = [1, 2, 3], V = [0.5, 1, 2], done = [0, 1, 0], V_T = 4.
    #   t=2: delta = 3 + 0.5*4 - 2 = 3;            A2 = 3
    #   t=1: done, delta = 2 + 0 - 1 = 1;          A1 = 1 (no carry from A2)
    #   t=0: delta = 1 + 0.5*1 - 0.5 = 1;          A0 = 1 + 0.25*1 = 1.25
    #   returns = A + V = [1.75, 2, 5]
    # Env 1: r = [0, 0, 1], V = [1, 2, 3], done = [0, 0, 0], V_T = 2.
    #   t=2: delta = 1 + 0.5*2 - 3 = -1;           A2 = -1
    #   t=1: delta = 0 + 0.5*3 - 2 = -0.5;         A1 = -0.5 + 0.25*(-1) = -0.75
    #   t=0: delta = 0 + 0.5*2 - 1 = 0;            A0 = 0 + 0.25*(-0.75) = -0.1875
    #   returns = [0.8125, 1.25, 2]
    rewards = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0]])
    values = torch.tensor([[0.5, 1.0], [1.0, 2.0], [2.0, 3.0]])
    dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    adv, ret = compute_gae(rewards, values, dones, torch.tensor([4.0, 2.0]), 0.5, 0.5)
    torch.testing.assert_close(adv, torch.tensor([[1.25, -0.1875], [1.0, -0.75], [3.0, -1.0]]))
    torch.testing.assert_close(ret, torch.tensor([[1.75, 0.8125], [2.0, 1.25], [5.0, 2.0]]))


def test_gae_gamma_lambda_one_is_reward_to_go():
    # Zero values and bootstrap: advantage is the undiscounted reward-to-go.
    # No done: [1+2+3, 2+3, 3] = [6, 5, 3]. Done at t=1: [1+2, 2, 3] = [3, 2, 3].
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    zeros = torch.zeros(3, 1)
    adv, ret = compute_gae(rewards, zeros, zeros, torch.zeros(1), 1.0, 1.0)
    torch.testing.assert_close(adv, torch.tensor([[6.0], [5.0], [3.0]]))
    torch.testing.assert_close(ret, adv)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    adv, _ = compute_gae(rewards, zeros, dones, torch.zeros(1), 1.0, 1.0)
    torch.testing.assert_close(adv, torch.tensor([[3.0], [2.0], [3.0]]))


def test_config_from_dict_rejects_unknown():
    assert PPOConfig.from_dict({"gamma": 0.9}).gamma == 0.9
    with pytest.raises(ValueError):
        PPOConfig.from_dict({"gamma": 0.9, "bogus": 1})


def _batch(model: ActorCritic, b: int = 64, zero_adv: bool = False) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    obs = torch.randn(b, 7, generator=g)
    with torch.no_grad():
        actions, logp, _, values = model.get_action_and_value(obs, generator=g)
    adv = torch.zeros(b) if zero_adv else torch.randn(b, generator=g)
    return {
        "obs": obs,
        "actions": actions,
        "logprobs": logp,
        "values": values,
        "advantages": adv,
        "returns": values + torch.randn(b, generator=g),
    }


def _run(zero_adv: bool = False, **cfg_kw):
    torch.manual_seed(0)
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = PPOConfig(update_epochs=3, num_minibatches=4, **cfg_kw)
    stats = ppo_update(model, opt, _batch(model, zero_adv=zero_adv), cfg, torch.Generator().manual_seed(1))
    return model, stats


def test_ppo_update_finite_stats():
    _, stats = _run()
    assert set(stats) == {
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
        "explained_variance", "grad_norm", "epochs_run",
    }  # fmt: skip
    assert all(math.isfinite(v) for v in stats.values())
    assert stats["epochs_run"] == 3
    assert 0 < stats["entropy"] <= math.log(4) + 1e-6


def test_ppo_update_deterministic():
    m1, s1 = _run()
    m2, s2 = _run()
    assert s1 == s2
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_zero_advantages_give_zero_policy_loss():
    _, stats = _run(zero_adv=True)
    assert abs(stats["policy_loss"]) < 1e-6


def test_target_kl_stops_early():
    _, stats = _run(target_kl=0.0)
    assert stats["epochs_run"] == 1
