"""A2C contracts: zero-sum targets, legal policies, detached losses and models."""

import math

import pytest
import torch

from rl.agents.a2c import A2CAgent, ActorCriticNetwork
from rl.agents.dqn import QNetwork
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation
from rl.training.a2c_math import (
    a2c_loss,
    masked_distribution,
    signed_n_step_returns,
)
from rl.training.common import evaluate


@pytest.fixture(scope="module", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_targets_match_hand_calculated_win_loss_draw_bootstrap_and_reset():
    gamma = 0.9
    # Columns: self-play win; fixed-opponent loss; draw; same-player
    # truncation; opposite-player truncation; terminal followed by a new game.
    rewards = torch.tensor(
        [[0, 0, 0, 0.1, 0, 1], [0, 0, 0, 0.2, 0, 0], [1, -1, 0, 0.3, 0, -1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    signs = torch.tensor([[-1, 1, -1, 1, -1, -1]] * 3)
    dones = torch.tensor(
        [[False] * 5 + [True], [False] * 6, [True, True, True, False, False, True]]
    )
    bootstrap = torch.tensor(
        [99, 99, 99, 0.7, 0.8, 99], dtype=torch.float64, requires_grad=True
    )
    targets = signed_n_step_returns(
        rewards, dones, signs, bootstrap, torch.ones_like(dones), gamma=gamma
    )
    expected = torch.tensor(
        [
            [
                gamma**2,
                -(gamma**2),
                0,
                0.1 + gamma * (0.2 + gamma * (0.3 + gamma * 0.7)),
                -(gamma**3) * 0.8,
                1,
            ],
            [
                -gamma,
                -gamma,
                0,
                0.2 + gamma * (0.3 + gamma * 0.7),
                gamma**2 * 0.8,
                gamma,
            ],
            [1, -1, 0, 0.3 + gamma * 0.7, -gamma * 0.8, -1],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(targets, expected)
    assert not targets.requires_grad and targets.grad_fn is None


def test_variable_padding_neither_advances_bootstrap_nor_uses_padding_reward():
    rewards = torch.tensor(
        [[0.0, 0.2, 900], [0, 900, 900], [1, 0.4, 900], [900, 900, 900]]
    )
    valid = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.bool
    )
    signs = torch.tensor([[-1, 1, -1]] * 4)
    dones = torch.tensor(
        [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.bool
    )
    targets = signed_n_step_returns(
        rewards, dones, signs, torch.tensor([0.0, 0.8, 0.9]), valid, gamma=0.9
    )
    expected = torch.tensor(
        [
            [0.81, 0.2 + 0.9 * (0.4 + 0.9 * 0.8), 0],
            [-0.9, 0, 0],
            [1, 0.4 + 0.9 * 0.8, 0],
            [0, 0, 0],
        ]
    )
    torch.testing.assert_close(targets, expected)


def test_masked_policy_has_exact_zero_illegal_mass_and_finite_entropy_gradient():
    logits = torch.tensor(
        [[1000.0, 100000.0, -1000.0], [0, 99, 0]], requires_grad=True
    )
    mask = torch.tensor([[True, False, True], [True, False, True]])
    policy = masked_distribution(logits, mask)
    assert torch.equal(policy.probs[:, 1], torch.zeros(2))
    torch.testing.assert_close(policy.probs.sum(-1), torch.ones(2))
    assert torch.isfinite(policy.entropy()).all()
    samples = policy.sample((100,))
    assert bool((samples != 1).all())
    loss, metrics = a2c_loss(
        logits,
        torch.tensor([0.1, -0.3], requires_grad=True),
        torch.tensor([0, 2]),
        torch.tensor([1.0, -1.0]),
        mask,
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))
    assert all(math.isfinite(value) for value in metrics.values())


def test_loss_detaches_return_target_and_policy_advantage():
    logits = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    values = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    returns = torch.tensor([0.75], dtype=torch.float64, requires_grad=True)
    loss, metrics = a2c_loss(
        logits,
        values,
        torch.tensor([0]),
        returns,
        torch.ones(1, 2, dtype=torch.bool),
        value_coef=0.5,
        entropy_coef=0,
    )
    assert loss.item() == pytest.approx(0.5 * math.log(2) + 0.5 * 0.5**2)
    assert metrics["policy_loss"] == pytest.approx(0.5 * math.log(2))
    assert metrics["value_loss"] == pytest.approx(0.5**2)
    loss.backward()
    torch.testing.assert_close(
        logits.grad, torch.tensor([[-0.25, 0.25]], dtype=torch.float64)
    )
    torch.testing.assert_close(values.grad, torch.tensor([-0.5], dtype=torch.float64))
    assert returns.grad is None


def test_policy_rejects_all_masked_rows_and_illegal_training_actions():
    with pytest.raises(ValueError, match="legal"):
        masked_distribution(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="legal"):
        a2c_loss(
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.tensor([1]),
            torch.ones(1),
            torch.tensor([[True, False]]),
        )


def test_network_capacity_and_four_channel_policy_value_shapes():
    network = ActorCriticNetwork()
    assert network.input_channels == 4 and network.channels == (32, 64, 64)
    assert network.hidden_units == 256
    assert sum(parameter.numel() for parameter in network.parameters()) == 1_405_042
    assert (
        sum(parameter.numel() for parameter in network.parameters())
        - sum(parameter.numel() for parameter in QNetwork().parameters())
        == 257
    )
    small = ActorCriticNetwork(hidden_units=64)
    assert sum(parameter.numel() for parameter in small.parameters()) == 393_778
    logits, values = network(torch.zeros(2, 4, 9, 9))
    assert logits.shape == (2, 81) and values.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def test_agent_roundtrip_restores_architecture_and_frozen_legal_argmax(tmp_path):
    agent = A2CAgent(device="cpu", seed=11)
    agent.net = ActorCriticNetwork(hidden_units=64).eval()
    with torch.no_grad():
        for parameter in agent.net.parameters():
            parameter.zero_()
        agent.net.policy_head.bias[40] = 100  # Illegal action cannot win argmax.
        agent.net.policy_head.bias[5] = 3
        agent.net.policy_head.bias[7] = 2
    state = observation(GomokuEnv())
    rng_before = agent.rng.getstate()
    assert [agent.select_action(state, [5, 7]) for _ in range(4)] == [5] * 4
    assert agent.rng.getstate() == rng_before
    path = tmp_path / "policy.pt"
    agent.save(path)
    restored = A2CAgent(device="cpu", seed=22)
    restored.load(path)
    assert restored.needs_last_move and not restored.net.training
    assert restored.net.hidden_units == 64
    assert restored.net.channels == (32, 64, 64)
    assert all(
        torch.equal(value, restored.net.state_dict()[name])
        for name, value in agent.net.state_dict().items()
    )
    assert restored.select_action(state, [7, 5]) == 5
    with pytest.raises(ValueError, match="legal"):
        restored.select_action(state, [])


def test_training_agent_samples_legal_policy_instead_of_greedy_action():
    agent = A2CAgent(device="cpu", seed=11, training=True)
    with torch.no_grad():
        for parameter in agent.net.parameters():
            parameter.zero_()
    # Equal legal probabilities should sample both actions with this fixed RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        actions = {
            agent.select_action(observation(GomokuEnv()), [5, 7]) for _ in range(64)
        }
    assert actions == {5, 7}


def test_evaluation_freezes_sampling_and_restores_training_state(monkeypatch):
    agent = A2CAgent(device="cpu", seed=11, training=True, epsilon=0.8)
    rng_before = agent.rng.getstate()
    torch_before = torch.get_rng_state().clone()
    weights = {name: value.clone() for name, value in agent.net.state_dict().items()}
    modes = []
    select = agent.select_action

    def record_mode(state, legal_actions):
        modes.append((agent.training, agent.net.training, agent.epsilon))
        return select(state, legal_actions)

    monkeypatch.setattr(agent, "select_action", record_mode)
    rows = evaluate(agent, games=1)
    assert len(rows) == 4 and modes
    assert set(modes) == {(False, False, 0.0)}
    assert agent.training and agent.net.training and agent.epsilon == 0.8
    assert agent.rng.getstate() == rng_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert all(
        torch.equal(value, agent.net.state_dict()[name])
        for name, value in weights.items()
    )
