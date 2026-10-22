"""The hybrid is a separate registry option using unchanged A2C weights."""

import hashlib

import pytest
import torch

from rl.agents import registry
from rl.agents.a2c import A2CAgent
from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation


def test_registry_keeps_raw_policy_and_loads_same_weights_for_rule_aware_hybrid(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    checkpoint = tmp_path / "model.pt"
    original = A2CAgent(device="cpu", seed=22)
    original.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(registry, "_local_models", lambda: {
        "a2c4": {"algorithm": "a2c", "checkpoint": "model.pt"},
    })
    monkeypatch.setenv("GOMOKU_AI_DEVICE", "cpu")
    monkeypatch.setenv("GOMOKU_RULES", "renju")
    assert "a2c4_tactical" in registry.available_agents()
    raw = registry.create_agent("a2c4")
    hybrid = registry.create_agent("a2c4_tactical")
    assert type(raw) is A2CAgent
    assert isinstance(hybrid, TacticalA2CAgent)
    for key, tensor in raw.net.state_dict().items():
        assert torch.equal(tensor, hybrid.net.state_dict()[key])
    env = GomokuEnv(rules="renju")
    for row, col in ((4, 3), (4, 5), (3, 4), (5, 4)):
        env.board[row][col] = 1
    for row, col in ((0, 0), (0, 2), (0, 4), (0, 6)):
        env.board[row][col] = -1
    supplied = [a for a in range(81) if env.board[a // 9][a % 9] == 0]
    assert 40 in supplied and 40 not in env.get_legal_actions()
    chosen = hybrid.select_action(observation(env), supplied)
    assert chosen in env.get_legal_actions()
    assert hybrid.last_tactical_decision is not None
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest


def test_hybrid_is_not_advertised_without_the_selected_source_model(monkeypatch):
    monkeypatch.setattr(registry, "_local_models", lambda: {})
    assert "a2c4_tactical" not in registry.available_agents()
    with pytest.raises(ValueError, match="requires"):
        registry.create_agent("a2c4_tactical")
