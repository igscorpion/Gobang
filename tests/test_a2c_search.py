"""Game-semantic and budget checks for the separate A2C + search hybrid."""

from copy import deepcopy
import hashlib

import pytest
import torch

from rl.agents.a2c import A2CAgent
from rl.agents.a2c_search import SearchA2CAgent, SearchConfig, _placed
from rl.agents.tactics import TacticalConfig, winning_actions
from rl.environment.rules import legal_actions


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixedPolicy:
    def __init__(self, favorite=None, value=0.0):
        self.favorite = favorite
        self.value = value
        self.observations = []

    def policy_value(self, state):
        self.observations.append(deepcopy(state))
        logits = torch.zeros(81)
        if self.favorite is not None:
            logits[self.favorite] = 100.0
        return logits, self.value


def quiet_position():
    # Black to move; a real six-stone position with no immediate winner.
    board = [[0] * 9 for _ in range(9)]
    for action, player in ((23, -1), (31, 1), (32, 1), (33, -1), (41, 1), (70, -1)):
        board[action // 9][action % 9] = player
    return board


def make_agent(*, depth=2, nodes=2000, policy=None, tactics=None, rules="freestyle", **kwargs):
    return SearchA2CAgent(
        device="cpu", policy_agent=policy or FixedPolicy(),
        search_config=SearchConfig(depth=depth, max_nodes=nodes, **kwargs),
        tactical_config=tactics, rules=rules,
    )


def test_general_search_accounts_for_a_quiet_opponent_countermove():
    board = quiet_position()
    legal = legal_actions(board, 1)
    shallow, deeper = make_agent(depth=1), make_agent(depth=2)
    assert shallow.select_action((board, 70), legal) == 21
    assert deeper.select_action((board, 70), legal) == 13
    assert shallow.last_tactical_decision.actions == deeper.last_tactical_decision.actions
    # Reply 11 refutes the attractive static estimate of 21 without an
    # immediate win, four, or forced block. A forcing-only search misses it.
    attacked = _placed(tuple(tuple(r) for r in board), 21, 1)
    replied = _placed(attacked, 11, -1)
    for position in (attacked, replied):
        assert not winning_actions(position, 1)
        assert not winning_actions(position, -1)
    assert deeper.last_search_details["completed_root_scores"][21] < 0
    assert deeper.last_search_depth == 2
    assert deeper.last_search_details["score_is_proof"] is False


def test_root_is_confined_to_the_same_tactical_candidates_and_supplied_moves():
    board = quiet_position()
    original = deepcopy(board)
    agent = make_agent(policy=FixedPolicy(favorite=80))
    supplied = [13, 21, 43, 80]
    expected = agent.tactics.analyze((board, 70), supplied)
    chosen = agent.select_action((board, 70), supplied)
    assert chosen in expected.actions
    assert set(agent.last_search_details["root_candidates"]) == set(expected.actions)
    assert set(agent.last_search_details["completed_root_scores"]) == set(expected.actions)
    assert set(expected.actions) <= set(supplied)
    assert board == original


def test_nonroot_immediate_wins_and_complete_blocks_survive_tiny_cap():
    agent = make_agent(candidate_limit=1, tactics=TacticalConfig(enabled=False))
    board = [[0] * 9 for _ in range(9)]
    for c in (2, 3, 4, 5):
        board[4][c] = 1
    board = tuple(tuple(r) for r in board)
    own = agent._ordered_candidates(board, 1, 41, agent._legal(board, 1))
    assert set(own) == {37, 42}  # Both winning endpoints, despite cap=1.
    blocked = [[0] * 9 for _ in range(9)]
    for c in range(4):
        blocked[0][c] = -1
    blocked = tuple(tuple(r) for r in blocked)
    replies = agent._ordered_candidates(blocked, 1, 3, agent._legal(blocked, 1))
    assert replies == (4,)
    assert not winning_actions(_placed(blocked, replies[0], 1), -1)


def test_zero_budget_uses_policy_within_tactical_set_without_claiming_safety():
    board = quiet_position()
    agent = make_agent(nodes=0, policy=FixedPolicy(favorite=21))
    assert agent.select_action((board, 70), legal_actions(board, 1)) == 21
    info = agent.last_search_details
    assert info["reason"] == "policy_fallback_no_complete_iteration"
    assert info["completed_depth"] == 0
    assert info["discarded_incomplete_depth"] == 1
    assert info["budget_exhausted"]
    assert info["nodes"] == 0
    assert info["completed_root_scores"] == {}
    assert not info["score_is_proof"]


def test_incomplete_root_iteration_is_discarded_as_a_whole():
    board = quiet_position()
    state, supplied = (board, 70), [13, 21]
    no_filter = TacticalConfig(enabled=False)
    full_depth1 = make_agent(depth=1, tactics=no_filter)
    expected = full_depth1.select_action(state, supplied)
    # Two nodes finish depth 1; depth 2 gets one leaf then exhausts. It must
    # retain the full previous iteration, not mix depths or partial roots.
    limited = make_agent(depth=2, nodes=4, tactics=no_filter)
    assert limited.select_action(state, supplied) == expected
    info = limited.last_search_details
    assert info["completed_depth"] == 1
    assert info["discarded_incomplete_depth"] == 2
    assert info["budget_exhausted"]
    assert info["completed_root_scores"] == full_depth1.last_search_details["completed_root_scores"]
    assert info["nodes"] <= 4


def test_root_tactical_certificate_is_preserved_outside_general_budget():
    board = [[0] * 9 for _ in range(9)]
    for c in (2, 3, 4, 5):
        board[4][c] = 1
    for action in (0, 2, 6, 8):
        board[action // 9][action % 9] = -1
    agent = make_agent(nodes=0, policy=FixedPolicy(favorite=80))
    chosen = agent.select_action((board, 8), legal_actions(board, 1))
    assert chosen in (37, 42)
    assert agent.last_tactical_decision.proven
    assert agent.last_tactical_decision.reason == "immediate_win"
    assert agent.last_search_nodes == 0
    assert not agent.last_search_details["score_is_proof"]


@pytest.mark.parametrize("player", (1, -1))
def test_renju_legality_is_absolute_color_aware_even_for_overbroad_supplied_moves(player):
    absolute = [[0] * 9 for _ in range(9)]
    for row, col in ((4, 3), (4, 5), (3, 4), (5, 4)):
        absolute[row][col] = 1
    # Counts establish actual black (even) / white (odd) turn.
    for action in ((0, 2, 6, 8) if player == 1 else (0, 2, 6)):
        absolute[action // 9][action % 9] = -1
    relative = [[cell * player for cell in row] for row in absolute]
    policy = FixedPolicy(favorite=40)
    agent = make_agent(nodes=0, policy=policy, rules="renju", tactics=TacticalConfig(enabled=False))
    chosen = agent.select_action((relative, 49 if player == -1 else 8), [40, 42])
    assert chosen in legal_actions(absolute, player, "renju")
    assert chosen == (42 if player == 1 else 40)
    assert agent.last_tactical_decision.details["absolute_player"] == player
    assert policy.observations[0][0] == tuple(tuple(r) for r in relative)
    if player == -1:
        # Black's illegal central double-three is not a legal threat for the
        # white static evaluator to value as compulsory defensive work.
        absolute_tuple = tuple(tuple(r) for r in absolute)
        assert agent._move_score(absolute_tuple, 40, -1) == 0


def test_injected_policy_sees_consistent_to_move_and_last_move_at_every_depth():
    board = quiet_position()
    policy = FixedPolicy()
    agent = make_agent(depth=3, policy=policy, candidate_limit=2)
    agent.select_action((board, 70), [13, 21])
    assert len(policy.observations) > 2
    for relative, last in policy.observations:
        assert relative[last // 9][last % 9] == -1
        ours = sum(v == 1 for row in relative for v in row)
        others = sum(v == -1 for row in relative for v in row)
        black_turn = (ours + others) % 2 == 0
        assert ours - others == (0 if black_turn else -1)


def test_handcrafted_leaf_value_is_antisymmetric_and_rotation_preserves_scores():
    board = tuple(tuple(r) for r in quiet_position())
    agent = make_agent()
    assert agent._evaluate(board, 1, 70) == -agent._evaluate(board, -1, 70)
    rotated = tuple(tuple(board[8-c][r] for c in range(9)) for r in range(9))

    def rotate_action(action):
        row, col = divmod(action, 9)
        return col * 9 + 8-row

    agent.select_action((board, 70), legal_actions(board, 1))
    original = agent.last_search_details["completed_root_scores"]
    agent.select_action((rotated, rotate_action(70)), legal_actions(rotated, 1))
    transformed = agent.last_search_details["completed_root_scores"]
    assert set(transformed) == {rotate_action(a) for a in original}
    for action, value in original.items():
        assert transformed[rotate_action(action)] == pytest.approx(value)


def test_learned_value_blend_is_explicit_bounded_and_default_is_handcrafted():
    board = tuple(tuple(r) for r in quiet_position())
    hand = make_agent(policy=FixedPolicy(value=99.0))
    blended = make_agent(policy=FixedPolicy(value=99.0), learned_value_weight=0.25)
    raw = hand._evaluate(board, 1, 70)
    assert blended._evaluate(board, 1, 70) == pytest.approx(0.75 * raw + 0.25)
    assert hand.search_config.learned_value_weight == 0


def test_search_does_not_change_loaded_weights_or_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "a2c.pt"
    original = A2CAgent(device="cpu")
    original.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent = SearchA2CAgent(device="cpu")
    agent.load(checkpoint)
    before = {name: value.clone() for name, value in agent.net.state_dict().items()}
    board = quiet_position()
    agent.select_action((board, 70), legal_actions(board, 1))
    assert not agent.net.training
    assert all(torch.equal(tensor, agent.net.state_dict()[name]) for name, tensor in before.items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("kwargs", (
    {"depth": 0}, {"depth": 4}, {"max_nodes": -1}, {"candidate_limit": 0},
    {"max_nodes": 1.5}, {"depth": True},
    {"learned_value_weight": -0.1}, {"learned_value_weight": 1.1},
    {"static_scale": 0}, {"static_scale": float("nan")},
))
def test_invalid_search_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)


def test_training_and_stochastic_override_are_rejected():
    with pytest.raises(ValueError, match="inference-only"):
        SearchA2CAgent(device="cpu", training=True)
    with pytest.raises(ValueError, match="epsilon"):
        SearchA2CAgent(device="cpu", epsilon=0.1)
