"""Game-semantic checks for the new exact-only inference guard."""

from copy import deepcopy

import pytest

from rl.agents.a2c_guarded import GuardedA2CAgent, uses_exact_override
from rl.agents.tactics import TacticalConfig, TacticalDecision, winning_actions
from rl.environment.rules import legal_actions


class FavoritePolicy:
    training = False
    board_size = 9

    def __init__(self, favorite=0):
        self.favorite = favorite
        self.calls = []

    def select_action(self, state, candidates):
        self.calls.append(tuple(candidates))
        return self.favorite if self.favorite in candidates else min(candidates)


def board_at(own=(), opponent=()):
    board = [[0] * 9 for _ in range(9)]
    for action, player in [(a, 1) for a in own] + [(a, -1) for a in opponent]:
        board[action // 9][action % 9] = player
    return board


def test_quiet_singleton_filter_is_released_without_claiming_safety():
    board = board_at()
    original = deepcopy(board)
    policy = FavoritePolicy(0)
    agent = GuardedA2CAgent(policy, tactical_config=TacticalConfig(candidate_limit=1))
    assert agent.select_action((board, None), range(81)) == 0
    assert agent.last_source_decision.actions == (40,)
    decision = agent.last_tactical_decision
    assert decision.reason == "guarded_policy_full_legal"
    assert decision.actions == tuple(range(81))
    assert not decision.proven and "proof_scope" not in decision.details
    assert not decision.details["exact_override"]
    assert board == original


def test_immediate_win_survives_a_policy_favoring_elsewhere():
    board = board_at((38, 39, 40, 41), (0, 2, 6, 8))
    policy = FavoritePolicy(80)
    agent = GuardedA2CAgent(policy)
    assert agent.select_action((board, 8), legal_actions(board)) in (37, 42)
    decision = agent.last_tactical_decision
    assert decision.reason == "immediate_win" and decision.proven
    assert decision.details["proof_scope"]
    assert decision.details["exact_override"]
    assert 80 not in policy.calls[-1]


def test_complete_one_move_defence_is_kept_without_deeper_safety_claim():
    board = board_at((18, 20, 24, 26), (0, 1, 2, 3))
    policy = FavoritePolicy(80)
    agent = GuardedA2CAgent(policy, tactical_config=TacticalConfig(max_nodes=0))
    assert agent.select_action((board, 3), legal_actions(board)) == 4
    decision = agent.last_tactical_decision
    assert decision.reason == "block_immediate_win" and not decision.proven
    assert "no deeper safety claim" in decision.details["proof_scope"]
    board[0][4] = 1
    assert not winning_actions(board, -1)


def test_double_winning_endpoints_are_kept():
    board = board_at((38, 39, 40), (0, 2, 6))
    agent = GuardedA2CAgent(FavoritePolicy(80))
    chosen = agent.select_action((board, 6), legal_actions(board))
    assert chosen in (37, 41)
    decision = agent.last_tactical_decision
    assert decision.reason == "create_double_threat" and decision.proven
    assert len(decision.details["winning_endpoints"][str(chosen)]) >= 2


def test_continuous_four_certificate_is_kept():
    board = board_at((37, 38, 39, 22, 31), (36,))
    agent = GuardedA2CAgent(FavoritePolicy(80))
    chosen = agent.select_action((board, 36), legal_actions(board))
    decision = agent.last_tactical_decision
    assert decision.reason == "forced_win" and decision.proven
    assert str(chosen) in decision.details["proof_lines"]
    assert decision.details["proof_scope"]


@pytest.mark.parametrize("reason", ("quiet_candidates", "tactical_candidates", "defend_forcing_threat"))
def test_heuristic_labels_never_become_compulsory_even_if_marked_proven(reason):
    board = board_at()
    policy = FavoritePolicy(0)
    agent = GuardedA2CAgent(policy)
    agent.tactics.analyze = lambda *_: TacticalDecision((40,), reason, True, {"proof_scope": "bad"})
    assert agent.select_action((board, None), range(81)) == 0
    assert not agent.last_tactical_decision.proven


def test_certificate_requires_explicit_scope_and_proven_status():
    assert not uses_exact_override(TacticalDecision((40,), "forced_win", True))
    assert not uses_exact_override(TacticalDecision((40,), "forced_win", False, {"proof_scope": "x"}))
    assert uses_exact_override(TacticalDecision((40,), "unavoidable_loss", True, {"proof_scope": "all lose next ply"}))


def test_renju_forbidden_point_is_filtered_from_full_policy_actions():
    board = board_at((39, 41, 31, 49), (0, 2, 6, 8))
    policy = FavoritePolicy(40)
    agent = GuardedA2CAgent(policy, rules="renju", tactical_config=TacticalConfig(enabled=False))
    chosen = agent.select_action((board, 8), range(81))
    assert chosen in legal_actions(board, 1, "renju") and chosen != 40
    assert 40 not in policy.calls[-1]
    assert all(not board[a // 9][a % 9] for a in policy.calls[-1])


def test_full_policy_set_still_respects_supplied_actions_and_training_rejection():
    policy = FavoritePolicy(0)
    agent = GuardedA2CAgent(policy, tactical_config=TacticalConfig(enabled=False))
    assert agent.select_action((board_at(), None), (70, 80)) == 70
    assert policy.calls[-1] == (70, 80)
    with pytest.raises(ValueError, match="No legal"):
        agent.select_action((board_at(), None), ())
    policy.training = True
    with pytest.raises(ValueError, match="inference-only"):
        GuardedA2CAgent(policy)
