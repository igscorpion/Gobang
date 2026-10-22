"""Game-semantic checks for the optional A2C tactical inference layer.

These fixtures ask whether returned moves actually win or avert the stated
threat under the shared game rules; they do not reproduce the search heuristic.
"""

from copy import deepcopy

import pytest

from rl.agents.tactics import TacticalConfig, TacticalEngine, analyze_move
from rl.environment.rules import forbidden_reason, legal_actions, winning_move


SIZE = 9
SPARSE = (
    (0, 0), (8, 8), (0, 8), (8, 0), (0, 2), (8, 6), (2, 0),
    (6, 8), (2, 8), (6, 0), (0, 6), (8, 2), (1, 1), (7, 7),
    (1, 7), (7, 1), (0, 4), (8, 4),
)


def action(row, col):
    return row * SIZE + col


def position(own=(), opponent=(), *, black=True):
    """A relative board with actual black/white move counts, padding sparsely."""
    board = [[0] * SIZE for _ in range(SIZE)]
    for value, cells in ((1, own), (-1, opponent)):
        for row, col in cells:
            assert board[row][col] == 0, "fixture has overlapping stones"
            board[row][col] = value
    target_delta = 0 if black else -1  # own minus opponent stones
    own_count, other_count = len(own), len(opponent)
    while own_count - other_count != target_delta:
        value = 1 if own_count - other_count < target_delta else -1
        cell = next((p for p in SPARSE if board[p[0]][p[1]] == 0), None)
        assert cell is not None, "fixture needs too many filler stones"
        board[cell[0]][cell[1]] = value
        own_count += value == 1
        other_count += value == -1
    assert sum(v != 0 for row in board for v in row) % 2 == (not black)
    return board


def absolute(board):
    color = 1 if sum(v != 0 for row in board for v in row) % 2 == 0 else -1
    return [[v * color for v in row] for row in board], color


def wins(board, player, rules="freestyle"):
    """Independent one-ply oracle: enumerate legal moves through game rules."""
    result = set()
    for candidate in legal_actions(board, player, rules):
        after = deepcopy(board)
        row, col = divmod(candidate, SIZE)
        after[row][col] = player
        if winning_move(after, row, col, player, rules):
            result.add(candidate)
    return result


def decide(board, *, rules="freestyle", supplied=None, config=None, last=None):
    actual, player = absolute(board)
    supplied = legal_actions(actual, player, rules) if supplied is None else supplied
    state = board if last is None else (board, last)
    return TacticalEngine(config=config, rules=rules, board_size=SIZE).analyze(
        state, supplied
    )


def assert_fork_is_forced(board, candidate, rules="freestyle"):
    actual, player = absolute(board)
    row, col = divmod(candidate, SIZE)
    actual[row][col] = player
    assert not wins(actual, -player, rules), "opponent can win before the fork"
    assert len(wins(actual, player, rules)) >= 2
    # Every legal reply must leave at least one immediate legal win.
    for reply in legal_actions(actual, -player, rules):
        after = deepcopy(actual)
        row, col = divmod(reply, SIZE)
        after[row][col] = -player
        assert wins(after, player, rules)


def test_take_immediate_win_instead_of_mechanically_defending():
    board = position(
        own=[(2, c) for c in range(4)],
        opponent=[(6, c) for c in range(4)],
    )
    decision = decide(board)
    assert decision.reason == "immediate_win"
    assert decision.proven
    assert set(decision.actions) == {action(2, 4)}


@pytest.mark.parametrize(
    "stones,expected",
    [
        ([(0, c) for c in range(4)], action(0, 4)),
        ([(4, c) for c in (1, 2, 4, 5)], action(4, 3)),
        ([(r, 8) for r in (1, 2, 3, 5)], action(4, 8)),
        ([(r, r) for r in (1, 2, 4, 5)], action(3, 3)),
    ],
)
def test_block_unique_edge_broken_vertical_and_diagonal_four(stones, expected):
    board = position(opponent=stones)
    actual, player = absolute(board)
    assert wins(actual, -player) == {expected}, "fixture must have a unique threat"
    assert not wins(actual, player)
    decision = decide(board)
    assert decision.reason == "block_immediate_win"
    assert decision.actions == (expected,)


def test_two_distinct_immediate_win_points_are_honestly_unavoidable():
    board = position(opponent=[(4, c) for c in (2, 3, 4, 5)])
    actual, player = absolute(board)
    assert wins(actual, -player) == {action(4, 1), action(4, 6)}
    assert not wins(actual, player)
    decision = decide(board)
    assert decision.reason == "unavoidable_loss"
    assert decision.proven
    assert decision.actions
    # It may choose a best practical block, but must not claim that move saves.
    for candidate in decision.actions:
        after = deepcopy(actual)
        row, col = divmod(candidate, SIZE)
        after[row][col] = player
        assert wins(after, -player)


def test_create_double_winning_endpoints_and_verify_every_opponent_reply():
    board = position(own=[(4, c) for c in (2, 3, 4)])
    decision = decide(board)
    assert decision.reason in ("create_double_threat", "forced_win")
    assert decision.proven and decision.actions
    for candidate in decision.actions:
        assert_fork_is_forced(board, candidate)


@pytest.mark.parametrize("rules", ("freestyle", "renju"))
def test_counterattack_fork_precedes_mechanical_block_of_an_open_three(rules):
    board = position(
        own=[(2, c) for c in (2, 3, 4)],
        opponent=[(6, c) for c in (3, 4, 5)],
    )
    decision = decide(board, rules=rules)
    assert decision.proven
    assert set(decision.actions).issubset({action(2, 1), action(2, 5)})
    for candidate in decision.actions:
        assert_fork_is_forced(board, candidate, rules)


def test_next_ply_loss_is_blocked_before_starting_an_own_fork():
    board = position(
        own=[(2, c) for c in (2, 3, 4)],
        opponent=[(6, c) for c in range(4)],
    )
    decision = decide(board, config=TacticalConfig(candidate_limit=1, max_nodes=1))
    assert decision.actions == (action(6, 4),)
    assert decision.reason == "block_immediate_win"


def test_continuous_four_proof_covers_forced_block_then_a_fork():
    board = position(
        own=[(4, 1), (4, 2), (4, 3), (2, 4), (3, 4)],
        opponent=[(4, 0)],
    )
    decision = decide(board)
    assert decision.reason == "forced_win" and decision.proven
    assert action(4, 4) in decision.actions
    # Independent game-rule proof of the five-ply combination. Root makes a
    # one-ended four; all non-blocking replies lose immediately. After its
    # forced block, a vertical open four wins against every legal reply.
    actual, player = absolute(board)
    actual[4][4] = player
    assert not wins(actual, -player)
    assert wins(actual, player) == {action(4, 5)}
    actual[4][5] = -player
    assert not wins(actual, -player)
    relative_after_block = [[value * player for value in row] for row in actual]
    assert_fork_is_forced(relative_after_block, action(5, 4))


def test_insufficient_vcf_budget_does_not_claim_the_five_ply_combination():
    board = position(
        own=[(4, 1), (4, 2), (4, 3), (2, 4), (3, 4)],
        opponent=[(4, 0)],
    )
    decision = decide(
        board, config=TacticalConfig(forcing_depth=5, max_nodes=1, candidate_limit=12)
    )
    assert not decision.proven
    assert decision.reason != "forced_win"


def test_prevent_opponent_open_four_when_no_faster_counterattack_exists():
    board = position(opponent=[(4, c) for c in (3, 4, 5)])
    decision = decide(board)
    assert decision.actions
    assert set(decision.actions).issubset({action(4, 2), action(4, 6)})
    assert not decision.proven, "defending a threat is not proof of winning"


@pytest.mark.parametrize(
    "stones,candidate,real,broken",
    [
        ([(4, 3), (4, 5)], action(4, 4), 1, False),
        ([(4, 0), (4, 1)], action(4, 2), 0, False),
        ([(4, 2), (4, 5)], action(4, 4), 1, True),
    ],
)
def test_real_open_three_broken_three_and_edge_pseudo_three(stones, candidate, real, broken):
    board = position(own=stones)
    actual, player = absolute(board)
    analysis = analyze_move(actual, candidate, player=player, rules="renju")
    assert analysis["legal"]
    assert analysis["real_open_threes"] == real
    if broken:
        assert analysis["broken_threes"] >= 1


def test_forbidden_extensions_do_not_turn_pseudo_three_into_real_three():
    # Extending the horizontal XXX at c2/c6 would make another vertical four:
    # both extensions are black double-four fouls. Only the vertical three is real.
    stones = [(4, 3), (4, 5), (3, 4), (5, 4)] + [
        (row, col) for row in (1, 2, 3) for col in (2, 6)
    ]
    board = position(own=stones)
    actual, player = absolute(board)
    analysis = analyze_move(actual, 40, player=player, rules="renju")
    assert analysis["legal"]
    assert analysis["real_open_threes"] == 1
    assert not analysis["double_three"]


def test_four_with_one_forbidden_endpoint_is_not_a_two_ended_open_four():
    board = position(own=[(4, col) for col in (1, 2, 3, 6)])
    actual, player = absolute(board)
    analysis = analyze_move(actual, 40, player=player, rules="renju")
    assert analysis["legal"]
    assert set(analysis["winning_actions"]) == {action(4, 0)}
    assert analysis["open_fours"] == 0
    assert not analysis["double_threat"]


@pytest.mark.parametrize(
    "own,reason",
    [
        ([(4, 3), (4, 5), (3, 4), (5, 4)], "double_three"),
        ([(4, 2), (4, 3), (4, 5), (2, 4), (3, 4), (5, 4)], "double_four"),
        ([(4, c) for c in (1, 2, 3, 5, 6)], "overline"),
    ],
)
def test_black_forbidden_move_filtered_even_if_caller_supplies_it(own, reason):
    board = position(own=own, black=True)
    actual, player = absolute(board)
    assert forbidden_reason(actual, 40, player, "renju") == reason
    supplied = [i for i in range(81) if not board[i // 9][i % 9]]
    assert 40 in supplied
    decision = decide(board, rules="renju", supplied=supplied)
    legal = set(legal_actions(actual, player, "renju"))
    assert decision.actions and set(decision.actions).issubset(legal)
    assert 40 not in decision.actions


def test_caller_supplies_only_a_forbidden_move_gets_no_legal_candidates():
    board = position(own=[(4, 3), (4, 5), (3, 4), (5, 4)])
    decision = decide(board, rules="renju", supplied=[40])
    assert decision.actions == ()
    assert not decision.proven


def test_white_overline_wins_and_is_not_filtered_as_a_black_foul():
    board = position(own=[(4, c) for c in (1, 2, 3, 5, 6)], black=False)
    actual, player = absolute(board)
    assert player == -1
    assert wins(actual, player, "renju") == {40}
    decision = decide(board, rules="renju")
    assert decision.reason == "immediate_win"
    assert decision.proven and decision.actions == (40,)


def test_exact_five_takes_precedence_over_simultaneous_black_overline():
    board = position(
        own=[(4, c) for c in (1, 2, 3, 5, 6)] + [(r, 4) for r in (0, 1, 2, 3)],
        black=True,
    )
    actual, player = absolute(board)
    assert forbidden_reason(actual, 40, player, "renju") is None
    assert 40 in wins(actual, player, "renju")
    decision = decide(board, rules="renju")
    assert decision.reason == "immediate_win" and decision.proven
    assert 40 in decision.actions


def test_opponents_black_overline_is_not_treated_as_an_immediate_win():
    board = position(opponent=[(4, c) for c in (1, 2, 3, 5, 6)], black=False)
    actual, player = absolute(board)
    assert player == -1 and forbidden_reason(actual, 40, -player, "renju") == "overline"
    assert not wins(actual, -player, "renju")
    decision = decide(board, rules="renju")
    assert decision.reason not in ("block_immediate_win", "unavoidable_loss")


def test_only_supplied_legal_actions_are_returned_and_unique_choice_is_honored():
    board = position(own=[(4, 4)], opponent=[(4, 5)])
    for supplied in ([action(1, 7)], [action(1, 7), action(7, 1)]):
        decision = decide(board, supplied=supplied)
        assert decision.actions and set(decision.actions).issubset(supplied)
    assert decide(board, supplied=[]).actions == ()


def test_board_and_last_action_are_not_changed_by_repeated_analysis():
    board = position(opponent=[(4, c) for c in (1, 2, 4, 5)])
    before = deepcopy(board)
    last = action(4, 5)
    engine = TacticalEngine(rules="freestyle", board_size=9)
    supplied = [i for i in range(81) if not board[i // 9][i % 9]]
    supplied_before = supplied[:]
    state = (board, last)
    first = engine.analyze(state, supplied)
    second = engine.analyze(state, supplied)
    assert (first.actions, first.reason, first.proven) == (
        second.actions, second.reason, second.proven
    )
    assert board == before and state[1] == last and supplied == supplied_before


def transform(board, turns, mirror):
    result = deepcopy(board)
    if mirror:
        result = [row[::-1] for row in result]
    for _ in range(turns):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def transform_action(candidate, turns, mirror):
    marker = [[0] * 9 for _ in range(9)]
    row, col = divmod(candidate, 9)
    marker[row][col] = 1
    transformed = transform(marker, turns, mirror)
    return next(i for i in range(81) if transformed[i // 9][i % 9])


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("mirror", (False, True))
def test_forced_unique_defense_is_rotation_and_reflection_equivariant(turns, mirror):
    board = position(opponent=[(4, c) for c in (1, 2, 4, 5)])
    decision = decide(transform(board, turns, mirror))
    assert decision.actions == (transform_action(action(4, 3), turns, mirror),)
    assert decision.reason == "block_immediate_win"


def test_exhausted_search_budget_does_not_invent_a_forced_win():
    board = position(own=[(4, 4)], opponent=[(3, 4)])
    decision = decide(
        board, config=TacticalConfig(forcing_depth=1, max_nodes=1, candidate_limit=1)
    )
    assert decision.actions
    assert not decision.proven
    assert decision.reason not in ("forced_win", "create_double_threat", "unavoidable_loss")


def test_disabled_tactics_retains_supplied_candidates_without_a_proof():
    board = position(own=[(4, 4)], opponent=[(4, 5)])
    supplied = [action(1, 7), action(7, 1)]
    decision = decide(board, supplied=supplied, config=TacticalConfig(enabled=False))
    assert set(decision.actions) == set(supplied)
    assert not decision.proven


def test_wrapper_uses_immediate_tactical_win_even_when_network_prefers_defense():
    import torch
    from rl.agents.a2c_tactical import TacticalA2CAgent

    class BadPreference(torch.nn.Module):
        def forward(self, observations):
            logits = torch.zeros((len(observations), 81), device=observations.device)
            logits[:, action(6, 4)] = 100
            return logits, torch.zeros(len(observations), device=observations.device)

    board = position(
        own=[(2, c) for c in range(4)],
        opponent=[(6, c) for c in range(4)],
    )
    actual, player = absolute(board)
    agent = TacticalA2CAgent(device="cpu")
    agent.net = BadPreference()
    chosen = agent.select_action((board, action(6, 3)), legal_actions(actual, player))
    assert chosen == action(2, 4)
    assert agent.last_tactical_decision.reason == "immediate_win"


def test_disabled_wrapper_is_same_policy_as_plain_a2c_with_same_weights():
    import torch
    from rl.agents.a2c import A2CAgent
    from rl.agents.a2c_tactical import TacticalA2CAgent

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        plain = A2CAgent(device="cpu", seed=11)
        hybrid = TacticalA2CAgent(
            device="cpu", seed=11, tactical_config=TacticalConfig(enabled=False)
        )
        hybrid.net.load_state_dict(plain.net.state_dict())
        board = position(own=[(4, 4)], opponent=[(4, 5)])
        actual, player = absolute(board)
        legal = legal_actions(actual, player)
        state = (board, action(4, 5))
        assert hybrid.select_action(state, legal) == plain.select_action(state, legal)
        assert sum(p.numel() for p in hybrid.net.parameters()) == 1_405_042
    finally:
        torch.set_num_threads(previous_threads)


def test_wrapper_rejects_on_policy_learning_before_and_after_construction():
    from rl.agents.a2c_tactical import TacticalA2CAgent

    with pytest.raises(ValueError, match="inference-only"):
        TacticalA2CAgent(device="cpu", training=True)
    agent = TacticalA2CAgent(device="cpu")
    agent.training = True
    board = [[0] * 9 for _ in range(9)]
    with pytest.raises(ValueError, match="learning policy"):
        agent.select_action(board, list(range(81)))
