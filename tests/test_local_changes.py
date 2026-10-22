"""Semantic checks for rules, migration, TD perspective and search legality."""

import copy
import pytest
import torch
from rl.environment.rules import forbidden_reason, winning_move
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation, state_to_tensor4
from rl.agents.alphazero import AlphaZeroNetwork, AlphaZeroAgent, _legal_actions, _move
from rl.agents.dqn import QNetwork, DQNAgent
from rl.training.local_alphazero import transfer_weights
from rl.training.local_dqn import double_dqn_target
from rl.training.common import (
    evaluate,
    append_csv,
    prepare_resume_logs,
    check_resume_config,
)


def board(points, color=1):
    b = [[0] * 9 for _ in range(9)]
    for r, c in points:
        b[r][c] = color
    return b


def rotate(b):
    return [list(row) for row in zip(*b[::-1])]


@pytest.mark.parametrize("turns", range(4))
def test_double_three_rotations_and_no_mutation(turns):
    b = board([(4, 3), (4, 5), (3, 4), (5, 4)])
    for _ in range(turns):
        b = rotate(b)
    old = copy.deepcopy(b)
    assert forbidden_reason(b, 40) == "double_three"
    assert b == old
    assert forbidden_reason(b, 40, -1) is None
    assert forbidden_reason(b, 40, rules="freestyle") is None


def test_broken_threes_and_edge_pseudo_three():
    b = board([(4, 2), (4, 5), (2, 4), (5, 4)])
    assert forbidden_reason(b, 40) == "double_three"
    # Horizontal three touches edge; cannot become a two-ended straight four.
    b = board([(4, 0), (4, 1), (3, 2), (5, 2)])
    assert forbidden_reason(b, 38) is None


def test_double_four_and_single_four_not_double_counted():
    b = board([(4, 2), (4, 3), (4, 5), (2, 4), (3, 4), (5, 4)])
    assert forbidden_reason(b, 40) == "double_four"
    b = board([(4, 2), (4, 3), (4, 5)])
    assert forbidden_reason(b, 40) is None


def test_overline_exact_five_and_white_exemption():
    b = board([(4, c) for c in (1, 2, 3, 5, 6)])
    assert forbidden_reason(b, 40) == "overline"
    assert forbidden_reason(b, 40, player=-1) is None
    # RIF 9.2: simultaneous exact five wins, including alongside an overline.
    for r in (0, 1, 2, 3):
        b[r][4] = 1
    assert forbidden_reason(b, 40) is None
    b[4][4] = 1
    assert winning_move(b, 4, 4, 1, "renju")
    b = board([(4, c) for c in range(6)], color=-1)
    assert winning_move(b, 4, 3, -1, "renju")


def test_same_direction_double_four_and_illegal_three_extensions():
    b = board([(4, c) for c in (1, 3, 5, 7)])
    assert forbidden_reason(b, 40) == "double_four"
    b = board(
        [(4, 3), (4, 5), (3, 4), (5, 4)] + [(r, c) for r in (1, 2, 3) for c in (2, 6)]
    )
    # The horizontal XXX can only extend at (4,2)/(4,6), both double-four moves.
    # Therefore only the vertical three is real: the center is legal.
    assert forbidden_reason(b, 40) is None


def test_environment_rejection_is_atomic_and_rules_optional():
    e = GomokuEnv(rules="renju")
    e.board = board([(4, 3), (4, 5), (3, 4), (5, 4)])
    before = e.get_board()
    assert 40 not in e.get_legal_actions()
    with pytest.raises(ValueError, match="double_three"):
        e.step(40)
    assert e.get_board() == before and e.current_player == 1
    e = GomokuEnv(rules="freestyle")
    e.board = before
    e.step(40)
    assert e.board[4][4] == 1


def test_shared_four_channel_and_alphazero_parameters():
    e = GomokuEnv()
    e.step(10)
    x = state_to_tensor4(observation(e))
    assert x.shape == (4, 9, 9) and x[1, 1, 1] == 1 and x[2, 1, 1] == 1
    assert x[3].sum() == 0
    old = AlphaZeroNetwork(channels=(64, 64, 64))
    wide = AlphaZeroNetwork()
    assert sum(p.numel() for p in old.parameters()) == 94935
    assert sum(p.numel() for p in wide.parameters()) == 132055
    p, v = wide(torch.zeros(2, 4, 9, 9))
    assert p.shape == (2, 81) and v.shape == (2, 1)


def test_warm_transfer_preserves_old_slices_and_keeps_new_random(tmp_path):
    old = AlphaZeroNetwork(channels=(64, 64, 64))
    path = tmp_path / "old.pt"
    torch.save(old.state_dict(), path)
    wide = AlphaZeroNetwork()
    new_slices = wide.conv[4].weight[64:].detach().clone()
    transfer_weights(wide, path)
    assert torch.equal(wide.conv[0].weight, old.conv[0].weight)
    assert torch.equal(wide.conv[4].weight[:64], old.conv[4].weight)
    assert torch.equal(wide.conv[4].weight[64:], new_slices)
    assert torch.equal(wide.policy_head[3].weight, old.policy_head[3].weight)


class Constant(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("v", torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.v.expand(len(x), -1)


def test_double_dqn_legal_mask_sign_and_terminal():
    online = Constant([2, 100, 3])
    target = Constant([7, 999, 11])
    x = torch.zeros(4, 4, 9, 9)
    mask = torch.tensor([[True, False, True]] * 3 + [[False, False, False]])
    out = double_dqn_target(
        online,
        target,
        x,
        mask,
        torch.tensor([0.0, 0.0, 1.0, -1.0]),
        torch.tensor([False, False, True, True]),
        torch.tensor([1.0, -1.0, 1.0, -1.0]),
    )
    assert torch.allclose(out, torch.tensor([10.89, -10.89, 1.0, -1.0]))


def test_neural_model_serialization_and_legacy_loading(tmp_path):
    for channels in (3, 4):
        a = DQNAgent(device="cpu", input_channels=channels)
        path = tmp_path / f"d{channels}.pt"
        a.save(path)
        b = DQNAgent(device="cpu")
        b.load(path)
        assert b.net.input_channels == channels
    for channels in ((64, 64, 64), (64, 64, 128)):
        a = AlphaZeroAgent(device="cpu", channels=channels, num_sims=4)
        path = tmp_path / f"a{channels[-1]}.pt"
        a.save(path)
        b = AlphaZeroAgent(device="cpu", num_sims=4)
        b.load(path)
        assert b.net.channels == channels


def test_mcts_matches_environment_rule_mask():
    e = GomokuEnv(rules="renju")
    e.board = board([(4, 3), (4, 5), (3, 4), (5, 4)])
    for r, c in [(0, 0), (0, 2), (0, 4), (0, 6)]:
        e.board[r][c] = -1
    s = observation(e)
    assert _legal_actions(s[0], 9, "renju") == e.get_legal_actions()
    a = AlphaZeroAgent(device="cpu", rules="renju", num_sims=4)
    action = a.select_action(s, e.get_legal_actions())
    assert action != 40
    next_board, won = _move(s[0], action, rules="renju")
    e.step(action)
    assert next_board == tuple(tuple(row) for row in e.get_state())


def test_evaluation_is_frozen_and_preserves_rng():
    torch.set_num_threads(1)
    a = DQNAgent(device="cpu", seed=4, epsilon=0.5)
    state = a.rng.getstate()
    weights = {k: v.clone() for k, v in a.net.state_dict().items()}
    rows = evaluate(a, games=1)
    assert len(rows) == 4 and a.epsilon == 0.5 and a.rng.getstate() == state
    assert all(torch.equal(v, a.net.state_dict()[k]) for k, v in weights.items())


def test_resume_discards_uncheckpointed_log_rows_and_rejects_changed_seed(tmp_path):
    import csv

    for episode in (1, 2, 3):
        append_csv(
            tmp_path / "training.csv", {"episode": episode, "loss": episode / 10}
        )
    prepare_resume_logs(tmp_path, 2)
    with (tmp_path / "training.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["episode"] for r in rows] == ["1", "2"]
    check_resume_config({"seed": 11}, {"seed": 11}, ("seed",))
    with pytest.raises(ValueError, match="seed"):
        check_resume_config({"seed": 11}, {"seed": 22}, ("seed",))


def test_suite_lock_rejects_a_second_writer_and_releases(tmp_path):
    from rl.training.common import exclusive_run

    path = tmp_path / "suite.lock"
    with exclusive_run(path):
        with pytest.raises(RuntimeError, match="already running"):
            with exclusive_run(path):
                pass
    with exclusive_run(path):
        pass
