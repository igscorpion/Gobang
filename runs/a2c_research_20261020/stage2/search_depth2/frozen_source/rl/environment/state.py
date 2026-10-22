"""Shared four-channel observation, always relative to the acting player."""

import torch


def unpack(state):
    if len(state) == 2 and (state[1] is None or isinstance(state[1], int)):
        return state
    return state, None


def state_to_tensor4(state, board_size=9):
    board, last_move = unpack(state)
    t = torch.as_tensor(board, dtype=torch.int8)
    if tuple(t.shape) != (board_size, board_size):
        raise ValueError("observation has wrong board shape")
    last = torch.zeros(board_size, board_size)
    if last_move is not None:
        last[last_move // board_size, last_move % board_size] = 1.0
    first = torch.full((board_size, board_size), float(int((t != 0).sum()) % 2 == 0))
    return torch.stack([(t == 1).float(), (t == -1).float(), last, first])


def last_move_action(last_move, board_size=9):
    return None if last_move is None else last_move[0] * board_size + last_move[1]


def observation(env):
    return (
        tuple(tuple(row) for row in env.get_state()),
        last_move_action(env.last_move, env.board_size),
    )
