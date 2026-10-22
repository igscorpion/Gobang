"""Shared move legality for freestyle and black forbidden-move rules.

This is a 9x9 experimental Renju restriction, without tournament opening rules.
Real open threes require a legal extension to a straight, two-ended four.
Four threats are deduplicated by their four stones, not by winning endpoints.
"""

from functools import lru_cache
import os

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def rule_mode(value=None):
    value = value or os.environ.get("GOMOKU_RULES", "freestyle")
    if value not in ("freestyle", "renju"):
        raise ValueError("rules must be freestyle or renju")
    return value


def _inside(board, p):
    return 0 <= p[0] < len(board) and 0 <= p[1] < len(board)


def _length(board, anchor, direction, player=1):
    total = 1
    for sign in (-1, 1):
        r, c = anchor
        while True:
            r += sign * direction[0]
            c += sign * direction[1]
            if not _inside(board, (r, c)) or board[r][c] != player:
                break
            total += 1
    return total


def winning_move(board, row, col, player, rules="freestyle", need=5):
    lengths = [_length(board, (row, col), d, player) for d in DIRECTIONS]
    if rules == "renju" and player == 1:
        return 5 in lengths
    return any(n >= need for n in lengths)


def _placed(board, p):
    result = [list(row) for row in board]
    result[p[0]][p[1]] = 1
    return tuple(tuple(row) for row in result)


def _four_groups(board, anchor):
    groups = {}
    for dr, dc in DIRECTIONS:
        for offset in range(-4, 1):
            cells = [
                (anchor[0] + (offset + k) * dr, anchor[1] + (offset + k) * dc)
                for k in range(5)
            ]
            if not all(_inside(board, p) for p in cells):
                continue
            black = frozenset(p for p in cells if board[p[0]][p[1]] == 1)
            empty = [p for p in cells if board[p[0]][p[1]] == 0]
            if len(black) != 4 or len(empty) != 1 or anchor not in black:
                continue
            after = _placed(board, empty[0])
            # An overline is not a valid winning endpoint.
            if winning_move(after, *empty[0], 1, "renju"):
                groups.setdefault(black, set()).add(empty[0])
    return groups


@lru_cache(maxsize=100000)
def _forbidden_after(board, anchor):
    lengths = [_length(board, anchor, d) for d in DIRECTIONS]
    if 5 in lengths:  # RIF 9.2: five takes precedence over all forbidden moves.
        return None
    if max(lengths) >= 6:
        return "overline"
    if len(_four_groups(board, anchor)) >= 2:
        return "double_four"
    threes = set()
    for dr, dc in DIRECTIONS:
        # A straight four reachable in one move must be within three cells.
        for delta in range(-3, 4):
            if delta == 0:
                continue
            extension = (anchor[0] + delta * dr, anchor[1] + delta * dc)
            if not _inside(board, extension) or board[extension[0]][extension[1]]:
                continue
            after = _placed(board, extension)
            if winning_move(after, *extension, 1, "renju"):
                continue
            for stones, endpoints in _four_groups(after, anchor).items():
                if extension not in stones or len(endpoints) != 2:
                    continue
                # Two endpoints alone are insufficient if the extension is forbidden.
                if _forbidden_after(after, extension) is None:
                    threes.add(stones - {extension})
                    if len(threes) >= 2:
                        return "double_three"
    return None


def forbidden_reason(board, action, player=1, rules="renju"):
    if rules != "renju" or player != 1:
        return None
    row, col = divmod(action, len(board))
    if not _inside(board, (row, col)) or board[row][col]:
        return "occupied_or_outside"
    # Fewer than four black stones cannot make two threes.
    if sum(cell == 1 for line in board for cell in line) < 4:
        return None
    return _forbidden_after(_placed(board, (row, col)), (row, col))


def legal_actions(board, player=1, rules="freestyle"):
    size = len(board)
    return [
        r * size + c
        for r in range(size)
        for c in range(size)
        if board[r][c] == 0 and not forbidden_reason(board, r * size + c, player, rules)
    ]


def relative_legal_actions(state, rules=None):
    rules = rule_mode(rules)
    stones = sum(cell != 0 for row in state for cell in row)
    color = 1 if stones % 2 == 0 else -1
    absolute = tuple(tuple(cell * color for cell in row) for row in state)
    return legal_actions(absolute, color, rules)
