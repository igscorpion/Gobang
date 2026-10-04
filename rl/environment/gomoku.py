"""9x9 五子棋环境。

棋盘内部存储使用绝对颜色：
    0  空
    1  黑（先手）
   -1  白

get_state() 返回相对当前玩家的视角：
    0  空
    1  当前玩家
   -1  对手
"""

from __future__ import annotations

BOARD_SIZE = 9
WIN_COUNT = 5
EMPTY = 0
BLACK = 1
WHITE = -1


def encode_action(row: int, col: int, board_size: int = BOARD_SIZE) -> int:
    return row * board_size + col


def decode_action(action, board_size: int = BOARD_SIZE) -> tuple[int, int]:
    if isinstance(action, (tuple, list)):
        return int(action[0]), int(action[1])
    action = int(action)
    return action // board_size, action % board_size


class GomokuEnv:
    """可独立于 Web 运行的五子棋 Environment。"""

    BOARD_SIZE = BOARD_SIZE
    WIN_COUNT = WIN_COUNT
    EMPTY = EMPTY
    BLACK = BLACK
    WHITE = WHITE

    def __init__(self, board_size: int = BOARD_SIZE, win_count: int = WIN_COUNT):
        self.board_size = board_size
        self.win_count = win_count
        self.board: list[list[int]] = []
        self.current_player = BLACK
        self.done = False
        self.winner: int | None = None
        self.last_move: tuple[int, int] | None = None
        self.reset()

    def reset(self) -> list[list[int]]:
        self.board = [
            [EMPTY for _ in range(self.board_size)]
            for _ in range(self.board_size)
        ]
        self.current_player = BLACK
        self.done = False
        self.winner = None
        self.last_move = None
        return self.get_state()

    def get_state(self) -> list[list[int]]:
        """相对当前玩家的棋盘：1 是自己，-1 是对手。"""
        return [
            [cell * self.current_player for cell in row]
            for row in self.board
        ]

    def get_board(self) -> list[list[int]]:
        """绝对棋盘：1 黑，-1 白。返回拷贝，避免外部修改内部状态。"""
        return [row[:] for row in self.board]

    def get_legal_actions(self) -> list[int]:
        if self.done:
            return []
        actions: list[int] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == EMPTY:
                    actions.append(encode_action(row, col, self.board_size))
        return actions

    def step(self, action) -> tuple[list[list[int]], float, bool]:
        """执行一步落子。

        返回:
            next_state, reward, done
        reward 相对「刚刚落子的玩家」：胜 +1，和 0，未结束 0。
        """
        if self.done:
            raise ValueError("对局已经结束，请先 reset()")

        row, col = decode_action(action, self.board_size)
        self._assert_legal(row, col)

        self.board[row][col] = self.current_player
        self.last_move = (row, col)

        winner = self.check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            return self.get_state(), 1.0, True

        if self._is_full():
            self.done = True
            self.winner = EMPTY
            return self.get_state(), 0.0, True

        self.current_player *= -1
        return self.get_state(), 0.0, False

    def check_winner(self) -> int | None:
        """返回 1 / -1，无人获胜则返回 None。和棋不是 winner。"""
        size = self.board_size
        need = self.win_count
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))

        for row in range(size):
            for col in range(size):
                player = self.board[row][col]
                if player == EMPTY:
                    continue
                for dr, dc in directions:
                    count = 1
                    r, c = row + dr, col + dc
                    while 0 <= r < size and 0 <= c < size and self.board[r][c] == player:
                        count += 1
                        if count >= need:
                            return player
                        r += dr
                        c += dc
        return None

    def _is_full(self) -> bool:
        return all(cell != EMPTY for row in self.board for cell in row)

    def _assert_legal(self, row: int, col: int) -> None:
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"落子越界: ({row}, {col})")
        if self.board[row][col] != EMPTY:
            raise ValueError(f"该位置已有棋子: ({row}, {col})")


def _run_self_checks() -> None:
    env = GomokuEnv()

    state = env.reset()
    assert len(state) == BOARD_SIZE and len(state[0]) == BOARD_SIZE
    assert all(cell == EMPTY for row in state for cell in row)
    assert env.current_player == BLACK
    assert len(env.get_legal_actions()) == BOARD_SIZE * BOARD_SIZE

    env.step((0, 0))
    try:
        env.step((0, 0))
        raise AssertionError("重复落子应当失败")
    except ValueError:
        pass

    env.reset()
    try:
        env.step((-1, 0))
        raise AssertionError("越界落子应当失败")
    except ValueError:
        pass

    env.reset()
    for col in range(WIN_COUNT):
        env.step((0, col))
        if col < WIN_COUNT - 1:
            env.step((1, col))
    assert env.done and env.winner == BLACK

    env.reset()
    for row in range(WIN_COUNT):
        env.step((row, 0))
        if row < WIN_COUNT - 1:
            env.step((row, 1))
    assert env.done and env.winner == BLACK

    env.reset()
    for i in range(WIN_COUNT):
        env.step((i, i))
        if i < WIN_COUNT - 1:
            env.step((i, i + 1))
    assert env.done and env.winner == BLACK

    full_env = GomokuEnv(win_count=BOARD_SIZE + 1)
    for action in range(BOARD_SIZE * BOARD_SIZE):
        next_state, reward, done = full_env.step(action)
    assert done and full_env.winner == EMPTY and reward == 0.0
    assert next_state is not None

    env.reset()
    env.step((0, 0))
    relative = env.get_state()
    assert relative[0][0] == -1, "轮到白棋时，黑子应显示为对手 (-1)"

    print("GomokuEnv self-check passed.")
    print(f"board_size={BOARD_SIZE}, legal_actions={len(GomokuEnv().get_legal_actions())}")


if __name__ == "__main__":
    _run_self_checks()
