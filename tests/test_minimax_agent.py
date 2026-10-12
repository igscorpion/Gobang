"""Minimax Agent 的接口与战术/剪枝回归测试。"""

import unittest

from rl.agents.minimax_agent import MinimaxAgent
from rl.agents.registry import create_agent
from rl.environment.gomoku import BOARD_SIZE


def legal_actions(board):
    return [r * BOARD_SIZE + c
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE) if board[r][c] == 0]


class MinimaxAgentTests(unittest.TestCase):
    def test_selects_immediate_winning_move(self):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(4):
            board[4][col] = 1
        agent = MinimaxAgent(depth=1)

        action = agent.select_action(board, legal_actions(board))

        self.assertEqual(action, 4 * BOARD_SIZE + 4)
        self.assertGreater(agent.last_search_nodes, 0)
        self.assertGreaterEqual(agent.last_search_seconds, 0.0)
        self.assertEqual(agent.last_search_depth, 1)

    def test_blocks_opponents_immediate_win(self):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(4):
            board[4][col] = -1
        agent = MinimaxAgent(depth=2, candidate_limit=1)

        action = agent.select_action(board, legal_actions(board))

        self.assertEqual(action, 4 * BOARD_SIZE + 4)

    def test_alpha_beta_matches_plain_minimax(self):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for row, col, value in (
            (4, 4, 1), (4, 5, -1), (5, 4, 1), (3, 4, -1),
        ):
            board[row][col] = value
        actions = legal_actions(board)
        pruned = MinimaxAgent(depth=3, candidate_limit=4, use_alpha_beta=True)
        plain = MinimaxAgent(depth=3, candidate_limit=4, use_alpha_beta=False)

        pruned_action = pruned.select_action(board, actions)
        plain_action = plain.select_action(board, actions)

        self.assertEqual(pruned_action, plain_action)
        self.assertAlmostEqual(pruned.last_search_value, plain.last_search_value)
        self.assertLessEqual(pruned.last_search_nodes, plain.last_search_nodes)

    def test_depth_expands_more_search_nodes(self):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[4][4] = 1
        board[4][5] = -1
        actions = legal_actions(board)
        shallow = MinimaxAgent(depth=1, candidate_limit=3)
        deeper = MinimaxAgent(depth=2, candidate_limit=3)

        shallow.select_action(board, actions)
        deeper.select_action(board, actions)

        self.assertEqual(shallow.last_search_depth, 1)
        self.assertEqual(deeper.last_search_depth, 2)
        self.assertGreater(deeper.last_search_nodes, shallow.last_search_nodes)

    def test_only_returns_supplied_legal_action(self):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[4][4] = 1
        supplied = [4 * BOARD_SIZE + 5, 5 * BOARD_SIZE + 4]
        agent = MinimaxAgent(depth=2)

        self.assertIn(agent.select_action(board, supplied), supplied)

    def test_rejects_invalid_search_configuration(self):
        with self.assertRaises(ValueError):
            MinimaxAgent(depth=0)
        with self.assertRaises(ValueError):
            MinimaxAgent(candidate_limit=0)

    def test_agent_is_available_through_registry(self):
        self.assertIsInstance(create_agent("minimax"), MinimaxAgent)


if __name__ == "__main__":
    unittest.main()
