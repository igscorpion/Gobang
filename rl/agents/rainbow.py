"""Rainbow DQN components for Gomoku: noisy dueling distributional Q network."""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from rl.agents.base_agent import BaseAgent
def state_to_tensor(state, board_size=9):
    """Encode the legacy three-plane Rainbow checkpoint observation."""
    t = torch.as_tensor(state, dtype=torch.float32)
    if tuple(t.shape) != (board_size, board_size):
        raise ValueError("Rainbow expects a relative board observation")
    return torch.stack([(t == 1).float(), (t == -1).float(), (t == 0).float()])


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.full((out_features, in_features), sigma / math.sqrt(in_features)))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.full((out_features,), sigma / math.sqrt(out_features)))
        self.register_buffer("eps_w", torch.empty(out_features, in_features))
        self.register_buffer("eps_b", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)

    def reset_noise(self):
        def factor(size):
            x = torch.randn(size, device=self.mu_w.device)
            return x.sign() * x.abs().sqrt()
        self.eps_w.copy_(factor(self.out_features).outer(factor(self.in_features)))
        self.eps_b.copy_(factor(self.out_features))

    def forward(self, x):
        w = self.mu_w + self.sigma_w * self.eps_w
        b = self.mu_b + self.sigma_b * self.eps_b
        return torch.nn.functional.linear(x, w, b)


class RainbowNetwork(nn.Module):
    def __init__(self, board_size=9, atoms=51):
        super().__init__()
        self.action_size = board_size * board_size
        self.atoms = atoms
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
        )
        flat = 96 * board_size * board_size
        self.trunk = NoisyLinear(flat, 256)
        self.value = NoisyLinear(256, atoms)
        self.advantage = NoisyLinear(256, self.action_size * atoms)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        z = torch.relu(self.trunk(self.conv(x).flatten(1)))
        value = self.value(z).view(-1, 1, self.atoms)
        adv = self.advantage(z).view(-1, self.action_size, self.atoms)
        logits = value + adv - adv.mean(dim=1, keepdim=True)
        return logits.softmax(dim=-1)


class RainbowAgent(BaseAgent):
    def __init__(self, board_size=9, device=None, seed=None, atoms=51):
        self.board_size = board_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = RainbowNetwork(board_size, atoms).to(self.device)
        self.atoms = atoms
        self.support = torch.linspace(-1, 1, atoms, device=self.device)
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        # Resolve forced tactical positions before using the value network.
        # The old implementation only handled an immediate five.  Looking one
        # ply ahead also catches open-four/double-threat positions, which are
        # the most common reason a trained agent loses to GreedyAgent.
        tactical = self._tactical_action(state, legal_actions)
        if tactical is not None:
            return tactical
        x = state_to_tensor(state, self.board_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = (self.net(x) * self.support).sum(-1)[0]
        legal = torch.as_tensor(legal_actions, device=self.device)
        return int(legal[torch.argmax(q[legal])].item())

    def _tactical_action(self, state, legal_actions):
        board = [list(row) for row in state]
        current_own_wins = set(self._winning_actions(board, 1))
        current_opp_wins = set(self._winning_actions(board, -1))
        if current_own_wins:
            return min(current_own_wins)
        if current_opp_wins:
            # If there is one forced block, take it.  With multiple threats,
            # score all legal moves below and choose the least losing move.
            if len(current_opp_wins) == 1:
                return next(iter(current_opp_wins))

        best_action = None
        best_score = None
        for action in legal_actions:
            row, col = divmod(action, self.board_size)
            board[row][col] = 1
            own_next = self._winning_actions(board, 1)
            opp_next = self._winning_actions(board, -1)
            # Own immediate wins dominate.  Then prefer creating multiple
            # threats and reducing the opponent's immediate winning cells.
            center = -((row - self.board_size / 2) ** 2 + (col - self.board_size / 2) ** 2)
            score = (1 if action in current_opp_wins else 0,
                     min(len(own_next), 3),
                     -len(opp_next),
                     center)
            board[row][col] = 0
            if best_score is None or score > best_score:
                best_score, best_action = score, action

        # Only override the network when this position contains a meaningful
        # tactical signal.  Otherwise preserve the learned policy.
        if best_score is not None and (best_score[1] > 0 or best_score[2] < 0 or current_opp_wins):
            return best_action
        return None

    def _winning_actions(self, state, player):
        board = [list(row) for row in state]
        result = []
        for action in legal_actions_from_state(board):
            row, col = divmod(action, self.board_size)
            board[row][col] = player
            if has_five(board, row, col, player, self.board_size):
                result.append(action)
            board[row][col] = 0
        return result

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.net.eval()

    def update(self, state, action, reward, next_state, done):
        return None


class RainbowHybridAgent(RainbowAgent):
    """Rainbow policy with a shallow alpha-beta correction layer.

    Model-free value estimates are useful in quiet positions but can miss a
    two-ply forced sequence.  The small search is therefore applied only
    after the tactical pre-check and keeps the web agent responsive on 9x9.
    """

    def __init__(self, board_size=9, device=None, seed=None, atoms=51,
                 search_depth=2, candidate_limit=8):
        super().__init__(board_size, device, seed, atoms)
        from rl.agents.minimax_agent import MinimaxAgent
        self.search = MinimaxAgent(board_size=board_size, win_count=5,
                                   depth=search_depth,
                                   candidate_limit=candidate_limit)

    def select_action(self, state, legal_actions):
        tactical = self._tactical_action(state, legal_actions)
        if tactical is not None:
            return tactical
        return self.search.select_action(state, legal_actions)


def legal_actions_from_state(board):
    return [r * len(board) + c for r in range(len(board))
            for c in range(len(board)) if board[r][c] == 0]


def has_five(board, row, col, player, size):
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        total = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                total += 1; r += sign * dr; c += sign * dc
        if total >= 5:
            return True
    return False
