"""Masked PPO policy/value agent for Gomoku."""
from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.agents.base_agent import BaseAgent


class PPONetwork(nn.Module):
    def __init__(self, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.body = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(64, 8, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(8 * self.action_size, self.action_size),
        )
        self.value = nn.Sequential(
            nn.Conv2d(64, 1, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(self.action_size, 128), nn.ReLU(), nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


def state_to_tensor(state, board_size=9):
    board, last_move, is_black = state
    cells = torch.as_tensor(board, dtype=torch.long)
    own = (cells == 1).float()
    opp = (cells == -1).float()
    last = torch.zeros((board_size, board_size), dtype=torch.float32)
    if last_move is not None:
        last[last_move // board_size, last_move % board_size] = 1.0
    first = torch.full((board_size, board_size), float(is_black))
    empty = (cells == 0).float()
    return torch.stack((own, opp, last, first, empty))


def masked_distribution(logits, legal_actions):
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[..., torch.as_tensor(legal_actions, dtype=torch.long, device=logits.device)] = True
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


class PPOAgent(BaseAgent):
    needs_last_move = True

    def __init__(self, board_size=9, device=None, seed=None):
        self.board_size = board_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PPONetwork(board_size).to(self.device)
        self.rng = random.Random(seed)

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        board, last_move = state
        is_black = sum(cell != 0 for row in board for cell in row) % 2 == 0
        x = state_to_tensor((board, last_move, is_black), self.board_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(x)
            dist = masked_distribution(logits[0], legal_actions)
            return int(torch.argmax(dist.probs).item())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.net.eval()

    def update(self, state, action, reward, next_state, done):
        return None
