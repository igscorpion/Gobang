"""Improved Double-Dueling DQN agent for Gomoku."""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from rl.agents.base_agent import BaseAgent


class QNetwork(nn.Module):
    """Dueling CNN: three relative-board planes -> one Q value per cell."""

    def __init__(self, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
        )
        flat = 96 * board_size * board_size
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(flat, 256), nn.ReLU())
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, self.action_size)

    def forward(self, x):
        z = self.trunk(self.conv(x))
        advantage = self.advantage(z)
        return self.value(z) + advantage - advantage.mean(dim=1, keepdim=True)


def state_to_tensor(state, board_size: int = 9):
    t = torch.as_tensor(state, dtype=torch.long)
    return torch.stack([(t == 1).float(), (t == -1).float(), (t == 0).float()])


class DQNAgent(BaseAgent):
    def __init__(self, board_size: int = 9, epsilon: float = 0.1,
                 device: str | None = None, seed: int | None = None):
        self.board_size = board_size
        self.epsilon = epsilon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNetwork(board_size).to(self.device)
        self.rng = random.Random(seed)

    def _q_values(self, state):
        x = state_to_tensor(state, self.board_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.net(x).squeeze(0).cpu()

    def select_action(self, state, legal_actions):
        if not legal_actions:
            raise ValueError("没有合法动作可选择")
        legal = list(legal_actions)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        q = self._q_values(state)[torch.tensor(legal, dtype=torch.long)]
        return legal[int(torch.argmax(q).item())]

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.net.to(self.device)

    def update(self, state, action, reward, next_state, done):
        return None
