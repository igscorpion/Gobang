"""Four-channel DQN; legacy three-channel models can be evaluated explicitly."""

import random
import torch
from torch import nn
from rl.agents.base_agent import BaseAgent
from rl.environment.state import unpack, state_to_tensor4


class QNetwork(nn.Module):
    def __init__(self, board_size=9, input_channels=4):
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * board_size**2, 256), nn.ReLU(), nn.Linear(256, board_size**2)
        )

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


def state_to_tensor(state, board_size=9, input_channels=4):
    if input_channels == 4:
        return state_to_tensor4(state, board_size)
    board, _ = unpack(state)
    t = torch.as_tensor(board)
    return torch.stack([(t == 1).float(), (t == -1).float(), (t == 0).float()])


class DQNAgent(BaseAgent):
    needs_last_move = True

    def __init__(
        self, board_size=9, epsilon=0.1, device=None, seed=None, input_channels=4
    ):
        self.board_size = board_size
        self.epsilon = epsilon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNetwork(board_size, input_channels).to(self.device)
        self.rng = random.Random(seed)

    def _q_values(self, state):
        x = (
            state_to_tensor(state, self.board_size, self.net.input_channels)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            return self.net(x).squeeze(0).cpu()

    def select_action(self, state, legal_actions):
        legal = list(legal_actions)
        if not legal:
            raise ValueError("No legal actions")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        q = self._q_values(state)
        return legal[int(q[legal].argmax())]

    def save(self, path):
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_channels": self.net.input_channels,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        weights = checkpoint.get("state_dict", checkpoint)
        self.net = QNetwork(self.board_size, weights["conv.0.weight"].shape[1]).to(
            self.device
        )
        self.net.load_state_dict(weights)

    def update(self, *args):
        return None
