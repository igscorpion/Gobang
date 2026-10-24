"""Four-channel actor-critic policy used by the synchronous A2C control."""

from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

from rl.agents.base_agent import BaseAgent
from rl.environment.state import state_to_tensor4
from rl.training.a2c_math import masked_distribution


class ActorCriticNetwork(nn.Module):
    """DQN-sized shared encoder with independent policy and value heads."""

    def __init__(self, board_size=9, channels=(32, 64, 64), hidden_units=256):
        super().__init__()
        if board_size < 1 or hidden_units < 1:
            raise ValueError("board_size and hidden_units must be positive")
        if len(channels) != 3 or any(width < 1 for width in channels):
            raise ValueError("channels must contain three positive widths")
        self.board_size = int(board_size)
        self.channels = tuple(int(width) for width in channels)
        self.hidden_units = int(hidden_units)
        self.input_channels = 4
        layers = []
        previous_width = self.input_channels
        for width in self.channels:
            layers.extend([nn.Conv2d(previous_width, width, 3, padding=1), nn.ReLU()])
            previous_width = width
        self.conv = nn.Sequential(*layers)
        self.shared = nn.Sequential(
            nn.Linear(self.channels[-1] * self.board_size**2, self.hidden_units),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(self.hidden_units, self.board_size**2)
        self.value_head = nn.Linear(self.hidden_units, 1)

    def forward(self, x):
        shared = self.shared(self.conv(x).flatten(1))
        return self.policy_head(shared), self.value_head(shared).squeeze(-1)


class A2CAgent(BaseAgent):
    needs_last_move = True

    def __init__(
        self, device=None, seed=None, training=False, epsilon=0.0, board_size=9
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.board_size = board_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.training = bool(training)
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)
        self.net = ActorCriticNetwork(board_size=board_size).to(self.device)
        self.net.train(self.training)

    def select_action(self, state, legal_actions):
        legal = list(legal_actions)
        if not legal:
            raise ValueError("No legal actions")
        if any(action < 0 or action >= self.board_size**2 for action in legal):
            raise ValueError("Legal action is outside the board")
        # Historical opponents use this epsilon; the learning policy uses zero.
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        x = state_to_tensor4(state, self.board_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, _ = self.net(x)
            legal_mask = torch.zeros_like(logits, dtype=torch.bool)
            legal_mask[0, legal] = True
            distribution = masked_distribution(logits, legal_mask)
            if self.training:
                return int(distribution.sample().item())
            return int(distribution.logits.argmax(dim=-1).item())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            torch.save(
                {
                    "state_dict": self.net.state_dict(),
                    "board_size": self.board_size,
                    "input_channels": self.net.input_channels,
                    "channels": self.net.channels,
                    "hidden_units": self.net.hidden_units,
                },
                temporary,
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if "state_dict" not in checkpoint:
            raise ValueError("A2C checkpoint is missing network metadata")
        if int(checkpoint.get("board_size", self.board_size)) != self.board_size:
            raise ValueError("A2C checkpoint board size does not match the agent")
        if int(checkpoint.get("input_channels", 4)) != 4:
            raise ValueError("A2C checkpoint must use four input channels")
        network = ActorCriticNetwork(
            board_size=self.board_size,
            channels=tuple(checkpoint.get("channels", (32, 64, 64))),
            hidden_units=int(checkpoint.get("hidden_units", 256)),
        ).to(self.device)
        network.load_state_dict(checkpoint["state_dict"], strict=True)
        network.train(self.training)
        self.net = network

    def update(self, *args):
        # Synchronous rollout updates live in the trainer, never per action.
        return None
