"""Isolated A2C research models; original agents and checkpoints stay unchanged.

The ``flat/base4`` model has exactly the original A2C parameter layout.  Other
architectures/features are explicit ablations, never inferred from weight shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import random

import torch
from torch import nn

from rl.agents.base_agent import BaseAgent
from rl.agents.q_learning import _features
from rl.environment.rules import legal_actions, winning_move
from rl.environment.state import state_to_tensor4, unpack
from rl.training.a2c_math import masked_distribution


@dataclass(frozen=True)
class ModelSpec:
    architecture: str = "flat"
    features: str = "base4"
    board_size: int = 9
    channels: tuple[int, int, int] = (32, 64, 64)
    hidden_units: int = 256
    residual_width: int = 64
    residual_blocks: int = 4

    def __post_init__(self):
        if self.architecture not in ("flat", "residual"):
            raise ValueError("Unknown architecture")
        if self.features not in ("base4", "tactical10"):
            raise ValueError("Unknown observation features")
        if self.board_size != 9:
            raise ValueError("Research protocol is defined for the 9x9 board")
        if len(self.channels) != 3 or min(self.channels) < 1:
            raise ValueError("Three positive convolution widths are required")
        if min(self.hidden_units, self.residual_width, self.residual_blocks) < 1:
            raise ValueError("Network dimensions must be positive")

    @property
    def input_channels(self):
        return 4 if self.features == "base4" else 10

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values["channels"] = tuple(values.get("channels", (32, 64, 64)))
        return cls(**values)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ResearchNetwork(nn.Module):
    def __init__(self, spec=ModelSpec()):
        super().__init__()
        self.spec = spec
        if spec.architecture == "flat":
            layers, previous = [], spec.input_channels
            for width in spec.channels:
                layers.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU()])
                previous = width
            self.conv = nn.Sequential(*layers)
            self.shared = nn.Sequential(nn.Linear(previous * spec.board_size**2,
                                                  spec.hidden_units), nn.ReLU())
            self.policy_head = nn.Linear(spec.hidden_units, spec.board_size**2)
            self.value_head = nn.Linear(spec.hidden_units, 1)
        else:
            width = spec.residual_width
            self.stem = nn.Sequential(nn.Conv2d(spec.input_channels, width, 3, padding=1),
                                      nn.ReLU())
            self.blocks = nn.Sequential(*(ResidualBlock(width)
                                          for _ in range(spec.residual_blocks)))
            self.policy_head = nn.Conv2d(width, 1, 1)
            self.value_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                             nn.Linear(width, width), nn.ReLU(),
                                             nn.Linear(width, 1))

    def forward(self, x):
        if self.spec.architecture == "flat":
            hidden = self.shared(self.conv(x).flatten(1))
            return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)
        hidden = self.blocks(self.stem(x))
        return self.policy_head(hidden).flatten(1), self.value_head(hidden).squeeze(-1)


FEATURE_DESCRIPTIONS = (
    "own_stones", "opponent_stones", "opponent_last_move", "is_black_to_move",
    "own_legal_points", "opponent_legal_points_if_playing_now",
    "own_legal_immediate_wins", "opponent_legal_immediate_wins",
    "own_local_threat_strength", "opponent_local_threat_strength",
)


def encode_state(state, spec=ModelSpec(), rules="freestyle"):
    """Optional planes are deterministic board-derived priors, not future labels.

Threat strength is min(1, open4 + half4/2 + open3/4 + half3/8 + open2/16).
It is a local contiguous-pattern heuristic, not a proof of a forcing sequence.
Opponent legality uses their actual absolute color (important for Renju).
"""
    if spec.features == "base4":
        return state_to_tensor4(state, spec.board_size)
    board, _ = unpack(state)
    _, last_move = unpack(state)
    immutable = tuple(tuple(int(cell) for cell in row) for row in board)
    return _cached_tactical_planes(immutable, last_move, spec, rules).clone()


@lru_cache(maxsize=4096)
def _cached_tactical_planes(board, last_move, spec, rules):
    """CPU cache for the optional feature ablation; base4 pays no cache cost."""
    base = state_to_tensor4((board, last_move), spec.board_size)
    color = 1 if base[3, 0, 0] else -1
    absolute = [[int(cell) * color for cell in row] for row in board]
    extra = torch.zeros(6, spec.board_size, spec.board_size)
    for index, player in enumerate((color, -color)):
        relative = [[int(cell) * player for cell in row] for row in absolute]
        for action in legal_actions(absolute, player, rules):
            row, col = divmod(action, spec.board_size)
            extra[index, row, col] = 1
            placed = [line[:] for line in absolute]
            placed[row][col] = player
            extra[index + 2, row, col] = float(winning_move(
                placed, row, col, player, rules))
            phi = _features(relative, action, spec.board_size, 5)
            extra[index + 4, row, col] = min(1.0, phi['open4'] + phi['half4'] / 2 +
                                             phi['open3'] / 4 + phi['half3'] / 8 +
                                             phi['open2'] / 16)
    return torch.cat((base, extra))


def base_to_state(base):
    """Recover the complete observable state stored in a base4 demonstration."""
    board = (base[0].to(torch.int8) - base[1].to(torch.int8)).tolist()
    locations = base[2].flatten().nonzero().flatten().tolist()
    if len(locations) > 1:
        raise ValueError("More than one last-move point")
    return board, locations[0] if locations else None


class ResearchA2CAgent(BaseAgent):
    needs_last_move = True

    def __init__(self, spec=ModelSpec(), device="cpu", seed=11, rules="freestyle",
                 training=False):
        self.spec, self.device, self.rules = spec, torch.device(device), rules
        self.training, self.epsilon = training, 0.0
        self.rng = random.Random(seed)
        self.net = ResearchNetwork(spec).to(self.device)
        self.net.train(training)

    def encode(self, state):
        return encode_state(state, self.spec, self.rules)

    def policy_value(self, state):
        """Frozen search interface: CPU logits[81] and an unclipped scalar value."""
        with torch.inference_mode():
            logits, values = self.net(self.encode(state).unsqueeze(0).to(self.device))
        return logits[0].detach().cpu(), float(values[0])

    def select_action(self, state, legal_actions):
        legal = list(legal_actions)
        if not legal:
            raise ValueError("No legal action")
        with torch.inference_mode():
            logits, _ = self.net(self.encode(state).unsqueeze(0).to(self.device))
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0, legal] = True
            distribution = masked_distribution(logits, mask)
            return int((distribution.sample() if self.training else
                        distribution.logits.argmax(-1)).item())

    def payload(self, **metadata):
        return {"format": "a2c_research_v1", "spec": asdict(self.spec),
                "model_variant": f"{self.spec.architecture}/{self.spec.features}",
                "input_channels": self.spec.input_channels, "rules": self.rules,
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in self.net.state_dict().items()},
                **metadata}

    def save(self, path, **metadata):
        from rl.training.common import atomic_torch
        atomic_torch(path, self.payload(**metadata))

    def load(self, path, *, allow_rule_transfer=False):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") == "a2c_research_v1":
            if ModelSpec.from_dict(payload["spec"]) != self.spec:
                raise ValueError("Model specification does not match checkpoint")
            if payload.get("input_channels") != self.spec.input_channels or \
                    payload.get("model_variant") != f"{self.spec.architecture}/{self.spec.features}":
                raise ValueError("Model variant metadata does not match its specification")
            if payload["rules"] != self.rules and not allow_rule_transfer:
                raise ValueError("Checkpoint rule mode does not match")
        else:
            # Original checkpoints are accepted only by the unchanged model.
            if self.spec != ModelSpec() or payload.get("input_channels") != 4:
                raise ValueError("Original A2C weights require the exact original model")
            if tuple(payload.get("channels", ())) != self.spec.channels or \
                    payload.get("hidden_units") != self.spec.hidden_units:
                raise ValueError("Original A2C network dimensions do not match")
        self.net.load_state_dict(payload["state_dict"], strict=True)
        self.net.train(self.training)


class ResearchTacticalAgent(BaseAgent):
    """A frozen wrapper for separate hybrid evaluation, never for rollouts."""
    needs_last_move = True

    def __init__(self, policy, tactical_config=None):
        from rl.agents.tactics import TacticalEngine
        if policy.training:
            raise ValueError("Tactical wrapper is inference-only")
        self.policy = policy
        self.tactics = TacticalEngine(tactical_config, policy.rules, policy.spec.board_size)
        self.last_tactical_decision = None

    def select_action(self, state, legal_actions):
        decision = self.tactics.analyze(state, legal_actions)
        self.last_tactical_decision = decision
        if not decision.actions:
            raise ValueError("No legal tactical candidate")
        return (decision.actions[0] if len(decision.actions) == 1 else
                self.policy.select_action(state, decision.actions))
