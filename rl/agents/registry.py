"""Agent 注册表：按名称创建 AI 对手。

新增算法时，只需在这里加一个「名称 -> 工厂函数」条目，
Web 端即可按名称切换，无需改动 backend / frontend 的切换逻辑。
"""

from __future__ import annotations

import os
from pathlib import Path

from rl.agents.greedy_agent import GreedyAgent
from rl.agents.minimax_agent import MinimaxAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.q_learning import QLearningAgent

# 项目根目录 = rl/agents 的上上级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "rl" / "models"

# AlphaZero 对战时的抽样温度：0 = 每局完全确定；>0 = 前若干手按概率抽样，
# 避免同一局面永远走同一步。抽样只在访问次数达到最高次数 CANDIDATE 倍的走法中进行，
# 以免把搜索已判定的强制手（如必须堵活三）换掉。
# 可用环境变量 GOMOKU_AI_TEMP / GOMOKU_AI_OPENING / GOMOKU_AI_CANDIDATE 覆盖。
ALPHAZERO_TEMPERATURE = float(os.environ.get("GOMOKU_AI_TEMP", "1.0"))
ALPHAZERO_OPENING_MOVES = int(os.environ.get("GOMOKU_AI_OPENING", "6"))
ALPHAZERO_CANDIDATE_RATIO = float(os.environ.get("GOMOKU_AI_CANDIDATE", "0.1"))


def _make_random():
    """随机落子，作为基准。"""
    return RandomAgent()


def _make_greedy():
    """贪心启发式，会进攻与防守，作为更强的基准 / 训练对手。"""
    return GreedyAgent()


def _make_minimax():
    """Negamax + Alpha-Beta 搜索；深度与分支上限可由环境变量配置。"""
    return MinimaxAgent(
        depth=int(os.environ.get("GOMOKU_MINIMAX_DEPTH", "2")),
        candidate_limit=int(os.environ.get("GOMOKU_MINIMAX_CANDIDATES", "8")),
    )


def _make_mcts_pure():
    """纯 MCTS（UCT + 随机模拟），不依赖训练产物，作为「无神经网络的搜索基线」。

    延迟导入：mcts_pure 复用 alphazero 里的棋盘工具函数，而后者会导入 torch。
    模拟次数由 GOMOKU_AI_PURE_PLAYOUTS 控制（默认 1000），便于逐级抬高基准强度。
    """
    from rl.agents.mcts_pure import PureMCTSAgent  # 延迟导入

    return PureMCTSAgent(
        num_playouts=int(os.environ.get("GOMOKU_AI_PURE_PLAYOUTS", "1000"))
    )


def _make_q_learning():
    """加载训练好的线性 Q-Learning 权重；无权重时用零权重。"""
    agent = QLearningAgent(epsilon=0.0)
    model = MODELS_DIR / "q_learning.json"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_dqn():
    """加载训练好的 DQN 权重；torch 未安装时给出明确报错。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise ValueError("dqn 需要 torch，请先在项目根目录执行 uv add torch")
    from rl.agents.dqn import DQNAgent  # 延迟导入，避免拖累无 torch 的启动

    agent = DQNAgent(epsilon=0.0)
    model = MODELS_DIR / "dqn.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_alphazero():
    """加载训练好的 AlphaZero 权重；torch 未安装时给出明确报错。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise ValueError("alphazero 需要 torch，请先执行 uv add torch")
    from rl.agents.alphazero import AlphaZeroAgent  # 延迟导入

    # 搜索次数（CPU 上约 1~2s/步）；温度 >0 时仅开局抽样，之后仍走最优手
    agent = AlphaZeroAgent(
        num_sims=400,
        device=os.environ.get("GOMOKU_AI_DEVICE", "cpu"),
        temperature=ALPHAZERO_TEMPERATURE,
        opening_moves=ALPHAZERO_OPENING_MOVES,
        candidate_ratio=ALPHAZERO_CANDIDATE_RATIO,
    )
    model = MODELS_DIR / "alphazero.pt"
    if model.exists():
        agent.load(str(model))
    return agent


# 名称 -> 工厂函数；后续新增算法在此加一行即可
REGISTRY = {
    "random": _make_random,
    "greedy": _make_greedy,
    "minimax": _make_minimax,
    "mcts_pure": _make_mcts_pure,
    "q_learning": _make_q_learning,
    "dqn": _make_dqn,
    "alphazero": _make_alphazero,
}


def available_agents():
    """返回可用的 AI 名称列表。"""
    local = _local_models()
    names = list(REGISTRY) + list(local)
    if local.get("a2c4", {}).get("algorithm") == "a2c" and "a2c4_tactical" not in names:
        names.append("a2c4_tactical")
    return names


def _make_research_a2c(item, path):
    """Published frozen research inference; old model factories stay separate."""
    import hashlib
    from dataclasses import asdict
    import torch
    from rl.agents.a2c_research import ModelSpec, ResearchA2CAgent
    from rl.agents.a2c_guarded import GuardedA2CAgent
    from rl.agents.tactics import TacticalConfig
    from rl.environment.rules import rule_mode

    checksum = item.get("checkpoint_sha256")
    if not isinstance(checksum, str) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Research checkpoint SHA256 does not match published metadata")
    if item.get("device", "cpu") != "cpu" or item.get("wrapper") != "guarded":
        raise ValueError("Published research recipe requires CPU and guarded inference")
    config = TacticalConfig()
    if item.get("tactical_config") != asdict(config):
        raise ValueError("Research tactical configuration differs from the published recipe")
    metadata = torch.load(path, map_location="cpu", weights_only=True)
    if metadata.get("format") != "a2c_research_v1" or "spec" not in metadata:
        raise ValueError("Research model requires explicit versioned ModelSpec metadata")
    model_spec = ModelSpec.from_dict(metadata["spec"])
    if ModelSpec.from_dict(item["model_spec"]) != model_spec:
        raise ValueError("Research model specification differs from published metadata")
    policy = ResearchA2CAgent(model_spec, device="cpu", rules=metadata["rules"], training=False)
    policy.load(path)
    # Rule-aware transfer is explicit: load under training rules first, then
    # use the current game's rules for BOTH feature planes and the guard.
    requested_rules = rule_mode()
    policy.rules = requested_rules
    return GuardedA2CAgent(policy, rules=requested_rules, tactical_config=config)


def create_agent(name):
    """按名称实例化一个 AI 对手。"""
    local = _local_models()
    tactical_a2c = name == "a2c4_tactical"
    source = "a2c4" if tactical_a2c else name
    if tactical_a2c and local.get(source, {}).get("algorithm") != "a2c":
        raise ValueError("a2c4_tactical requires a validation-selected a2c4 checkpoint")
    if source in local:
        item = local[source]
        path = (PROJECT_ROOT / item["checkpoint"]).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise ValueError("Local model is missing or outside project")
        if item["algorithm"] == "alphazero":
            from rl.agents.alphazero import AlphaZeroAgent

            agent = AlphaZeroAgent(
                num_sims=400,
                temperature=0.0,
                device=os.environ.get("GOMOKU_AI_DEVICE", "cpu"),
            )
        elif item["algorithm"] == "a2c_research":
            return _make_research_a2c(item, path)
        elif item["algorithm"] == "a2c":
            if tactical_a2c:
                from rl.agents.a2c_tactical import TacticalA2CAgent
                from rl.agents.tactics import TacticalConfig

                config = TacticalConfig(
                    forcing_depth=int(os.environ.get("GOMOKU_A2C_TACTIC_DEPTH", "5")),
                    max_nodes=int(os.environ.get("GOMOKU_A2C_TACTIC_NODES", "1500")),
                    candidate_limit=int(os.environ.get("GOMOKU_A2C_TACTIC_CANDIDATES", "12")),
                )
                agent = TacticalA2CAgent(
                    device=os.environ.get("GOMOKU_AI_DEVICE", "cpu"),
                    tactical_config=config,
                )
            else:
                from rl.agents.a2c import A2CAgent

                agent = A2CAgent(device=os.environ.get("GOMOKU_AI_DEVICE", "cpu"))
        else:
            from rl.agents.dqn import DQNAgent

            agent = DQNAgent(epsilon=0.0)
        agent.load(path)
        return agent
    if name not in REGISTRY:
        raise ValueError(f"未知 AI '{name}'，可选: {available_agents()}")
    return REGISTRY[name]()


def _local_models():
    import json

    models = {}
    for name in ("models.json", "a2c_models.json"):
        path = PROJECT_ROOT / "runs/production" / name
        if path.exists():
            models.update(json.loads(path.read_text(encoding="utf-8")))
    path = PROJECT_ROOT / "runs/production/a2c_research_models.json"
    if path.exists():
        published = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(published, dict):
            raise ValueError("Research model catalog must be an object")
        for name, item in published.items():
            if not isinstance(item, dict) or item.get("enabled") is not True:
                continue
            if name in models or name in REGISTRY or name == "a2c4_tactical":
                raise ValueError("Research model name collides with an existing option")
            if item.get("algorithm") != "a2c_research":
                raise ValueError("Research catalog contains an unsupported algorithm")
            models[name] = item
    return models
