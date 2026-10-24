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
        num_playouts=int(os.environ.get("GOMOKU_AI_PURE_PLAYOUTS", "1000")))


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


def _make_dqn_improved():
    """Load the Double-Dueling DQN checkpoint trained by train_dqn."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("dqn_improved 需要 torch")
    from rl.agents.dqn import DQNAgent
    agent = DQNAgent(epsilon=0.0)
    model = MODELS_DIR / "dqn_improved.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_dqn_league():
    """Load DQN trained with alternating colors and a mixed opponent league."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("dqn_league 需要 torch")
    from rl.agents.dqn import DQNAgent
    agent = DQNAgent(epsilon=0.0)
    model = MODELS_DIR / "dqn_league.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_dqn_vs_alphazero():
    """Load the experimental DQN checkpoint trained against AlphaZero."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("dqn_vs_alphazero 需要 torch")
    from rl.agents.dqn import DQNAgent
    agent = DQNAgent(epsilon=0.0)
    model = MODELS_DIR / "dqn_league_vs_alphazero.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_ppo():
    """Load the masked PPO checkpoint."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("ppo 需要 torch")
    from rl.agents.ppo import PPOAgent
    agent = PPOAgent()
    model = MODELS_DIR / "ppo.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_rainbow():
    """Load the Rainbow DQN checkpoint."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("rainbow 需要 torch")
    from rl.agents.rainbow import RainbowAgent
    agent = RainbowAgent()
    model = MODELS_DIR / "rainbow.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_rainbow_v2():
    """Load Rainbow DQN with tactical rewards and expert imitation."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("rainbow_v2 需要 torch")
    from rl.agents.rainbow import RainbowAgent
    agent = RainbowAgent()
    model = MODELS_DIR / "rainbow_v2.pt"
    if model.exists():
        agent.load(str(model))
    return agent


def _make_rainbow_hybrid():
    """Rainbow checkpoint plus shallow alpha-beta correction search."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ValueError("rainbow_hybrid 需要 torch")
    from rl.agents.rainbow import RainbowHybridAgent
    agent = RainbowHybridAgent(
        search_depth=int(os.environ.get("GOMOKU_RAINBOW_SEARCH_DEPTH", "2")),
        candidate_limit=int(os.environ.get("GOMOKU_RAINBOW_SEARCH_CANDIDATES", "8")),
    )
    model = MODELS_DIR / "rainbow_v2.pt"
    if model.exists():
        agent.load(model)
    return agent


def _make_alphazero():
    """加载训练好的 AlphaZero 权重；torch 未安装时给出明确报错。"""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise ValueError("alphazero 需要 torch，请先执行 uv add torch")
    from rl.agents.alphazero import AlphaZeroAgent  # 延迟导入

    # 搜索次数（CPU 上约 1~2s/步）；温度 >0 时仅开局抽样，之后仍走最优手
    agent = AlphaZeroAgent(num_sims=400,
                           temperature=ALPHAZERO_TEMPERATURE,
                           opening_moves=ALPHAZERO_OPENING_MOVES,
                           candidate_ratio=ALPHAZERO_CANDIDATE_RATIO)
    model = MODELS_DIR / "alphazero.pt"
    if model.exists():
        try:
            agent.load(str(model))
        except Exception:
            pass  # 检查点与网络结构不兼容时，退回未训练权重
    return agent


# 名称 -> 工厂函数；后续新增算法在此加一行即可
REGISTRY = {
    "random": _make_random,
    "greedy": _make_greedy,
    "minimax": _make_minimax,
    "mcts_pure": _make_mcts_pure,
    "q_learning": _make_q_learning,
    "dqn": _make_dqn,
    "dqn_improved": _make_dqn_improved,
    "dqn_league": _make_dqn_league,
    "dqn_vs_alphazero": _make_dqn_vs_alphazero,
    "ppo": _make_ppo,
    "rainbow": _make_rainbow,
    "rainbow_v2": _make_rainbow_v2,
    "rainbow_hybrid": _make_rainbow_hybrid,
    "alphazero": _make_alphazero,
}


def available_agents():
    """返回可用的 AI 名称列表。"""
    return list(REGISTRY)


def create_agent(name):
    """按名称实例化一个 AI 对手。"""
    if name not in REGISTRY:
        raise ValueError(f"未知 AI '{name}'，可选: {available_agents()}")
    return REGISTRY[name]()
