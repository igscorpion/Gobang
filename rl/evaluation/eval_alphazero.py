"""AlphaZero 检查点评估：对 greedy 分执黑 / 执白统计胜率，并给出自对弈手数分布。

固定使用确定性决策（温度为 0），保证同一检查点的结果可复现。

用法（项目根目录，需装有 torch 的解释器）：
    py -3.11 -m rl.evaluation.eval_alphazero [games] [num_sims]
    games    ：每个颜色的对局数，默认 20
    num_sims ：每步 MCTS 模拟次数，默认 400（与对战、训练保持一致）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.agents.alphazero import AlphaZeroAgent
from rl.environment.gomoku import BLACK, GomokuEnv
from rl.training.train_alphazero import evaluate, selfplay_game

MODEL_PATH = ROOT / "rl" / "models" / "alphazero.pt"
META_PATH = ROOT / "rl" / "models" / "alphazero_meta.json"
SELFPLAY_GAMES = 10  # 自对弈手数分布的采样局数


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    total = "未知"
    if META_PATH.exists():
        total = json.loads(META_PATH.read_text(encoding="utf-8")).get("total_games", "未知")
    print(f"累计 {total} 局 | 每色 {games} 局 | 模拟 {num_sims} 次", flush=True)

    agent = AlphaZeroAgent(board_size=9, num_sims=num_sims)
    agent.load(str(MODEL_PATH))
    env = GomokuEnv()

    for color, name in ((BLACK, "执黑"), (-BLACK, "执白")):
        wr = evaluate(agent, env, "greedy", games=games, agent_color=color)
        print(f"vs greedy {name}: {wr:.2f}（{round(wr * games)}/{games}）", flush=True)

    lengths = []
    for _ in range(SELFPLAY_GAMES):
        data, _winner = selfplay_game(agent.net, agent.device, 9, num_sims, 1.4, 1.0)
        lengths.append(len(data) // 8)
    short = sum(1 for n in lengths if n <= 11)
    print(f"自对弈 {lengths} 均长 {sum(lengths) / len(lengths):.1f}，"
          f"短局(≤11手) {short}/{len(lengths)}", flush=True)


if __name__ == "__main__":
    main()
