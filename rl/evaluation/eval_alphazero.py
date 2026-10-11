"""AlphaZero 检查点评估：对 greedy 分执黑 / 执白统计胜 / 和 / 负，并给出自对弈手数分布。

决策是确定性的（温度为 0），但 GPU 浮点计算存在非确定性：接近的走法被打破平局的方式可能改变，
整局棋随之走向不同。因此同一检查点重复测量仍有波动，10~20 局不足以判定 0.1 量级的差异。

用法（项目根目录，需装有 torch 的解释器）：
    py -3.11 -m rl.evaluation.eval_alphazero [games] [num_sims] [pure_playouts]
    games         ：每个颜色的对局数，默认 20
    num_sims      ：每步 MCTS 模拟次数，默认 400（与对战、训练保持一致）
    pure_playouts ：大于 0 时额外对纯 MCTS 打同样局数，默认 0（不测）
"""

from __future__ import annotations

import json
import os
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
    pure_playouts = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # >0 时加测纯 MCTS 基准

    total = "未知"
    if META_PATH.exists():
        total = json.loads(META_PATH.read_text(encoding="utf-8")).get("total_games", "未知")
    print(f"累计 {total} 局 | 每色 {games} 局 | 模拟 {num_sims} 次"
          + (f" | 纯 MCTS {pure_playouts} 次" if pure_playouts else ""), flush=True)

    agent = AlphaZeroAgent(board_size=9, num_sims=num_sims)
    agent.load(str(MODEL_PATH))
    env = GomokuEnv()

    for color, name in ((BLACK, "执黑"), (-BLACK, "执白")):
        wr, dr, lr = evaluate(agent, env, "greedy", games=games,
                              agent_color=color)
        print(f"vs greedy {name}: 胜 {wr:.2f} / 和 {dr:.2f} / 负 {lr:.2f}"
              f"（{round(wr * games)} 胜 / {round(dr * games)} 和 / "
              f"{round(lr * games)} 负）", flush=True)

    if pure_playouts > 0:
        # 纯 MCTS 基准：模拟次数逐级抬高，用于观察「网络+搜索」相对纯搜索的优势
        os.environ["GOMOKU_AI_PURE_PLAYOUTS"] = str(pure_playouts)
        for color, name in ((BLACK, "执黑"), (-BLACK, "执白")):
            wr, dr, lr = evaluate(agent, env, "mcts_pure", games=games,
                                  agent_color=color)
            print(f"vs 纯MCTS({pure_playouts}) {name}: 胜 {wr:.2f} / 和 {dr:.2f} / "
                  f"负 {lr:.2f}", flush=True)

    lengths = []
    for _ in range(SELFPLAY_GAMES):
        data, _winner = selfplay_game(agent.net, agent.device, 9, num_sims, 1.4, 1.0)
        lengths.append(len(data) // 8)
    short = sum(1 for n in lengths if n <= 11)
    print(f"自对弈 {lengths} 均长 {sum(lengths) / len(lengths):.1f}，"
          f"短局(≤11手) {short}/{len(lengths)}", flush=True)


if __name__ == "__main__":
    main()
