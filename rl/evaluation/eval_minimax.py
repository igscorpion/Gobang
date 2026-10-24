"""Minimax 对照评估：按执黑/执白统计胜和负及搜索成本。

用法（项目根目录）：
    uv run python -m rl.evaluation.eval_minimax greedy --games 10 --depth 2
    uv run python -m rl.evaluation.eval_minimax alphazero --games 10 --depth 2

--games 是每种棋色各自的对局数。AlphaZero 对手需要已安装 torch，且会尝试
加载 rl/models/alphazero.pt。所有结果均报告 Minimax 一方的胜/和/负。
"""

from __future__ import annotations

import argparse

from rl.agents.minimax_agent import MinimaxAgent
from rl.agents.registry import available_agents, create_agent
from rl.environment.gomoku import BLACK, GomokuEnv, encode_action


def _play_color(agent, opponent, env, agent_color, games):
    wins = draws = 0
    decisions = nodes = 0
    seconds = 0.0

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_actions()
            if env.current_player == agent_color:
                action = agent.select_action(state, legal)
                decisions += 1
                nodes += agent.last_search_nodes
                seconds += agent.last_search_seconds
            else:
                opponent_state = state
                if opponent.needs_last_move:
                    last = None
                    if env.last_move is not None:
                        last = encode_action(env.last_move[0], env.last_move[1],
                                             env.board_size)
                    opponent_state = (state, last)
                action = opponent.select_action(opponent_state, legal)
            state, _reward, done = env.step(action)

        if env.winner == agent_color:
            wins += 1
        elif env.winner == env.EMPTY:
            draws += 1

    losses = games - wins - draws
    avg_nodes = nodes / decisions if decisions else 0.0
    avg_seconds = seconds / decisions if decisions else 0.0
    print(
        f"Minimax 执{'黑' if agent_color == BLACK else '白'}："
        f"胜 {wins}/{games} ({wins / games:.3f}) / "
        f"和 {draws}/{games} ({draws / games:.3f}) / "
        f"负 {losses}/{games} ({losses / games:.3f}) | "
        f"搜索深度 {agent.depth} | 决策 {decisions} 次 | "
        f"平均节点 {avg_nodes:.1f}/步 | 平均耗时 {avg_seconds:.4f} 秒/步"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opponent", choices=[name for name in available_agents()
                                               if name != "minimax"])
    parser.add_argument("--games", type=int, default=10,
                        help="每种棋色的对局数（默认：10）")
    parser.add_argument("--depth", type=int, default=2,
                        help="Minimax 搜索半回合深度（默认：2）")
    parser.add_argument("--candidate-limit", type=int, default=8,
                        help="每层普通候选动作上限（默认：8）")
    parser.add_argument("--neighborhood", type=int, default=2,
                        help="候选点与棋子的最大 Chebyshev 距离（默认：2）")
    parser.add_argument("--no-alpha-beta", action="store_true",
                        help="关闭 Alpha-Beta，用于搜索结果对照")
    parser.add_argument("--opponent-sims", type=int,
                        help="AlphaZero 对手每步模拟数（可选，用于控制评估耗时）")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games 必须至少为 1")

    agent = MinimaxAgent(
        depth=args.depth,
        candidate_limit=args.candidate_limit,
        neighborhood=args.neighborhood,
        use_alpha_beta=not args.no_alpha_beta,
    )
    opponent = create_agent(args.opponent)
    if args.opponent_sims is not None:
        if args.opponent_sims < 1:
            parser.error("--opponent-sims 必须至少为 1")
        if not hasattr(opponent, "mcts"):
            parser.error("--opponent-sims 仅适用于 AlphaZero 对手")
        opponent.num_sims = args.opponent_sims
        opponent.mcts.num_sims = args.opponent_sims
    env = GomokuEnv()
    opponent_sims = (f"对手模拟={args.opponent_sims} | "
                     if args.opponent_sims is not None else "")
    print(
        f"对手={args.opponent} | 每色 {args.games} 局 | 深度={args.depth} | "
        f"候选上限={args.candidate_limit} | 邻域={args.neighborhood} | "
        f"{opponent_sims}Alpha-Beta={'关闭' if args.no_alpha_beta else '开启'}"
    )
    for color in (BLACK, -BLACK):
        _play_color(agent, opponent, env, color, args.games)


if __name__ == "__main__":
    main()
