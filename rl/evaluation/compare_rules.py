"""Matched-seed same-policy rule comparison; not proof of a solved game."""

from pathlib import Path
import argparse, os, random
from rl.environment.gomoku import GomokuEnv
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.alphazero import AlphaZeroAgent
from rl.training.common import choose, append_csv, atomic_json, seed_everything


def run(args):
    seed_everything(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    (out / "games.csv").write_text("", encoding="utf-8")
    for rules in ("freestyle", "renju"):
        os.environ["GOMOKU_RULES"] = rules
        for game in range(args.games):
            rng = random.Random(args.seed + game)
            env = GomokuEnv(rules=rules)
            if args.algorithm == "greedy":
                black = GreedyAgent(seed=args.seed + game)
                white = GreedyAgent(seed=args.seed + game)
            else:
                black = AlphaZeroAgent(
                    num_sims=args.sims,
                    device=args.device,
                    rules=rules,
                    seed=args.seed + game,
                )
                black.load(args.model)
                white = AlphaZeroAgent(
                    num_sims=args.sims,
                    device=args.device,
                    rules=rules,
                    seed=args.seed + game,
                )
                white.net.load_state_dict(black.net.state_dict())
            opening = []
            moves = 0
            while not env.done:
                if moves < args.opening_moves:
                    action = rng.choice(env.get_legal_actions())
                    opening.append(action)
                else:
                    action = choose(black if env.current_player == 1 else white, env)
                env.step(action)
                moves += 1
            row = {
                "rules": rules,
                "game": game,
                "seed": args.seed + game,
                "winner": env.winner,
                "moves": moves,
                "opening": ",".join(map(str, opening)),
                "algorithm": args.algorithm,
                "sims": args.sims,
            }
            rows.append(row)
            append_csv(out / "games.csv", row)
            if (game + 1) % 10 == 0:
                atomic_json(
                    out / "progress.json",
                    {
                        "rules": rules,
                        "completed_games": game + 1,
                        "target_per_rule": args.games,
                    },
                )
            if (game + 1) % 50 == 0:
                print(f"{args.algorithm} {rules} {game + 1}/{args.games}", flush=True)
        selected = [r for r in rows if r["rules"] == rules]
        print(
            f"{args.algorithm} {rules}: black={sum(r['winner'] == 1 for r in selected)}/{len(selected)}",
            flush=True,
        )
    summaries = []
    for rules in ("freestyle", "renju"):
        sample = [r for r in rows if r["rules"] == rules]
        n = len(sample)
        p = sum(r["winner"] == 1 for r in sample) / n
        z = 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
        summaries.append(
            {
                "rules": rules,
                "games": n,
                "black_win_rate": p,
                "draw_rate": sum(r["winner"] == 0 for r in sample) / n,
                "wilson_low": center - radius,
                "wilson_high": center + radius,
            }
        )
    assert all(
        rows[i]["opening"] == rows[i + args.games]["opening"] for i in range(args.games)
    )
    atomic_json(
        out / "summary.json",
        {
            "rows": summaries,
            "matched_openings": True,
            "config": vars(args),
            "interpretation": "Rule effects under the frozen policy and random opening protocol; cannot prove first-player forced win or exclude learned color bias.",
        },
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/rules_greedy")
    p.add_argument("--algorithm", choices=["greedy", "alphazero"], default="greedy")
    p.add_argument("--model")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--opening-moves", type=int, default=4)
    p.add_argument("--seed", type=int, default=700000)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
