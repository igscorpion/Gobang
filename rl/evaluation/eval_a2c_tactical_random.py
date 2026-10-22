"""Frozen tactical-A2C versus uniform legal Random, with replayable move logs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.tactics import TacticalConfig
from rl.environment.gomoku import GomokuEnv
from rl.evaluation.eval_a2c_tactical import wilson
from rl.training.common import append_csv, atomic_json, choose, exclusive_run, seed_everything

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "rl/agents/a2c.py", "rl/agents/a2c_tactical.py", "rl/agents/tactics.py",
    "rl/agents/random_agent.py", "rl/environment/rules.py", "rl/environment/gomoku.py",
    "rl/environment/state.py", "rl/training/common.py",
    "rl/evaluation/eval_a2c_tactical.py", "rl/evaluation/eval_a2c_tactical_random.py",
)


def source_digests():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def run(args):
    if not 1 <= args.games <= 100000:
        raise ValueError("games must be between 1 and 100000 to keep seed ranges disjoint")
    if len(set(args.rules)) != len(args.rules):
        raise ValueError("rules may not contain duplicate modes")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "games.csv").exists() or (out / "summary.json").exists():
        raise ValueError("Use a fresh output directory; existing results are preserved")
    source = json.loads((ROOT / "runs/production/a2c_models.json").read_text(encoding="utf-8"))["a2c4"]
    checkpoint = (ROOT / source["checkpoint"]).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sources = source_digests()
    config = TacticalConfig()
    seed_everything(args.seed)
    protocol = {
        "algorithm": "a2c4_tactical", "opponent": "random",
        "opponent_policy": "uniform independent choice over the environment's legal actions",
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": digest,
        "checkpoint_selection": source["selection"], "source_training_seed": source["seed"],
        "selected_episode": source["selected_episode"], "training_performed": False,
        "rules": args.rules, "games_per_color_per_rule": args.games,
        "base_seed": args.seed, "color_offset": 100000, "rule_offset": 200000,
        "seed_mapping": "base + (0 if freestyle else 200000) + (0 if black else 100000) + game_index",
        "device": args.device, "tactical_config": asdict(config), "source_sha256": sources,
        "started_at": datetime.now().astimezone().isoformat(),
        "limitations": [
            "One preselected trained checkpoint, not replicated training across seeds.",
            "Renju uses freestyle-trained weights with rule-aware inference, not Renju retraining.",
            "Win-rate intervals describe this frozen policy against uniform legal Random.",
            "No model, tactical configuration or search budget is selected using these results.",
        ],
    }
    atomic_json(out / "protocol.json", protocol)
    started = time.perf_counter()
    results = []
    for rules in args.rules:
        agent = TacticalA2CAgent(device=args.device, seed=args.seed, rules=rules, tactical_config=config)
        agent.load(checkpoint)
        for color in (1, -1):
            outcomes, latencies, game_lengths, reasons = [], [], [], Counter()
            for game in range(args.games):
                seed = args.seed + (0 if rules == "freestyle" else 200000) + (0 if color == 1 else 100000) + game
                opponent = RandomAgent(seed=seed)
                agent.rng.seed(seed + 500000)
                env = GomokuEnv(rules=rules)
                ply, game_latency = 0, 0.0
                game_reasons = Counter()
                while not env.done:
                    player = env.current_player
                    is_agent = player == color
                    before = time.perf_counter()
                    action = choose(agent if is_agent else opponent, env)
                    elapsed = time.perf_counter() - before
                    decision = agent.last_tactical_decision if is_agent else None
                    if is_agent:
                        latencies.append(elapsed)
                        game_latency += elapsed
                        reasons[decision.reason] += 1
                        game_reasons[decision.reason] += 1
                    if action not in env.get_legal_actions():
                        raise AssertionError("Illegal move returned by a policy")
                    env.step(action)
                    ply += 1
                    append_csv(out / "moves.csv", {
                        "rules": rules, "color": color, "game": game, "seed": seed,
                        "ply": ply, "player": player, "actor": "a2c4_tactical" if is_agent else "random",
                        "action": action, "inference_seconds": elapsed,
                        "tactical_reason": decision.reason if decision else "",
                        "proven": decision.proven if decision else "",
                        "proof_scope": decision.details.get("proof_scope", "") if decision else "",
                    })
                outcome = 0 if env.winner == 0 else (1 if env.winner == color else -1)
                outcomes.append(outcome)
                game_lengths.append(ply)
                append_csv(out / "games.csv", {
                    "algorithm": "a2c4_tactical", "opponent": "random", "rules": rules,
                    "color": color, "game": game, "seed": seed, "outcome": outcome,
                    "moves": ply, "policy_seconds": game_latency,
                    "tactical_reasons": json.dumps(dict(game_reasons), sort_keys=True),
                })
                atomic_json(out / "status.json", {
                    "state": "evaluating", "rules": rules, "color": color,
                    "completed_games_in_cell": game + 1, "target_games_in_cell": args.games,
                    "completed_cells": len(results),
                })
                if (game + 1) % 10 == 0 or game + 1 == args.games:
                    print(f"{rules} color={color} {game + 1}/{args.games} W/D/L="
                          f"{outcomes.count(1)}/{outcomes.count(0)}/{outcomes.count(-1)}", flush=True)
            wins, draws, losses = (outcomes.count(value) for value in (1, 0, -1))
            ordered = sorted(latencies)
            results.append({
                "algorithm": "a2c4_tactical", "opponent": "random", "rules": rules, "color": color,
                "games": args.games, "wins": wins, "draws": draws, "losses": losses,
                "win_rate": wins / args.games, "score": (wins + .5 * draws) / args.games,
                "wilson_95": wilson(wins, args.games), "mean_game_plies": statistics.mean(game_lengths),
                "inference_samples": len(latencies), "mean_move_ms": 1000 * statistics.mean(latencies),
                "median_move_ms": 1000 * statistics.median(latencies),
                "p95_move_ms": 1000 * ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
                "max_move_ms": 1000 * ordered[-1], "tactical_reasons": dict(reasons),
            })
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    if source_digests() != sources:
        raise RuntimeError("Source changed during evaluation; freeze source and use a fresh output")
    summary = {"protocol": protocol, "results": results, "weights_unchanged": True,
               "wall_seconds": time.perf_counter() - started, "total_games": args.games * 2 * len(args.rules)}
    atomic_json(out / "summary.json", summary)
    lines = [
        "# A2C 战术增强版对 Random", "",
        "使用现有验证选出的 seed22／第19,000局权重，保持战术设置不变；没有重训。Random 在当前规则的全部合法落点中均匀随机选择。", "",
        "| 规则 | 增强版执子 | 局数 | 胜／和／负 | 胜率（95% Wilson） | 平均总手数 | 每步中位／P95毫秒 |",
        "|---|---|---:|---|---|---:|---|",
    ]
    for row in results:
        low, high = row["wilson_95"]
        lines.append(f"| {'无禁手' if row['rules'] == 'freestyle' else '黑方禁手（9×9）'} | "
                     f"{'黑' if row['color'] == 1 else '白'} | {row['games']} | {row['wins']}／{row['draws']}／{row['losses']} | "
                     f"{row['win_rate']:.1%} ({low:.1%}–{high:.1%}) | {row['mean_game_plies']:.2f} | "
                     f"{row['median_move_ms']:.2f}／{row['p95_move_ms']:.2f} |")
    lines.extend([
        "", f"合计 {summary['total_games']} 局；源文件与权重校验保持一致。逐局见 games.csv，全部落子及逐步时延见 moves.csv。", "",
        "本结果只描述一个固定增强策略对随机合法策略的表现，不替代多训练种子验证，也不能据此推断对 Greedy／人类的棋力。",
        "禁手项为 freestyle 权重迁移，未进行禁手重训。没有使用本次结果选模型或调整战术参数。", "",
    ])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(out / "status.json", {"state": "complete", "games": summary["total_games"], "weights_unchanged": True})
    print(json.dumps({"state": "complete", "report": str(out / "report.md"), "results": results}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1600000)
    parser.add_argument("--rules", nargs="+", choices=("freestyle", "renju"), default=["freestyle", "renju"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a2c_tactical_random_20261020")
    args = parser.parse_args()
    with exclusive_run(args.output / "evaluation.lock"):
        try:
            run(args)
        except BaseException as error:
            if not (args.output / "summary.json").exists():
                atomic_json(args.output / "status.json", {
                    "state": "failed", "error_type": type(error).__name__, "message": str(error),
                })
            raise


if __name__ == "__main__":
    main()
