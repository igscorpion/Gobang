"""Frozen paired smoke comparison of raw and tactical A2C, without training."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from rl.agents.a2c import A2CAgent
from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.tactics import TacticalConfig
from rl.environment.gomoku import GomokuEnv
from rl.training.common import append_csv, atomic_json, choose, exclusive_run, seed_everything

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "rl/agents/a2c.py", "rl/agents/a2c_tactical.py", "rl/agents/tactics.py",
    "rl/environment/rules.py", "rl/evaluation/eval_a2c_tactical.py",
)


def source_digests():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def wilson(wins, games):
    z = 1.96
    rate = wins / games
    scale = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def run(args):
    if args.games < 1:
        raise ValueError("games must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "games.csv").exists() or (out / "summary.json").exists():
        raise ValueError("Use a fresh output directory; prior results are never overwritten")
    registry = json.loads((ROOT / "runs/production/a2c_models.json").read_text(encoding="utf-8"))
    source = registry["a2c4"]
    checkpoint = (ROOT / source["checkpoint"]).resolve()
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sources = source_digests()
    config = TacticalConfig(
        forcing_depth=args.forcing_depth, max_nodes=args.max_nodes,
        candidate_limit=args.candidate_limit,
    )
    seed_everything(args.seed)
    results = []
    started = time.perf_counter()
    protocol = {
        "kind": "paired_smoke_comparison_not_training_seed_replication",
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": checksum,
        "checkpoint_selection": source["selection"], "training_performed": False,
        "rules": args.rules, "games_per_color": args.games, "opponent": "greedy",
        "base_seed": args.seed, "color_offset": 100000, "rule_offset": 200000,
        "device": args.device, "tactical_config": asdict(config),
        "source_sha256": sources,
        "started_at": datetime.now().astimezone().isoformat(),
        "limitations": [
            "Both policies use one already validation-selected A2C checkpoint.",
            "Opponent seeds are paired, but policies induce different game trajectories.",
            "This is not a new three-training-seed experiment or a replacement for the original final tests.",
            "Freestyle-trained weights in renju measure rule transfer, not renju training.",
            "Tactical A2C uses additional CPU search/heuristics; runtime is reported separately.",
        ],
    }
    atomic_json(out / "protocol.json", protocol)
    for rule_index, rules in enumerate(args.rules):
        for name, cls in (("a2c4", A2CAgent), ("a2c4_tactical", TacticalA2CAgent)):
            options = {"device": args.device, "seed": args.seed}
            if name == "a2c4_tactical":
                options.update(tactical_config=config, rules=rules)
            agent = cls(**options)
            agent.load(checkpoint)
            for color in (1, -1):
                outcomes, latencies, reasons = [], [], Counter()
                for game in range(args.games):
                    seed = args.seed + rule_index * 200000 + (100000 if color == -1 else 0) + game
                    opponent = GreedyAgent(seed=seed)
                    env = GomokuEnv(rules=rules)
                    game_reasons = Counter()
                    game_latencies = []
                    moves = 0
                    while not env.done:
                        if env.current_player == color:
                            before = time.perf_counter()
                            action = choose(agent, env)
                            elapsed = time.perf_counter() - before
                            latencies.append(elapsed)
                            game_latencies.append(elapsed)
                            decision = getattr(agent, "last_tactical_decision", None)
                            if decision is not None:
                                reasons[decision.reason] += 1
                                game_reasons[decision.reason] += 1
                        else:
                            action = choose(opponent, env)
                        if action not in env.get_legal_actions():
                            raise AssertionError("Policy selected an illegal move")
                        env.step(action)
                        moves += 1
                    outcome = 0 if env.winner == 0 else (1 if env.winner == color else -1)
                    outcomes.append(outcome)
                    append_csv(out / "games.csv", {
                        "algorithm": name, "rules": rules, "color": color,
                        "game": game, "seed": seed, "outcome": outcome, "moves": moves,
                        "policy_seconds": sum(game_latencies),
                        "tactical_reasons": json.dumps(dict(game_reasons), sort_keys=True),
                    })
                    atomic_json(out / "status.json", {
                        "state": "evaluating", "rules": rules, "algorithm": name,
                        "color": color, "completed_games_in_cell": game + 1,
                        "target_games_in_cell": args.games,
                    })
                    print(f"{rules} {name} color={color} game={game + 1}/{args.games} outcome={outcome}", flush=True)
                wins, draws, losses = (outcomes.count(value) for value in (1, 0, -1))
                ordered = sorted(latencies)
                results.append({
                    "algorithm": name, "rules": rules, "color": color,
                    "games": args.games, "wins": wins, "draws": draws, "losses": losses,
                    "win_rate": wins / args.games, "score": (wins + .5 * draws) / args.games,
                    "wilson_95": wilson(wins, args.games),
                    "inference_samples": len(latencies),
                    "mean_move_ms": 1000 * statistics.mean(latencies),
                    "median_move_ms": 1000 * statistics.median(latencies),
                    "p95_move_ms": 1000 * ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
                    "tactical_reasons": dict(reasons),
                })
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
    if source_digests() != sources:
        raise RuntimeError("Source changed during evaluation; repeat from a fresh output after freezing code")
    summary = {"protocol": protocol, "results": results, "weights_unchanged": True,
               "wall_seconds": time.perf_counter() - started}
    atomic_json(out / "summary.json", summary)
    lines = [
        "# A2C 战术增强配对冒烟评估", "",
        "原始 A2C 与增强版复用同一个验证选出的权重；没有重训。每个对手/颜色使用配对的新种子。", "",
        "| 规则 | 方法 | 执子 | 胜/和/负 | 胜率（95% Wilson） | 每步中位 / P95 毫秒 |",
        "|---|---|---|---|---|---|",
    ]
    for row in results:
        low, high = row["wilson_95"]
        lines.append(f"| {row['rules']} | {row['algorithm']} | {'黑' if row['color'] == 1 else '白'} | "
                     f"{row['wins']}/{row['draws']}/{row['losses']} | {row['win_rate']:.1%} ({low:.1%}–{high:.1%}) | "
                     f"{row['median_move_ms']:.2f} / {row['p95_move_ms']:.2f} |")
    lines.extend(["", "此处是单一权重上的功能与棋力抽查，不代表三训练种子稳定优势。禁手项使用 freestyle 权重迁移，未进行禁手重训。",
                  "增强版包含手工战术和有限搜索，额外耗时计入每步延迟。结果不覆盖原始 A2C 正式实验。", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(out / "status.json", {"state": "complete", "games": args.games * 4 * len(args.rules),
                                      "weights_unchanged": True})
    print(json.dumps({"state": "complete", "report": str(out / "report.md")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1200000)
    parser.add_argument("--rules", nargs="+", choices=("freestyle", "renju"), default=["freestyle", "renju"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--forcing-depth", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=1500)
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a2c_tactics")
    args = parser.parse_args()
    with exclusive_run(args.output / "evaluation.lock"):
        try:
            run(args)
        except BaseException as error:
            # An accidental attempt to reuse completed output must leave it intact.
            if not (args.output / "summary.json").exists():
                atomic_json(args.output / "status.json", {
                    "state": "failed", "error_type": type(error).__name__, "message": str(error),
                })
            raise


if __name__ == "__main__":
    main()
