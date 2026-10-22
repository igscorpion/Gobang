"""Frozen tactical-A2C versus seeded Greedy, with replayable move logs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.tactics import TacticalConfig
from rl.environment.gomoku import GomokuEnv
from rl.evaluation.eval_a2c_tactical import wilson
from rl.training.common import append_csv, atomic_json, choose, exclusive_run, seed_everything

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "rl/agents/a2c.py", "rl/agents/a2c_tactical.py", "rl/agents/tactics.py",
    "rl/agents/greedy_agent.py", "rl/agents/q_learning.py", "rl/environment/rules.py", "rl/environment/gomoku.py",
    "rl/environment/state.py", "rl/training/common.py",
    "rl/evaluation/eval_a2c_tactical.py", "rl/evaluation/eval_a2c_tactical_greedy.py",
)


def source_digests():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def save_result(out, protocol, results, wall_seconds, **details):
    summary = {"protocol": protocol, "results": results, "weights_unchanged": True,
               "wall_seconds": wall_seconds, "total_games": sum(row["games"] for row in results), **details}
    atomic_json(out / "summary.json", summary)
    lines = [
        "# A2C 战术增强版对 Greedy", "",
        "使用现有验证选出的 seed22／第19,000局权重，战术设置保持不变，没有重训。Greedy 按棋型评分选择合法落点，同分时由每局独立种子随机决定。", "",
        "| 规则 | 增强版执子 | 局数 | 胜／和／负 | 胜率（95% Wilson） | score | 平均总手数 | 每步中位／P95毫秒 |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    for row in results:
        low, high = row["wilson_95"]
        lines.append(f"| {'无禁手' if row['rules'] == 'freestyle' else '黑方禁手（9×9）'} | "
                     f"{'黑' if row['color'] == 1 else '白'} | {row['games']} | {row['wins']}／{row['draws']}／{row['losses']} | "
                     f"{row['win_rate']:.1%} ({low:.1%}–{high:.1%}) | {row['score']:.3f} | {row['mean_game_plies']:.2f} | "
                     f"{row['median_move_ms']:.2f}／{row['p95_move_ms']:.2f} |")
    lines.extend([
        "", f"合计 {summary['total_games']} 局；源文件与权重校验保持一致。逐局见 games.csv，全部落子及逐步时延见 moves.csv。", "",
        "score=胜率+0.5×和棋率。Wilson 区间仅对应胜率，不是 score 的区间。",
        "这是单一验证选定权重对固定 Greedy 策略的评估，不是多个训练种子的重复。禁手项使用 freestyle 权重迁移，未进行禁手重训。",
        f"评估进程数为 {protocol['parallel_workers']}；CPU 每进程单线程，缓存独立。时延包含当前运行负载，不与之前串行耗时直接比较。",
        "本轮使用新对手种子，未将之前每色20局的抽查并入，也未使用本次结果选模型或调整战术参数。", "",
    ])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(out / "status.json", {"state": "complete", "games": summary["total_games"], "weights_unchanged": True})
    print(json.dumps({"state": "complete", "report": str(out / "report.md"), "results": results}))
    return summary


def parallel_cells(args, out, protocol, started):
    """Each worker owns a different cell directory and its own inference cache."""
    cases = [(rules, color) for rules in args.rules for color in args.colors]
    atomic_json(out / "status.json", {"state": "evaluating", "completed_cells": 0, "target_cells": len(cases)})
    print(f"Starting {len(cases)} independent evaluation cells", flush=True)

    def launch(case):
        rules, color = case
        directory = out / "cells" / f"{rules}_{'black' if color == 1 else 'white'}"
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-m", "rl.evaluation.eval_a2c_tactical_greedy",
                   "--games", str(args.games), "--seed", str(args.seed), "--rules", rules,
                   "--colors", str(color), "--device", args.device, "--workers", "1", "--output", str(directory)]
        with (directory / "console.log").open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            atomic_json(directory / "launcher.json", {"pid": process.pid, "command": command})
            code = process.wait()
        if code:
            raise RuntimeError(f"Evaluation cell {rules}/{color} exited {code}; see {directory / 'console.log'}")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if len(summary["results"]) != 1 or summary["protocol"]["source_sha256"] != protocol["source_sha256"]:
            raise RuntimeError("Cell source hashes or shape do not match the frozen parent protocol")
        if summary["protocol"]["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise RuntimeError("Cell checkpoint differs from the parent checkpoint")
        return directory, summary

    collected = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(cases))) as pool:
        futures = {pool.submit(launch, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            collected[case] = future.result()
            row = collected[case][1]["results"][0]
            atomic_json(out / "status.json", {"state": "evaluating", "completed_cells": len(collected),
                                              "target_cells": len(cases), "last_completed": list(case)})
            print(f"cell complete {case}: {row['wins']}/{row['draws']}/{row['losses']}", flush=True)
    for filename in ("games.csv", "moves.csv"):
        with (out / filename).open("w", encoding="utf-8", newline="") as destination:
            writer = None
            for case in cases:
                with (collected[case][0] / filename).open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if writer is None:
                        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)
    if source_digests() != protocol["source_sha256"]:
        raise RuntimeError("Source changed during evaluation")
    if hashlib.sha256((ROOT / protocol["checkpoint"]).read_bytes()).hexdigest() != protocol["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during evaluation")
    return save_result(out, protocol, [collected[case][1]["results"][0] for case in cases],
                       time.perf_counter() - started, parallel_workers=min(args.workers, len(cases)),
                       cell_wall_seconds={f"{r}/{c}": collected[(r, c)][1]["wall_seconds"] for r, c in cases})


def run(args):
    if not 1 <= args.games <= 100000:
        raise ValueError("games must be between 1 and 100000 to keep seed ranges disjoint")
    if len(set(args.rules)) != len(args.rules):
        raise ValueError("rules may not contain duplicate modes")
    if len(set(args.colors)) != len(args.colors) or not 1 <= args.workers <= 4:
        raise ValueError("colors must be unique and workers must be between 1 and 4")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("Parallel evaluation uses CPU; set workers=1 for CUDA")
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
        "algorithm": "a2c4_tactical", "opponent": "greedy",
        "opponent_policy": "highest hand-coded tactical score over legal actions, with seeded uniform tie-breaking",
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": digest,
        "checkpoint_selection": source["selection"], "source_training_seed": source["seed"],
        "selected_episode": source["selected_episode"], "training_performed": False,
        "rules": args.rules, "games_per_color_per_rule": args.games,
        "colors": args.colors, "parallel_workers": min(args.workers, len(args.rules) * len(args.colors)),
        "base_seed": args.seed, "color_offset": 100000, "rule_offset": 200000,
        "seed_mapping": "base + (0 if freestyle else 200000) + (0 if black else 100000) + game_index",
        "device": args.device, "tactical_config": asdict(config), "source_sha256": sources,
        "started_at": datetime.now().astimezone().isoformat(),
        "limitations": [
            "One preselected trained checkpoint, not replicated training across seeds.",
            "Renju uses freestyle-trained weights with rule-aware inference, not Renju retraining.",
            "Win-rate intervals describe this frozen policy against the fixed Greedy policy with seeded tie-breaking.",
            "No model, tactical configuration or search budget is selected using these results.",
        ],
    }
    atomic_json(out / "protocol.json", protocol)
    started = time.perf_counter()
    if args.workers > 1 and len(args.rules) * len(args.colors) > 1:
        return parallel_cells(args, out, protocol, started)
    results = []
    for rules in args.rules:
        agent = TacticalA2CAgent(device=args.device, seed=args.seed, rules=rules, tactical_config=config)
        agent.load(checkpoint)
        for color in args.colors:
            outcomes, latencies, game_lengths, reasons = [], [], [], Counter()
            for game in range(args.games):
                seed = args.seed + (0 if rules == "freestyle" else 200000) + (0 if color == 1 else 100000) + game
                opponent = GreedyAgent(seed=seed)
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
                        "ply": ply, "player": player, "actor": "a2c4_tactical" if is_agent else "greedy",
                        "action": action, "inference_seconds": elapsed,
                        "tactical_reason": decision.reason if decision else "",
                        "proven": decision.proven if decision else "",
                        "proof_scope": decision.details.get("proof_scope", "") if decision else "",
                    })
                outcome = 0 if env.winner == 0 else (1 if env.winner == color else -1)
                outcomes.append(outcome)
                game_lengths.append(ply)
                append_csv(out / "games.csv", {
                    "algorithm": "a2c4_tactical", "opponent": "greedy", "rules": rules,
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
                "algorithm": "a2c4_tactical", "opponent": "greedy", "rules": rules, "color": color,
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
    return save_result(out, protocol, results, time.perf_counter() - started, parallel_workers=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2000000)
    parser.add_argument("--rules", nargs="+", choices=("freestyle", "renju"), default=["freestyle", "renju"])
    parser.add_argument("--colors", nargs="+", type=int, choices=(1, -1), default=[1, -1])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a2c_tactical_greedy_20261020")
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
