"""Development diagnostics: frozen tactical candidates, neural versus hand ranking.

This study does not train a model and does not constitute a held-out final test.
The old 400-game evaluation is preserved. Its losses are descriptive evidence,
never labels proving that a different earlier move would have won.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

import torch

from rl.agents.a2c_tactical import TacticalA2CAgent
from rl.agents.greedy_agent import GreedyAgent, GREEDY_WEIGHTS
from rl.agents.minimax_agent import MinimaxAgent
from rl.agents.q_learning import _features
from rl.agents.tactics import TacticalConfig
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation, state_to_tensor4, unpack
from rl.evaluation.eval_a2c_tactical import wilson
from rl.training.common import append_csv, atomic_json, choose, exclusive_run, seed_everything

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "runs/a2c_tactical_greedy_20261020"
CHECKPOINT = ROOT / "runs/production/a2c4_seed22/best.pt"
CHECKPOINT_SHA = "3ab5a205e2137ca8f3bde0d92c36d59930548a75bc6995cdcb2f98894f3ccbd2"
SOURCES = (
    "rl/agents/a2c.py", "rl/agents/a2c_tactical.py", "rl/agents/tactics.py",
    "rl/agents/greedy_agent.py", "rl/agents/q_learning.py", "rl/agents/minimax_agent.py",
    "rl/environment/gomoku.py", "rl/environment/rules.py", "rl/environment/state.py",
    "rl/training/common.py", "rl/evaluation/a2c_candidate_study.py",
)
DIRECT_REASONS = frozenset(("immediate_win", "block_immediate_win", "unavoidable_loss",
                            "create_double_threat", "forced_win"))
SEEDS = {("freestyle", 1): 3100000, ("freestyle", -1): 3200000,
         ("renju", 1): 3300000, ("renju", -1): 3400000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digests():
    return {name: digest(ROOT / name) for name in SOURCES}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rank_scores(scores, candidates):
    """Descending score, deterministic smallest-action tie break, no RNG use."""
    return sorted(candidates, key=lambda a: (-float(scores[a]), int(a)))


def hand_scores(board, legal):
    return {a: sum(GREEDY_WEIGHTS.get(k, 0.0) * v
                   for k, v in _features(board, a, 9, 5).items()) for a in legal}


def ranked_action(decision, neural_scores, manual_scores, ranking):
    if not decision.actions:
        raise ValueError("No tactical candidates")
    # All certified/direct tactical branches retain the original neural tie
    # resolution, even when more than one exact winning/blocking action exists.
    keep_neural = len(decision.actions) == 1 or decision.reason in DIRECT_REASONS
    scores = neural_scores if keep_neural or ranking == "a2c" else manual_scores
    return rank_scores(scores, decision.actions)[0]


class CandidateAgent(TacticalA2CAgent):
    def __init__(self, ranking="a2c", **kwargs):
        super().__init__(**kwargs)
        if ranking not in ("a2c", "hand"):
            raise ValueError("Unknown ranking")
        self.ranking = ranking
        self.last_study = None

    def select_action(self, state, legal_actions):
        decision = self.tactics.analyze(state, legal_actions)
        self.last_tactical_decision = decision
        x = state_to_tensor4(state, self.board_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, value = self.net(x)
        neural = {a: float(logits[0, a]) for a in legal_actions}
        manual = hand_scores(unpack(state)[0], legal_actions)
        neural_order = rank_scores(neural, decision.actions)
        hand_order = rank_scores(manual, decision.actions)
        action = ranked_action(decision, neural, manual, self.ranking)
        self.last_study = {
            "neural": neural, "manual": manual, "neural_order": neural_order,
            "hand_order": hand_order, "value": float(value[0]), "chosen": action,
            "ranking_applied": len(decision.actions) > 1 and decision.reason not in DIRECT_REASONS,
        }
        return action


def snapshot(agent, env, action):
    decision, data = agent.last_tactical_decision, agent.last_study
    return {
        "action": action, "candidate_actions": json.dumps(decision.actions),
        "candidate_count": len(decision.actions), "action_in_candidates": action in decision.actions,
        "neural_top": data["neural_order"][0], "hand_top": data["hand_order"][0],
        "neural_rank": data["neural_order"].index(action) + 1 if action in decision.actions else "",
        "hand_rank": data["hand_order"].index(action) + 1 if action in decision.actions else "",
        "ranking_applied": data["ranking_applied"], "ranking_disagrees": data["neural_order"][0] != data["hand_order"][0],
        "tactical_reason": decision.reason, "proven": decision.proven,
        "proof_scope": decision.details.get("proof_scope", ""),
        "nodes": decision.details.get("nodes", 0),
        "budget_exhausted": decision.details.get("budget_exhausted", False),
        "value": data["value"], "absolute_board": json.dumps(env.get_board(), separators=(",", ":")),
    }


def candidate_rows(agent, common, actual):
    data, candidates = agent.last_study, agent.last_tactical_decision.actions
    nr = {a: i + 1 for i, a in enumerate(data["neural_order"])}
    hr = {a: i + 1 for i, a in enumerate(data["hand_order"])}
    for action in candidates:
        yield {**common, "candidate_action": action, "neural_logit": data["neural"][action],
               "hand_score": data["manual"][action], "neural_rank": nr[action],
               "hand_rank": hr[action], "is_observed_action": action == actual}


def verify(protocol):
    if source_digests() != protocol["source_sha256"] or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Frozen source/checkpoint changed during the study")
    for name, expected in protocol["baseline_sha256"].items():
        if digest(BASELINE / name) != expected:
            raise RuntimeError("Baseline artifact changed: " + name)


def replay_losses(args, out, protocol):
    games = [r for r in read_csv(BASELINE / "games.csv") if int(r["outcome"]) == -1
             and r["rules"] == args.rules and int(r["color"]) == args.color]
    keys = {(r["rules"], int(r["color"]), int(r["game"])) for r in games}
    grouped = defaultdict(list)
    for row in read_csv(BASELINE / "moves.csv"):
        key = (row["rules"], int(row["color"]), int(row["game"]))
        if key in keys:
            grouped[key].append(row)
    agent = CandidateAgent(rules=args.rules, device="cpu", ranking="a2c")
    agent.load(CHECKPOINT)
    counters, example_rows = Counter(), []
    for index, game in enumerate(games):
        key = (args.rules, args.color, int(game["game"]))
        rows = grouped[key]
        agent_plies = [int(r["ply"]) for r in rows if r["actor"] == "a2c4_tactical"]
        last_agent_plies = agent_plies[-args.reference_turns:] if args.reference_turns else []
        env = GomokuEnv(rules=args.rules)
        opponent = GreedyAgent(seed=int(game["seed"]))
        for row in rows:
            action = int(row["action"])
            if env.done or env.current_player != int(row["player"]) or action not in env.get_legal_actions():
                raise AssertionError("Invalid original trajectory")
            if env.current_player == args.color:
                predicted = choose(agent, env)
                if predicted != action:
                    raise AssertionError(f"Original A2C not reproduced: {key}, ply {row['ply']}")
                common = {"rules": args.rules, "color": args.color, "game": int(game["game"]),
                          "seed": int(game["seed"]), "ply": int(row["ply"])}
                detail = {**common, **snapshot(agent, env, action)}
                detail.update(reference_action="", reference_in_candidates="", reference_value="", reference_nodes="")
                if int(row["ply"]) in last_agent_plies:
                    reference = MinimaxAgent(depth=2, candidate_limit=8, neighborhood=2, rules=args.rules)
                    ref_action = choose(reference, env)
                    detail.update(reference_action=ref_action,
                                  reference_in_candidates=ref_action in agent.last_tactical_decision.actions,
                                  reference_value=reference.last_search_value,
                                  reference_nodes=reference.last_search_nodes)
                    counters["reference_positions"] += 1
                    counters["reference_action_omitted"] += int(not detail["reference_in_candidates"])
                    if len(example_rows) < 24 and (not detail["reference_in_candidates"] or detail["ranking_disagrees"]):
                        example_rows.append(detail)
                append_csv(out / "positions.csv", detail)
                for candidate in candidate_rows(agent, common, action):
                    append_csv(out / "candidates.csv", candidate)
                counters["agent_positions"] += 1
                counters["ranking_positions"] += int(detail["ranking_applied"])
                counters["disagreements"] += int(detail["ranking_applied"] and detail["ranking_disagrees"])
                counters["budget_exhausted"] += int(detail["budget_exhausted"])
                counters["actual_outside_candidates"] += int(not detail["action_in_candidates"])
                counters["reason:" + detail["tactical_reason"]] += 1
            elif choose(opponent, env) != action:
                raise AssertionError("Original seeded Greedy action not reproduced")
            env.step(action)
        if not env.done or env.winner == args.color or env.winner == 0 or len(rows) != int(game["moves"]):
            raise AssertionError("Original loss outcome not reproduced")
        atomic_json(out / "status.json", {"state": "replaying", "losses_complete": index + 1, "losses": len(games)})
        print(f"replay {args.rules}/{args.color} {index + 1}/{len(games)}", flush=True)
    verify(protocol)
    summary = {"mode": "replay", "rules": args.rules, "color": args.color,
               "losses": len(games), "counts": dict(counters),
               "limitation": "Observed loss trajectories only. Minimax depth2 is a pruned heuristic reference, not ground truth or a counterfactual win label."}
    atomic_json(out / "examples.json", example_rows)
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", {"state": "complete", **summary})
    return summary


def development(args, out, protocol):
    agent = CandidateAgent(rules=args.rules, ranking=args.ranking, device="cpu")
    agent.load(CHECKPOINT)
    outcomes, timing = [], []
    for game in range(args.games):
        seed = SEEDS[(args.rules, args.color)] + game
        env, opponent = GomokuEnv(rules=args.rules), GreedyAgent(seed=seed)
        ply, counters = 0, Counter()
        while not env.done:
            actor = env.current_player == args.color
            before = time.perf_counter()
            action = choose(agent if actor else opponent, env)
            elapsed = time.perf_counter() - before
            common = {"ranking": args.ranking, "rules": args.rules, "color": args.color,
                      "game": game, "seed": seed, "ply": ply + 1}
            detail = snapshot(agent, env, action) if actor else None
            append_csv(out / "moves.csv", {**common, "player": env.current_player,
                       "actor": args.ranking if actor else "greedy", "action": action,
                       "inference_seconds": elapsed, "study": json.dumps(detail, separators=(",", ":")) if actor else ""})
            if actor:
                timing.append(elapsed)
                counters[agent.last_tactical_decision.reason] += 1
                for candidate in candidate_rows(agent, common, action):
                    append_csv(out / "candidates.csv", candidate)
            if action not in env.get_legal_actions():
                raise AssertionError("Illegal selected action")
            env.step(action)
            ply += 1
        outcome = 0 if env.winner == 0 else (1 if env.winner == args.color else -1)
        outcomes.append(outcome)
        append_csv(out / "games.csv", {"ranking": args.ranking, "rules": args.rules,
                   "color": args.color, "game": game, "seed": seed, "outcome": outcome,
                   "moves": ply, "reasons": json.dumps(dict(counters), sort_keys=True)})
        atomic_json(out / "status.json", {"state": "development", "completed": game + 1,
                                          "target": args.games, "wins": outcomes.count(1), "draws": outcomes.count(0), "losses": outcomes.count(-1)})
        print(f"dev {args.ranking}/{args.rules}/{args.color} {game + 1}/{args.games} "
              f"{outcomes.count(1)}/{outcomes.count(0)}/{outcomes.count(-1)}", flush=True)
    verify(protocol)
    wins, draws, losses = [outcomes.count(x) for x in (1, 0, -1)]
    summary = {"mode": "development", "ranking": args.ranking, "rules": args.rules,
               "color": args.color, "games": len(outcomes), "wins": wins, "draws": draws,
               "losses": losses, "win_rate": wins / len(outcomes),
               "score": (wins + .5 * draws) / len(outcomes), "wilson_95": wilson(wins, len(outcomes)),
               "mean_inference_ms": statistics.mean(timing) * 1000,
               "base_seed": SEEDS[(args.rules, args.color)]}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", {"state": "complete", **summary})
    return summary


def paired_summary(out, results):
    output = []
    for rules, color in SEEDS:
        a_path = out / f"development_a2c_{rules}_{color}" / "games.csv"
        h_path = out / f"development_hand_{rules}_{color}" / "games.csv"
        if not a_path.exists() or not h_path.exists():
            continue
        a, h = read_csv(a_path), read_csv(h_path)
        if [(r["seed"], r["game"]) for r in a] != [(r["seed"], r["game"]) for r in h]:
            raise AssertionError("Paired seed mismatch")
        differences = [(int(y["outcome"]) - int(x["outcome"])) / 2 for x, y in zip(a, h)]
        row = {"rules": rules, "color": color, "pairs": len(a),
               "score_difference_hand_minus_a2c": statistics.mean(differences),
               "paired_sample_sd": statistics.stdev(differences) if len(a) > 1 else None,
               "hand_better": sum(d > 0 for d in differences),
               "equal": sum(d == 0 for d in differences), "hand_worse": sum(d < 0 for d in differences)}
        output.append(row)
        for x, y, d in zip(a, h, differences):
            append_csv(out / "paired_games.csv", {"rules": rules, "color": color, "seed": x["seed"],
                       "a2c_outcome": x["outcome"], "hand_outcome": y["outcome"], "score_difference": d})
    return output


def report(out, protocol, results, paired):
    summary = {"protocol": protocol, "results": results, "paired": paired}
    atomic_json(out / "summary.json", summary)
    lines = ["# A2C candidate-ranking development study", "",
             "Frozen seed22/19000 weights; no training. These repeatedly usable development seeds are not a final held-out test.", "",
             "Direct/certified tactical branches retain the original A2C action. Only ordinary multi-candidate ranking changes.",
             "Hand score is exactly Greedy._features weighted by GREEDY_WEIGHTS; both rankings break ties by smallest action.",
             "Each branch uses the same TacticalEngine, but after a different choice subsequent boards and candidate sets can diverge.", "",
             "| Rule | Color | Ranking | W/D/L | Win rate | Wilson95 | Score |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        if r["mode"] == "development":
            lines.append(f"| {r['rules']} | {r['color']} | {r['ranking']} | {r['wins']}/{r['draws']}/{r['losses']} | {r['win_rate']:.1%} | {r['wilson_95'][0]:.1%}–{r['wilson_95'][1]:.1%} | {r['score']:.3f} |")
    lines += ["", "Paired score differences (hand minus A2C):", "", "```json", json.dumps(paired, indent=2), "```", "",
              "Loss replays are observational. Ranking disagreements and Minimax reference omissions do not prove causes of a defeat.",
              "Minimax uses depth2, candidate_limit8, neighborhood2 and alpha-beta; it is a heuristic reference, not a solver.",
              "Renju uses original freestyle-trained weights and rule-aware candidate legality. No result here establishes multi-seed trained superiority.", ""]
    for r in results:
        if r["mode"] == "replay":
            lines += [f"## Loss replay {r['rules']}/{r['color']}", "", "```json", json.dumps(r, indent=2), "```", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(out / "status.json", {"state": "complete", "cells": len(results)})


def protocol_for(args):
    baseline_protocol = json.loads((BASELINE / "protocol.json").read_text(encoding="utf-8"))
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("Unexpected selected weights")
    for name, expected in baseline_protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Original baseline source no longer matches: " + name)
    return {"started_at": datetime.now().astimezone().isoformat(), "study_role": "reusable development validation",
            "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(), "checkpoint_sha256": CHECKPOINT_SHA,
            "source_sha256": source_digests(), "tactical_config": asdict(TacticalConfig()),
            "baseline_sha256": {f: digest(BASELINE / f) for f in ("games.csv", "moves.csv", "protocol.json", "summary.json")},
            "training_performed": False, "hand_weights": GREEDY_WEIGHTS,
            "direct_reasons_unchanged": sorted(DIRECT_REASONS), "tie_break": "smallest action index",
            "games_per_ranking_per_cell": args.games,
            "development_seeds": {f"{r}/{c}": s for (r, c), s in SEEDS.items()},
            "reference": {"depth": 2, "candidate_limit": 8, "neighborhood": 2, "last_agent_turns_per_loss": args.reference_turns}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "replay", "development"), default="all")
    parser.add_argument("--rules", choices=("freestyle", "renju"))
    parser.add_argument("--color", type=int, choices=(1, -1))
    parser.add_argument("--ranking", choices=("a2c", "hand"))
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--reference-turns", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/a2c_candidate_study_20261020")
    args = parser.parse_args()
    if not 1 <= args.games <= 100000 or not 1 <= args.workers <= 4 or args.reference_turns < 0:
        raise ValueError("Invalid study budget")
    if args.mode != "all" and (args.rules is None or args.color is None):
        parser.error("Cell runs require --rules and --color")
    if args.mode == "development" and args.ranking is None:
        parser.error("Development cell requires --ranking")
    out = args.output.resolve()
    with exclusive_run(out / "study.lock"):
        if (out / "protocol.json").exists():
            raise ValueError("Use fresh output directory; old results are preserved")
        seed_everything(11)
        protocol = protocol_for(args)
        atomic_json(out / "protocol.json", protocol)
        try:
            if args.mode == "replay":
                replay_losses(args, out, protocol)
            elif args.mode == "development":
                development(args, out, protocol)
            else:
                cases = [("replay", r, c, None) for r, c in SEEDS]
                cases += [("development", r, c, rank) for r, c in SEEDS for rank in ("a2c", "hand")]
                def launch(case):
                    mode, rules, color, ranking = case
                    name = f"{mode}_{ranking + '_' if ranking else ''}{rules}_{color}"
                    target = out / name
                    target.mkdir(parents=True)
                    command = [sys.executable, "-u", "-m", "rl.evaluation.a2c_candidate_study", "--mode", mode,
                               "--rules", rules, "--color", str(color), "--games", str(args.games),
                               "--reference-turns", str(args.reference_turns), "--workers", "1", "--output", str(target)]
                    if ranking:
                        command += ["--ranking", ranking]
                    with (target / "console.log").open("a", encoding="utf-8") as stream:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        atomic_json(target / "launcher.json", {"pid": child.pid, "command": command})
                        code = child.wait()
                    if code:
                        raise RuntimeError(f"Cell {name} failed; see console.log")
                    result = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                    child_protocol = json.loads((target / "protocol.json").read_text(encoding="utf-8"))
                    if child_protocol["source_sha256"] != protocol["source_sha256"]:
                        raise RuntimeError("Cell source mismatch")
                    return result
                results = []
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(launch, case): case for case in cases}
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        atomic_json(out / "status.json", {"state": "running", "complete_cells": len(results), "total_cells": len(cases)})
                        print(json.dumps(result), flush=True)
                verify(protocol)
                report(out, protocol, results, paired_summary(out, results))
        except BaseException as error:
            atomic_json(out / "status.json", {"state": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
