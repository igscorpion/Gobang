"""Validation/final evaluation of separately named A2C research policies.

The test split is deliberately a separate explicit invocation. All policies
are frozen; this module never trains or selects a checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from rl.agents.a2c import A2CAgent
from rl.agents.tactics import TacticalConfig, TacticalEngine
from rl.environment.gomoku import GomokuEnv
from rl.evaluation.eval_a2c_tactical import wilson
from rl.training.common import append_csv, atomic_json, choose, exclusive_run, seed_everything

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TacticalPolicy:
    needs_last_move = True

    def __init__(self, policy, rules):
        self.policy = policy
        self.tactics = TacticalEngine(TacticalConfig(), rules)
        self.last_tactical_decision = None

    def select_action(self, state, legal_actions):
        decision = self.tactics.analyze(state, legal_actions)
        self.last_tactical_decision = decision
        if not decision.actions:
            raise ValueError("No legal tactical candidates")
        if len(decision.actions) == 1:
            return decision.actions[0]
        return self.policy.select_action(state, decision.actions)


def load_policy(path, kind, device, rules):
    if kind == "standard":
        agent = A2CAgent(device=device)
    else:
        from rl.agents.a2c_research import ModelSpec, ResearchA2CAgent
        metadata = torch.load(path, map_location="cpu", weights_only=True)
        spec = ModelSpec.from_dict(metadata["spec"]) if "spec" in metadata else ModelSpec()
        agent = ResearchA2CAgent(spec, device=device, rules=metadata.get("rules", "freestyle"))
    agent.load(path)
    if kind == "research":
        # This is explicitly recorded as transfer evaluation in the protocol.
        agent.rules = rules
    return agent


def make_agent(args):
    if args.mode == "minimax":
        from rl.agents.minimax_agent import MinimaxAgent
        return MinimaxAgent(depth=args.depth, candidate_limit=args.search_candidates, rules=args.rules)
    if args.mode == "alphazero":
        from rl.agents.alphazero import AlphaZeroAgent
        agent = AlphaZeroAgent(num_sims=args.sims, temperature=0.0, device=args.device)
        agent.load(args.checkpoint)
        return agent
    policy = load_policy(args.checkpoint, args.network_kind, args.device, args.rules)
    if args.mode == "policy":
        return policy
    if args.mode == "tactical":
        return TacticalPolicy(policy, args.rules)
    if args.mode == "hand":
        if args.network_kind != "standard":
            raise ValueError("The ranking ablation uses the original standard checkpoint")
        from rl.evaluation.a2c_candidate_study import CandidateAgent
        agent = CandidateAgent(ranking="hand", rules=args.rules, device=args.device)
        agent.load(args.checkpoint)
        return agent
    from rl.agents.a2c_search import SearchA2CAgent, SearchConfig
    config = SearchConfig(depth=args.depth, max_nodes=args.nodes,
                          candidate_limit=args.search_candidates,
                          learned_value_weight=args.value_weight)
    kwargs = {"rules": args.rules, "device": args.device, "search_config": config,
              "tactical_config": TacticalConfig(candidate_limit=args.tactical_candidates)}
    if args.network_kind == "research":
        kwargs["policy_agent"] = policy
    agent = SearchA2CAgent(**kwargs)
    if args.network_kind == "standard":
        agent.load(args.checkpoint)
    return agent


def source_hashes(args):
    files = ["rl/agents/a2c.py", "rl/agents/tactics.py", "rl/agents/greedy_agent.py",
             "rl/agents/q_learning.py", "rl/environment/rules.py", "rl/environment/gomoku.py",
             "rl/environment/state.py", "rl/training/common.py",
             "rl/evaluation/a2c_research_eval.py"]
    if args.network_kind == "research":
        files.append("rl/agents/a2c_research.py")
    if args.mode == "search":
        files.append("rl/agents/a2c_search.py")
    if args.mode == "hand":
        files.extend(["rl/evaluation/a2c_candidate_study.py", "rl/agents/a2c_tactical.py"])
    if args.mode in ("minimax", "alphazero"):
        files.append("rl/agents/" + ("minimax_agent.py" if args.mode == "minimax" else "alphazero.py"))
    return {name: digest(ROOT/name) for name in files}


def run(args):
    from rl.agents.greedy_agent import GreedyAgent
    from rl.agents.random_agent import RandomAgent
    if not 1 <= args.games <= 100000:
        raise ValueError("games must fit a disjoint 100000-seed range")
    if args.seed_black == args.seed_white or abs(args.seed_black-args.seed_white) < args.games:
        raise ValueError("Color seed intervals must be disjoint")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ("games.csv", "moves.csv", "summary.json", "protocol.json")):
        raise ValueError("Use a new output directory; old evaluations are preserved")
    seed_everything(args.seed_black)
    hashes = source_hashes(args)
    checkpoint_sha = digest(args.checkpoint) if args.checkpoint else None
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint and args.network_kind == "research" else {}
    protocol = {**vars(args), "checkpoint_sha256": checkpoint_sha,
                "source_sha256": hashes, "training_performed": False,
                "selection_performed": False,
                "training_rules": metadata.get("rules", "freestyle"),
                "rule_transfer": metadata.get("rules", "freestyle") != args.rules,
                "tactical_config": asdict(TacticalConfig()),
                "started_at": datetime.now().astimezone().isoformat()}
    atomic_json(out/"protocol.json", protocol)
    agent = make_agent(args)
    results = []
    started = time.perf_counter()
    for color, base_seed in ((1, args.seed_black), (-1, args.seed_white)):
        counts = Counter()
        times, lengths, reasons, nodes = [], [], Counter(), []
        for game in range(args.games):
            seed = base_seed+game
            opponent = GreedyAgent(seed=seed) if args.opponent == "greedy" else RandomAgent(seed=seed)
            env = GomokuEnv(rules=args.rules)
            ply, policy_seconds = 0, 0.0
            while not env.done:
                player = env.current_player
                is_agent = player == color
                begin = time.perf_counter()
                action = choose(agent if is_agent else opponent, env)
                elapsed = time.perf_counter()-begin
                decision = getattr(agent, "last_tactical_decision", None) if is_agent else None
                reason = decision.reason if decision else ""
                search_nodes = getattr(agent, "last_search_nodes", 0) if is_agent else 0
                if is_agent:
                    times.append(elapsed)
                    policy_seconds += elapsed
                    reasons[reason] += 1
                    nodes.append(search_nodes)
                if action not in env.get_legal_actions():
                    raise AssertionError("Policy returned an illegal action")
                env.step(action)
                ply += 1
                append_csv(out/"moves.csv", {
                    "rules":args.rules, "color":color, "game":game, "seed":seed,
                    "ply":ply, "player":player, "actor":"candidate" if is_agent else args.opponent,
                    "action":action, "inference_seconds":elapsed,
                    "tactical_reason":reason, "search_nodes":search_nodes})
            outcome = 0 if env.winner == 0 else (1 if env.winner == color else -1)
            counts[outcome] += 1
            lengths.append(ply)
            append_csv(out/"games.csv", {
                "split":args.split, "mode":args.mode, "opponent":args.opponent,
                "rules":args.rules, "color":color, "game":game, "seed":seed,
                "outcome":outcome, "moves":ply, "policy_seconds":policy_seconds})
            atomic_json(out/"status.json", {"state":"evaluating", "color":color,
                        "completed_games_in_color":game+1,"target_games_per_color":args.games})
            if (game+1)%10 == 0:
                print(f"{args.mode} {args.rules} color={color} {game+1}/{args.games} "
                      f"W/D/L={counts[1]}/{counts[0]}/{counts[-1]}", flush=True)
        ordered = sorted(times)
        row = {"color":color,"games":args.games,"wins":counts[1],"draws":counts[0],
               "losses":counts[-1],"win_rate":counts[1]/args.games,
               "score":(counts[1]+.5*counts[0])/args.games,
               "wilson_95":wilson(counts[1],args.games),
               "mean_game_plies":statistics.mean(lengths), "inference_samples":len(times),
               "mean_move_ms":1000*statistics.mean(times),
               "median_move_ms":1000*statistics.median(times),
               "p95_move_ms":1000*ordered[min(len(ordered)-1,int(.95*len(ordered)))],
               "max_move_ms":1000*ordered[-1],"tactical_reasons":dict(reasons),
               "mean_search_nodes":statistics.mean(nodes)}
        results.append(row)
    assert source_hashes(args) == hashes, "Evaluation source changed while running"
    if checkpoint_sha:
        assert digest(args.checkpoint) == checkpoint_sha, "Checkpoint changed while running"
    summary = {"protocol":protocol,"results":results,"total_games":2*args.games,
               "mean_score":statistics.mean(r["score"] for r in results),
               "mean_win_rate":statistics.mean(r["win_rate"] for r in results),
               "wall_seconds":time.perf_counter()-started,"sources_and_weights_unchanged":True}
    atomic_json(out/"summary.json", summary)
    atomic_json(out/"status.json", {"state":"complete","games":2*args.games})
    print(json.dumps({"state":"complete","output":str(out),"results":results}),flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--network-kind",choices=("standard","research"),default="standard")
    p.add_argument("--mode",choices=("policy","tactical","hand","search","minimax","alphazero"),default="tactical")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--rules",choices=("freestyle","renju"),default="freestyle")
    p.add_argument("--opponent",choices=("greedy","random"),default="greedy")
    p.add_argument("--split",choices=("validation","test","probe"),default="validation")
    p.add_argument("--games",type=int,default=40)
    p.add_argument("--seed-black",type=int,default=3100000)
    p.add_argument("--seed-white",type=int,default=3200000)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--nodes",type=int,default=2000)
    p.add_argument("--search-candidates",type=int,default=8)
    p.add_argument("--tactical-candidates",type=int,default=12)
    p.add_argument("--value-weight",type=float,default=0.0)
    p.add_argument("--sims",type=int,default=400)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    if args.mode != "minimax" and not args.checkpoint:
        p.error("This mode requires --checkpoint")
    with exclusive_run(Path(args.output)/"eval.lock"):
        try:
            run(args)
        except BaseException as error:
            atomic_json(Path(args.output)/"status.json", {"state":"failed", "error":repr(error)})
            raise


if __name__ == "__main__":
    main()
