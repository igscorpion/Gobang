"""Fresh strong-teacher demonstrations with per-game commits and global splitting.

The default teacher is the existing wide AlphaZero checkpoint with 400 MCTS
simulations and zero temperature. This identifies a teacher, not a claim about
its strength: evaluate teacher quality separately before launching production.
No old evaluation game is imported. Each worker owns a locked shard; a complete
game is the smallest durable unit, so an interrupted game is regenerated from
its independent seed. Validation leakage is removed globally after all games.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import torch

from rl.agents.alphazero import AlphaZeroAgent
from rl.agents.greedy_agent import GreedyAgent
from rl.environment.gomoku import GomokuEnv
from rl.environment.state import observation, state_to_tensor4
from rl.training.a2c_research import (DEMO_FORMAT, _pack_rows, canonical_position_id,
    load_demonstrations, reject_seed_overlap, source_hashes, split_position_id)
from rl.training.common import atomic_json, atomic_torch, choose, exclusive_run, seed_everything

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = ROOT / "runs/production/az_width128_fixed/best.pt"
GAME_FORMAT = "a2c_expert_complete_game_v1"
GENERATOR_FORMAT = "a2c_parallel_expert_generator_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule(args):
    train = list(range(args.train_seed, args.train_seed + args.train_games))
    validation = list(range(args.validation_seed, args.validation_seed + args.validation_games))
    reject_seed_overlap(train, validation)
    return [(split, index, seed) for split, seeds in (("train", train), ("validation", validation))
            for index, seed in enumerate(seeds)]


def game_id(split, index, seed):
    return f"{split}_{index:06d}_seed{seed}"


def game_path(out, global_index, entry, workers):
    return Path(out) / "shards" / f"worker_{global_index % workers:02d}" / "games" / (game_id(*entry) + ".pt")


def source_provenance():
    result = source_hashes()
    for name in ("rl/agents/alphazero.py", "rl/agents/base_agent.py", "rl/training/a2c_expert_dataset.py"):
        result[name] = sha256(ROOT / name)
    return result


def make_protocol(args, expert_metadata=None):
    settings = {key: value for key, value in vars(args).items()
                if key not in ("output", "resume", "worker_index")}
    settings["teacher_checkpoint"] = str(Path(args.teacher_checkpoint).resolve())
    settings["sims"] = args.simulations
    if expert_metadata is None:
        path = Path(args.teacher_checkpoint).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        weights = payload.get("state_dict", payload)
        channels = payload.get("channels", tuple(weights[f"conv.{i}.weight"].shape[0] for i in (0, 2, 4)))
        expert_metadata = {
            "kind": "alphazero_mcts", "checkpoint": str(path), "checkpoint_sha256": sha256(path),
            "channels": list(channels), "input_channels": 4,
            "num_simulations": args.simulations, "c_puct": args.c_puct,
            "temperature": 0.0, "opening_moves": 6, "candidate_ratio": 0.1,
            "root_dirichlet_noise": False, "noise_eps_inactive": 0.25,
            "noise_alpha_inactive": 0.3, "reuse_tree_between_labels": True,
            "new_tree_each_game": True, "device": args.device,
            "board_size": 9, "win_count": 5, "rules": args.rules,
        }
    payload = {
        "format": GENERATOR_FORMAT, "source": "fresh_generated_expert_games",
        "config": settings, "sources": source_provenance(), "expert": expert_metadata,
        "external_trajectories_allowed": False,
        "split": "disjoint game seeds; global board+side+D4 train/validation exclusion, plus validation internal deduplication",
        "behavior": {"teacher": args.teacher_probability, "greedy": args.greedy_probability,
                     "random": args.random_probability},
        "prefix": {"distribution": "independent uniform integer 0..maximum, then uniform legal actions",
                   "maximum_plies": args.random_opening, "prefix_states_labeled": False},
        "durable_unit": "complete independently seeded game", "torch_version": str(torch.__version__),
        "python_version": sys.version,
    }
    payload["protocol_id"] = stable_hash(payload)
    return payload


def verify_current(protocol, args, expert_metadata=None):
    current = make_protocol(args, expert_metadata)
    if current != protocol:
        raise ValueError("Source, teacher checkpoint or generation settings changed; use a fresh dataset")


def make_teacher(args, seed):
    teacher = AlphaZeroAgent(num_sims=args.simulations, c_puct=args.c_puct,
                            device=args.device, seed=seed, temperature=0.0, rules=args.rules)
    teacher.load(args.teacher_checkpoint)
    teacher.net.eval()
    teacher.mcts.add_noise = False
    return teacher


def complete_game(args, entry, protocol, expert_factory=None):
    split, index, seed = entry
    seed_everything(seed)
    rng, env = random.Random(seed), GomokuEnv(rules=args.rules)
    teacher = expert_factory(seed, args.rules) if expert_factory else make_teacher(args, seed)
    greedy = GreedyAgent(seed=seed + 100_000_000)
    prefix, trace, rows = [], [], []
    for _ in range(rng.randrange(args.random_opening + 1)):
        if env.done:
            break
        action = rng.choice(env.get_legal_actions())
        prefix.append(action)
        env.step(action)
    while not env.done:
        legal = env.get_legal_actions()
        action = int(choose(teacher, env))
        if action not in legal:
            raise ValueError("Teacher returned an illegal label")
        base = state_to_tensor4(observation(env)).to(torch.uint8)
        mask = torch.zeros(81, dtype=torch.bool)
        mask[legal] = True
        number = rng.random()
        if number < args.teacher_probability:
            behavior, played = "teacher", action
        elif number < args.teacher_probability + args.greedy_probability:
            behavior, played = "greedy", choose(greedy, env)
        else:
            behavior, played = "random", rng.choice(legal)
        rows.append((base, mask, action, seed, canonical_position_id(base), split_position_id(base)))
        trace.append({"ply": len(prefix) + len(trace) + 1, "player": env.current_player,
                      "teacher_action": action, "behavior": behavior, "action": int(played)})
        env.step(played)
    return {"format": GAME_FORMAT, "protocol_id": protocol["protocol_id"],
            "game_id": game_id(*entry), "split": split, "index": index, "seed": seed,
            "prefix": prefix, "trace": trace, "rows": rows, "winner": env.winner,
            "total_plies": len(prefix) + len(trace)}


def verify_game(game, entry, protocol):
    """Independent trajectory replay checks labels, masks, IDs and game outcomes.

    It does not rerun the expensive teacher search; label provenance is frozen
    by checkpoint/source/config hashes. Prefix and behavior RNG are reproduced.
    """
    split, index, seed = entry
    if (game.get("format") != GAME_FORMAT or game.get("protocol_id") != protocol["protocol_id"]
            or game.get("game_id") != game_id(*entry) or game.get("split") != split
            or game.get("index") != index or game.get("seed") != seed):
        raise ValueError("Committed game identity/protocol mismatch")
    config = protocol["config"]
    rng, env = random.Random(seed), GomokuEnv(rules=config["rules"])
    prefix_length = rng.randrange(config["random_opening"] + 1)
    if len(game["prefix"]) != prefix_length:
        raise ValueError("Prefix length differs from seeded schedule")
    for action in game["prefix"]:
        if env.done or action != rng.choice(env.get_legal_actions()):
            raise ValueError("Prefix differs from seeded legal behavior")
        env.step(action)
    if len(game["rows"]) != len(game["trace"]) or not game["rows"]:
        raise ValueError("Game has missing label rows")
    greedy = GreedyAgent(seed=seed + 100_000_000)
    counts = Counter()
    for position, (row, move) in enumerate(zip(game["rows"], game["trace"])):
        if env.done:
            raise ValueError("Trajectory continues after terminal state")
        base, mask, action, row_seed, identity, split_id = row
        legal = env.get_legal_actions()
        expected_mask = torch.zeros(81, dtype=torch.bool)
        expected_mask[legal] = True
        if (not torch.equal(base, state_to_tensor4(observation(env)).to(torch.uint8))
                or not torch.equal(mask, expected_mask) or action not in legal or row_seed != seed
                or identity != canonical_position_id(base) or split_id != split_position_id(base)):
            raise ValueError("Invalid teacher label, observation, mask or position identity")
        if move["ply"] != len(game["prefix"]) + position + 1 or move["player"] != env.current_player or move["teacher_action"] != action:
            raise ValueError("Move/label alignment mismatch")
        sample = rng.random()
        if sample < config["teacher_probability"]:
            behavior, played = "teacher", action
        elif sample < config["teacher_probability"] + config["greedy_probability"]:
            behavior, played = "greedy", choose(greedy, env)
        else:
            behavior, played = "random", rng.choice(legal)
        if move["behavior"] != behavior or move["action"] != played or played not in legal:
            raise ValueError("Seeded behavior trajectory mismatch")
        env.step(played)
        counts[behavior] += 1
    if not env.done or env.winner != game["winner"] or game["total_plies"] != len(game["prefix"]) + len(game["trace"]):
        raise ValueError("Incomplete or incorrect game outcome")
    return dict(counts)


def worker(args, protocol, index, expert_factory=None):
    out = Path(args.output)
    shard = out / "shards" / f"worker_{index:02d}"
    with exclusive_run(shard / "run.lock"):
        entries = [(n, entry) for n, entry in enumerate(schedule(args)) if n % args.workers == index]
        (shard / "games").mkdir(parents=True, exist_ok=True)
        expected_names = {game_path(out, n, entry, args.workers).name for n, entry in entries}
        if {p.name for p in (shard / "games").glob("*.pt")} - expected_names:
            raise ValueError("Unexpected committed games in worker shard")
        for count, (n, entry) in enumerate(entries, 1):
            path = game_path(out, n, entry, args.workers)
            reused = path.exists()
            if reused:
                game = torch.load(path, map_location="cpu", weights_only=True)
            else:
                game = complete_game(args, entry, protocol, expert_factory)
            verify_game(game, entry, protocol)
            if not reused:
                atomic_torch(path, game)
            atomic_json(shard / "status.json", {"state": "generating", "pid": os.getpid(),
                "committed_games": count, "target_games": len(entries), "last_game_id": game["game_id"]})
            print(f"worker={index} complete {count}/{len(entries)} {game['game_id']} reused={reused}", flush=True)
        atomic_json(shard / "status.json", {"state": "complete", "pid": os.getpid(),
            "committed_games": len(entries), "target_games": len(entries)})


def merge_games(args, out, protocol):
    rows, train_seen, validation_seen = {"train": [], "validation": []}, set(), set()
    dropped, fingerprints, behavior = Counter(), {}, Counter()
    schedule_rows = schedule(args)
    expected_paths = {game_path(out, n, e, args.workers).resolve() for n, e in enumerate(schedule_rows)}
    actual_paths = {p.resolve() for p in (out / "shards").glob("worker_*/games/*.pt")}
    if expected_paths != actual_paths:
        raise ValueError(f"Committed game schedule mismatch: missing={len(expected_paths-actual_paths)}, extra={len(actual_paths-expected_paths)}")
    manifest_games = []
    for n, entry in enumerate(schedule_rows):
        path = game_path(out, n, entry, args.workers)
        game = torch.load(path, map_location="cpu", weights_only=True)
        behavior.update(verify_game(game, entry, protocol))
        fingerprints[path.relative_to(out).as_posix()] = sha256(path)
        split = entry[0]
        for row in game["rows"]:
            split_id = row[5]
            if split == "validation" and split_id in train_seen:
                dropped["train_overlap"] += 1
            elif split == "validation" and split_id in validation_seen:
                dropped["validation_duplicate"] += 1
            else:
                rows[split].append(row)
                (train_seen if split == "train" else validation_seen).add(split_id)
        manifest_games.append({"game_id": game["game_id"], "split": split, "index": entry[1],
            "seed": entry[2], "raw_labels": len(game["rows"]), "winner": game["winner"], "plies": game["total_plies"]})
    if train_seen & validation_seen:
        raise AssertionError("Global D4 split exclusion failed")
    data = {"format": DEMO_FORMAT, "protocol": protocol, "rules": args.rules,
            "train": _pack_rows(rows["train"]), "validation": _pack_rows(rows["validation"]),
            "validation_d4_overlap_removed": dropped["train_overlap"],
            "validation_internal_duplicates_removed": dropped["validation_duplicate"]}
    atomic_torch(out / "demonstrations.pt", data)
    checked = load_demonstrations(out / "demonstrations.pt", args.rules)
    if len(checked["train"]["actions"]) != len(rows["train"]) or len(checked["validation"]["actions"]) != len(rows["validation"]):
        raise AssertionError("Merged data failed exact sample count check")
    manifest = {"format": DEMO_FORMAT, "sha256": sha256(out / "demonstrations.pt"), "protocol": protocol,
        "train_samples": len(rows["train"]), "validation_samples": len(rows["validation"]),
        "train_games": args.train_games, "validation_games": args.validation_games,
        "committed_games": len(manifest_games), "games": manifest_games,
        "train_seed_range": [args.train_seed, args.train_seed + args.train_games - 1],
        "validation_seed_range": [args.validation_seed, args.validation_seed + args.validation_games - 1],
        "validation_d4_overlap_removed": dropped["train_overlap"],
        "validation_internal_duplicates_removed": dropped["validation_duplicate"],
        "behavior_counts": dict(behavior), "game_file_sha256": fingerprints,
        "audit": "all game identities/seeds/prefix and behavior RNG/teacher legality/observations/masks/D4 IDs/outcomes replayed; load_demonstrations passed",
        "teacher_search_recomputed_in_audit": False}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def validate_args(args):
    if min(args.train_games, args.validation_games, args.simulations) < 1 or not 1 <= args.workers <= 4:
        raise ValueError("Game/simulation budgets must be positive, workers must be 1..4")
    if not 0 <= args.random_opening <= 6 or args.c_puct <= 0:
        raise ValueError("Invalid search or opening settings")
    probabilities = (args.teacher_probability, args.greedy_probability, args.random_probability)
    if any(not 0 <= x <= 1 for x in probabilities) or abs(sum(probabilities) - 1) > 1e-12:
        raise ValueError("Behavior probabilities must be nonnegative and sum to one")
    if args.device == "cuda" and args.workers != 1:
        raise ValueError("Use one worker for a CUDA teacher; parallel workers use CPU")
    schedule(args)


def generate(args, expert_factory=None, expert_metadata=None):
    validate_args(args)
    if expert_factory and (not expert_metadata or args.workers != 1):
        raise ValueError("Injected teachers require explicit provenance and workers=1")
    out = Path(args.output).resolve()
    args.output = str(out)
    args.teacher_checkpoint = str(Path(args.teacher_checkpoint).resolve())
    seed_everything(args.train_seed)
    current = make_protocol(args, expert_metadata)
    if args.worker_index is not None:
        saved = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        if saved != current or not 0 <= args.worker_index < args.workers:
            raise ValueError("Worker protocol/index does not match its parent")
        worker(args, saved, args.worker_index)
        verify_current(saved, args)
        return
    with exclusive_run(out / "run.lock"):
        protocol_path = out / "protocol.json"
        if protocol_path.exists():
            if not args.resume:
                raise ValueError("Existing dataset requires --resume")
            saved = json.loads(protocol_path.read_text(encoding="utf-8"))
            if saved != current:
                raise ValueError("Cannot resume after changing source, teacher or configuration")
        else:
            if any(p.name != "run.lock" for p in out.iterdir()):
                raise ValueError("Use a new empty dataset directory")
            atomic_json(protocol_path, current)
            atomic_json(out / "created.json", {"created_at": datetime.now().astimezone().isoformat()})
            for rel, expected in current["sources"].items():
                target = out / "frozen_source" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / rel).read_bytes())
                if sha256(target) != expected:
                    raise RuntimeError("Source changed while archiving")
        atomic_json(out / "status.json", {"state": "generating", "pid": os.getpid(),
            "target_games": args.train_games + args.validation_games, "workers": args.workers})
        try:
            if args.workers == 1:
                worker(args, current, 0, expert_factory)
            else:
                def launch(index):
                    target = out / "shards" / f"worker_{index:02d}"
                    target.mkdir(parents=True, exist_ok=True)
                    flags = [sys.executable, "-u", "-m", "rl.training.a2c_expert_dataset"]
                    for key, value in vars(args).items():
                        if key in ("resume", "worker_index"):
                            continue
                        flags += ["--" + key.replace("_", "-"), str(value)]
                    flags += ["--worker-index", str(index)]
                    with (target / "console.log").open("a", encoding="utf-8") as stream:
                        process = subprocess.Popen(flags, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        atomic_json(target / "launcher.json", {"pid": process.pid, "command": flags})
                        code = process.wait()
                    if code:
                        raise RuntimeError(f"Dataset worker {index} failed; see {target / 'console.log'}")
                    return index
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(launch, i) for i in range(args.workers)]
                    for future in as_completed(futures):
                        print(f"worker {future.result()} complete", flush=True)
            verify_current(current, args, expert_metadata)
            manifest = merge_games(args, out, current)
            verify_current(current, args, expert_metadata)
            atomic_json(out / "status.json", {"state": "complete", "pid": os.getpid(),
                "games": manifest["committed_games"], "target_games": args.train_games + args.validation_games,
                "train_samples": manifest["train_samples"], "validation_samples": manifest["validation_samples"],
                "dataset_sha256": manifest["sha256"]})
            return manifest
        except BaseException as error:
            atomic_json(out / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "pid": os.getpid(), "error": repr(error), "recovery": "resume unchanged settings; complete per-game commits are retained"})
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    p.add_argument("--simulations", "--sims", dest="simulations", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=1.4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--rules", choices=("freestyle",), default="freestyle")
    p.add_argument("--train-games", type=int, default=512)
    p.add_argument("--validation-games", type=int, default=64)
    p.add_argument("--train-seed", type=int, default=5_000_000)
    p.add_argument("--validation-seed", type=int, default=5_100_000)
    p.add_argument("--random-opening", type=int, default=6)
    p.add_argument("--teacher-probability", type=float, default=.8)
    p.add_argument("--greedy-probability", type=float, default=.1)
    p.add_argument("--random-probability", type=float, default=.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    return p


def main():
    generate(parser().parse_args())


if __name__ == "__main__":
    main()
