"""Independent, read-only replay audit of explicitly named A2C evaluations.

No discovery of test directories, candidate inference, tactics, training, or
MCTS is performed. Default inputs must be frozen final tests. Development
smoke checks require the explicit --allow-development option. Only the new
audit output is written, never an experiment, report, model, or registry.

Manifest format (paths resolve relative to the manifest):
  {"format": "a2c_independent_audit_inputs_v1", "evaluations": [
    {"directory": "...", "training_seed": 11, "recipe": "residual10_ce_a2c",
     "training_run": "...", "checkpoint_sha256": "...",
     "training_protocol_sha256": "...", "frozen_selection": "..."}
  ]}
Only directory is required. Training seed is verified from the actual training
protocol; an explicit seed or hash is an additional assertion, never evidence
by itself. Inferred training runs are checkpoint.parent, or its parent for
candidate checkpoints. Exact paths, protocols and hashes are saved in output.
Contextual baselines require role="contextual_reference" and a recipe label;
their config.json may supply provenance, and Minimax has no training seed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import traceback

sys.dont_write_bytecode = True

FORMAT = "a2c_independent_final_audit_v1"
IDENTITY_FIELDS = (
    "network_kind", "mode", "device", "rules", "opponent", "games",
    "seed_black", "seed_white", "depth", "nodes", "search_candidates",
    "tactical_candidates", "value_weight", "sims",
)
ARTIFACTS = ("protocol.json", "summary.json", "status.json", "games.csv", "moves.csv")
BASE_SOURCES = (
    "rl/agents/a2c.py", "rl/agents/tactics.py", "rl/agents/greedy_agent.py",
    "rl/agents/q_learning.py", "rl/agents/random_agent.py", "rl/environment/rules.py",
    "rl/environment/gomoku.py", "rl/environment/state.py", "rl/training/common.py",
    "rl/training/a2c_math.py", "rl/evaluation/eval_a2c_tactical.py",
    "rl/evaluation/a2c_research_eval.py",
)


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def resolve(path, base):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def within(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def equal(actual, expected, label="value"):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), label + ": keys differ")
        for key, value in actual.items():
            equal(value, expected[key], label + "." + str(key))
    elif isinstance(actual, (tuple, list)):
        require(isinstance(expected, (tuple, list)) and len(actual) == len(expected), label + ": length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, label + f"[{index}]")
    elif isinstance(actual, float):
        require(isinstance(expected, (float, int)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
                f"{label}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{label}: {actual!r} != {expected!r}")


@contextmanager
def existing_unheld_lock(path):
    """Nonmutating OS lock held for the whole read; no stale PID assumptions."""
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, f"Missing/malformed lock: {path}")
    with path.open("rb") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise AuditError(f"Active writer lock: {path}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def wilson(wins, games):
    # Match the frozen experiment's explicitly used 1.96 normal quantile.
    z = 1.96
    fraction = wins / games
    denominator = 1 + z * z / games
    middle = (fraction + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / games + z * z / (4 * games * games)) / denominator
    return [max(0., middle - radius), min(1., middle + radius)]


def check_sources(sources, project_root, required=()):
    require(isinstance(sources, dict) and set(required) <= set(sources), "Missing source hash coverage")
    for relative, expected in sources.items():
        path = resolve(relative, project_root)
        require(within(path, project_root), f"Source escapes source root: {relative}")
        require(path.is_file() and digest(path) == expected, f"Frozen source differs: {relative}")


def frozen_identity(protocol, entry, manifest_base, project_root, allow_development):
    identity = {key: protocol[key] for key in IDENTITY_FIELDS}
    identity.update(checkpoint_sha256=protocol["checkpoint_sha256"],
                    source_sha256=protocol["source_sha256"])
    games = protocol["games"]
    require(type(games) is int and 1 <= games <= 100000, "Invalid per-color game budget")
    black, white = protocol["seed_black"], protocol["seed_white"]
    require(type(black) is int and type(white) is int and abs(black - white) >= games,
            "Color seed ranges overlap")
    if protocol["split"] != "test":
        require(allow_development, "Default audit accepts final tests only; use --allow-development for smoke")
        for seed in (black, white):
            require(seed + games - 1 < 8000000 or seed >= 8500000,
                    "Development overlaps reserved final-test seeds")
        require(protocol.get("frozen_selection") is None, "Development has unexpected test manifest")
        return {"verified": False, "reason": "explicit development smoke", "identity": identity}
    require(games >= 100, "Final tests need at least 100 games per color")
    expected = (8100000, 8200000) if protocol["rules"] == "freestyle" else (8300000, 8400000)
    require((black, white) == expected, "Final test uses unreserved color seeds")
    frozen = protocol.get("frozen_selection")
    require(isinstance(frozen, dict), "Final test lacks recorded frozen-selection provenance")
    path = resolve(frozen["path"], project_root)
    if entry.get("frozen_selection"):
        require(path == resolve(entry["frozen_selection"], manifest_base), "Explicit frozen path differs")
    require(digest(path) == frozen["sha256"], "Frozen selection file changed")
    selected = read_json(path)
    require(selected.get("status") == "frozen" and identity in selected.get("evaluations", []),
            "Complete policy identity is absent from frozen selection")
    equal(frozen.get("identity"), identity, "recorded frozen identity")
    equal(frozen.get("selected_using"), selected.get("selected_using"), "selection criterion")
    context = None
    if identity["checkpoint_sha256"] is not None:
        context = selected.get("models", {}).get(identity["checkpoint_sha256"])
        require(isinstance(context, dict), "Frozen models must index provenance by checkpoint SHA256")
        require({"role", "recipe", "training_seed", "training_run", "training_protocol_sha256"} <= set(context),
                "Frozen model context lacks recipe/seed/training provenance")
    return {"verified": True, "path": str(path), "sha256": frozen["sha256"], "identity": identity,
            "selected_using": selected.get("selected_using"), "model_context": context}


def model_fingerprint(payload, torch):
    state = payload.get("state_dict")
    require(isinstance(state, dict) and bool(state), "Model checkpoint lacks state_dict")
    result, count = hashlib.sha256(), 0
    for name, tensor in sorted(state.items()):
        require(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all()),
                "Invalid/nonfinite model tensor: " + name)
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode()); result.update(str(value.dtype).encode())
        result.update(json.dumps(list(value.shape)).encode())
        result.update(value.view(torch.uint8).numpy().tobytes()); count += value.numel()
    return result.hexdigest(), count


def training_provenance(checkpoint, payload, entry, manifest_base, project_root, is_final, stack):
    import torch
    inferred = checkpoint.parent.parent if checkpoint.parent.name == "candidates" else checkpoint.parent
    run = resolve(entry.get("training_run", str(inferred)), manifest_base)
    require(within(checkpoint, run), "Checkpoint is outside its asserted training run")
    role = entry.get("role", "candidate")
    path = run / "protocol.json"
    legacy = not path.exists()
    if legacy:
        require(role == "contextual_reference", "Research candidate lacks training protocol.json")
        path = run / "config.json"
    training = read_json(path)
    config = training if legacy else training.get("config", {})
    if not legacy: check_sources(training.get("sources", {}), project_root)
    seed = config.get("seed")
    require(type(seed) is int, "Actual training protocol has no integer seed")
    if "training_seed" in entry:
        require(type(entry["training_seed"]) is int and entry["training_seed"] == seed,
                "Claimed seed differs from actual training protocol")
    protocol_sha = digest(path)
    if "training_protocol_sha256" in entry:
        require(entry["training_protocol_sha256"] == protocol_sha, "Training protocol hash mismatch")
    if is_final:
        if (run / "run.lock").exists():
            stack.enter_context(existing_unheld_lock(run / "run.lock"))
        else:
            require(role == "contextual_reference", "Final candidate training lacks OS lock artifact")
        require(read_json(run / "status.json").get("state") == "complete", "Selected training run is unfinished")
    if payload.get("spec") is not None:
        equal(normalized(payload["spec"]), normalized(training.get("spec")), "model/training spec")
    require(payload.get("rules", "freestyle") == config.get("rules", "freestyle"), "Model training rules disagree")
    weights_sha, parameters = model_fingerprint(payload, torch)
    companion = checkpoint.with_suffix(".checkpoint")
    companion_sha = None
    if checkpoint.parent.name == "candidates":
        require(companion.is_file(), "Immutable candidate lacks complete companion checkpoint")
        companion_sha = digest(companion)
        # These local research checkpoints contain Python/NumPy RNG state and
        # therefore require trusted local pickle, unlike the weights-only file.
        full = torch.load(companion, map_location="cpu", weights_only=False)
        require(full["protocol"]["config"]["seed"] == seed, "Companion training seed mismatch")
        equal(full["protocol"]["sources"], training["sources"], "companion training source hashes")
        equal(normalized(full["protocol"]["spec"]), normalized(training["spec"]), "companion spec")
        for key, value in payload["state_dict"].items():
            require(key in full["net"] and torch.equal(value.cpu(), full["net"][key].cpu()), "Model/companion weights differ")
        require(set(payload["state_dict"]) == set(full["net"]), "Model/companion state keys differ")
        require(payload.get("episode") == full.get("episode"), "Candidate episode mismatch")
    excluded = {"output", "resume", "seed", "initial_model", "dataset"}
    initial = None
    if config.get("initial_model"):
        initial_path = resolve(config["initial_model"], project_root)
        require(digest(initial_path) == training.get("initial_sha256"), "Demonstration/warm-start source changed")
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        initial_fingerprint, _ = model_fingerprint(initial_payload, torch)
        initial_protocol_path = initial_path.parent / "protocol.json"
        initial = {"path": str(initial_path), "sha256": training["initial_sha256"],
            "stage": initial_payload.get("stage"), "epoch": initial_payload.get("epoch"),
            "state_sha256": initial_fingerprint, "selected_state_equals_initial": weights_sha == initial_fingerprint,
            "pretraining_protocol_sha256": digest(initial_protocol_path) if initial_protocol_path.exists() else None}
        if initial_protocol_path.exists():
            origin_protocol = read_json(initial_protocol_path)
            initial["pretraining_seed"] = origin_protocol.get("config", {}).get("seed")
            initial["same_seed_as_A2C_run"] = initial["pretraining_seed"] == seed
    episode, epoch = payload.get("episode"), payload.get("epoch")
    if episode == 0:
        require(initial and initial["selected_state_equals_initial"], "Episode-zero candidate is not its recorded initial state")
        selected_stage = "demonstration_pretraining_only" if initial.get("stage") == "imitation" else "warm_start_only"
    elif config.get("command") == "pretrain" or (epoch is not None and episode is None):
        selected_stage = "demonstration_pretraining_only"
    elif episode is not None:
        selected_stage = "after_A2C_updates"
    else:
        selected_stage = "legacy_reference_selection_metadata_unavailable"
    recipe = {"config": {k: v for k, v in config.items() if k not in excluded},
              "spec": training.get("spec"), "sources": training.get("sources"),
              "dataset_sha256": training.get("dataset_sha256"),
              "algorithm": training.get("algorithm"), "precision": training.get("precision"),
              "opponent_episode_probabilities": training.get("opponent_episode_probabilities")}
    model_spec = payload.get("spec")
    if model_spec is None:
        model_spec = {k: payload.get(k, config.get(k)) for k in
                      ("board_size", "channels", "input_channels", "hidden_units")}
    return {"run": str(run), "training_seed": seed, "training_protocol_path": str(path),
            "training_protocol_sha256": protocol_sha, "legacy_config_provenance": legacy,
            "training_os_lock_present": (run / "run.lock").is_file(),
            "training_completion_verified": bool(is_final),
            "checkpoint_state_sha256": weights_sha, "parameter_count": parameters,
            "complete_candidate_checkpoint_path": str(companion) if companion_sha else None,
            "complete_candidate_checkpoint_sha256": companion_sha,
            "model_spec": normalized(model_spec), "model_variant": payload.get("model_variant"),
            "selected_episode": episode, "selected_epoch": epoch,
            "selected_checkpoint_stage": selected_stage,
            "selected_candidate_has_A2C_updates": episode is not None and episode > 0,
            "demonstration_or_warm_start_origin": initial,
            "training_rules": payload.get("rules", "freestyle"), "recipe_signature": stable_hash(recipe),
            "recipe": entry.get("recipe", payload.get("model_variant", "unspecified_recipe")),
            "initial_model": config.get("initial_model"), "initial_sha256": training.get("initial_sha256"),
            "dataset_sha256": training.get("dataset_sha256"),
            "limitations": ["Protocol seed verifies lineage; checkpoint bytes do not prove independent initialization.",
                            "Common expert data and any shared warm-start must be disclosed."]}


def deterministic_reference(protocol, entry):
    require(entry.get("role") == "contextual_reference" and entry.get("recipe"),
            "A checkpoint-free reference needs explicit contextual_reference role and recipe")
    require(protocol["mode"] == "minimax" and protocol.get("checkpoint_sha256") is None,
            "Only Minimax may be a checkpoint-free reference")
    return {"run": None, "training_seed": None, "training_protocol_path": None,
        "training_protocol_sha256": None, "checkpoint_state_sha256": None, "parameter_count": 0,
        "model_spec": {"algorithm": "minimax"}, "model_variant": "minimax", "selected_episode": None,
        "selected_epoch": None, "selected_checkpoint_stage": "deterministic_reference_no_training",
        "selected_candidate_has_A2C_updates": False, "training_rules": "freestyle",
        "recipe_signature": stable_hash({"algorithm": "minimax"}), "recipe": entry["recipe"],
        "dataset_sha256": None, "initial_model": None, "initial_sha256": None,
        "limitations": ["Deterministic contextual reference; no training-seed replication."]}


def replay_game(moves, color, seed, rules, opponent_name, modules):
    Env, Greedy, Random, choose = modules
    opponent = (Greedy if opponent_name == "greedy" else Random)(seed=seed)
    env = Env(rules=rules)
    times, reasons, nodes = [], Counter(), []
    opponent_actions = 0
    for ply, row in enumerate(moves, 1):
        require(not env.done, "Moves continue after terminal position")
        player, action = env.current_player, int(row["action"])
        is_candidate = player == color
        require(int(row["ply"]) == ply and int(row["seed"]) == seed
                and int(row["player"]) == player and int(row["color"]) == color
                and row["rules"] == rules, "Move identity/turn mismatch")
        require(row["actor"] == ("candidate" if is_candidate else opponent_name), "Actor mismatch")
        require(action in env.get_legal_actions(), "Illegal recorded move")
        duration = float(row["inference_seconds"])
        require(math.isfinite(duration) and duration >= 0, "Invalid inference duration")
        if is_candidate:
            times.append(duration); reasons[row["tactical_reason"]] += 1
            count = int(row["search_nodes"]); require(count >= 0, "Negative search nodes"); nodes.append(count)
        else:
            require(int(choose(opponent, env)) == action, f"Seeded {opponent_name} action differs at ply {ply}")
            opponent_actions += 1
        env.step(action)
    require(env.done, "Incomplete game trajectory")
    outcome = 0 if env.winner == 0 else (1 if env.winner == color else -1)
    return outcome, times, reasons, nodes, opponent_actions


def audit_one(entry, base, project_root, allow_development, modules):
    import torch
    directory = resolve(entry["directory"], base)
    with ExitStack() as stack:
        stack.enter_context(existing_unheld_lock(directory / "eval.lock"))
        # Read only protocol first: forbidden non-test 8m inputs are rejected
        # before opening/hashing any game/move/result artifact.
        protocol = read_json(directory / "protocol.json")
        frozen = frozen_identity(protocol, entry, base, project_root, allow_development)
        if frozen.get("model_context"):
            entry = dict(entry)
            for key, value in frozen["model_context"].items():
                if key not in ("role", "recipe", "training_seed", "training_run", "training_protocol_sha256"):
                    continue
                if key in entry:
                    if key == "training_run":
                        require(resolve(entry[key], base) == resolve(value, project_root), "Input/frozen training run differs")
                    else:
                        equal(entry[key], value, "input/frozen context." + key)
                else:
                    entry[key] = str(resolve(value, project_root)) if key == "training_run" else value
        before = {name: digest(directory / name) for name in ARTIFACTS}
        summary, status = (read_json(directory / name) for name in ARTIFACTS[1:3])
        require(status.get("state") == "complete", "Evaluation is unfinished")
        equal(summary["protocol"], protocol, "summary/protocol")
        require(protocol["opponent"] in ("greedy", "random"), "Unsupported opponent")
        require(protocol["rules"] in ("freestyle", "renju"), "Unsupported rules")
        require(protocol["network_kind"] in ("standard", "research"), "Unsupported network kind")
        require(protocol["mode"] in ("policy", "guarded", "tactical", "hand", "search", "minimax", "alphazero"),
                "Unsupported wrapper/algorithm")
        require(protocol.get("training_performed") is False and protocol.get("selection_performed") is False,
                "Evaluation claims training or selection")
        require(summary.get("sources_and_weights_unchanged") is True, "Evaluation did not confirm source guards")
        required = [*BASE_SOURCES]
        if protocol["network_kind"] == "research": required.append("rl/agents/a2c_research.py")
        if protocol["mode"] == "search": required.append("rl/agents/a2c_search.py")
        if protocol["mode"] == "hand":
            required.extend(("rl/evaluation/a2c_candidate_study.py", "rl/agents/a2c_tactical.py"))
        if protocol["mode"] in ("minimax", "alphazero"):
            required.append("rl/agents/" + ("minimax_agent.py" if protocol["mode"] == "minimax" else "alphazero.py"))
        if protocol["mode"] == "guarded": required.extend(("rl/agents/a2c_guarded.py", "rl/evaluation/a2c_research_eval_v2.py"))
        if any(name in protocol["source_sha256"] for name in ("rl/agents/a2c_guarded.py", "rl/evaluation/a2c_research_eval_v2.py")):
            required.extend(("rl/agents/a2c_guarded.py", "rl/evaluation/a2c_research_eval_v2.py"))
        check_sources(protocol["source_sha256"], project_root, required)
        require(entry.get("role", "candidate") in ("candidate", "contextual_reference"), "Unknown replicate role")
        if entry.get("role") == "contextual_reference": require(bool(entry.get("recipe")), "Contextual reference needs distinct recipe label")
        checkpoint, checkpoint_sha = None, None
        if protocol.get("checkpoint"):
            checkpoint = resolve(protocol["checkpoint"], project_root)
            checkpoint_sha = digest(checkpoint)
            require(checkpoint_sha == protocol["checkpoint_sha256"], "Checkpoint SHA differs from protocol")
            if entry.get("checkpoint_sha256"):
                require(entry["checkpoint_sha256"] == checkpoint_sha, "Explicit checkpoint hash differs")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            training = training_provenance(checkpoint, payload, entry, base, project_root,
                                           protocol["split"] == "test", stack)
        else:
            training = deterministic_reference(protocol, entry)
        if frozen.get("model_context"):
            context = frozen["model_context"]
            for key in ("training_seed", "training_protocol_sha256", "recipe"):
                equal(training[key], context[key], "actual/frozen model context." + key)
            for key in ("model_spec", "selected_episode", "selected_epoch"):
                if key in context and (training[key] is not None or key == "model_spec"):
                    equal(normalized(training[key]), normalized(context[key]), "actual/frozen model context." + key)
        require(protocol.get("training_rules", "freestyle") == training["training_rules"], "Recorded training rules differ")
        require(protocol.get("rule_transfer", False) == (training["training_rules"] != protocol["rules"]), "Rule-transfer flag differs")
        expected_tactics = {"enabled": True, "forcing_depth": 5, "max_nodes": 1500,
            "candidate_limit": protocol["tactical_candidates"] if protocol["mode"] == "search" else 12,
            "cache_size": 4096}
        equal(protocol["tactical_config"], expected_tactics, "effective tactical defaults")
        with (directory / "games.csv").open(newline="", encoding="utf-8-sig") as stream:
            games = list(csv.DictReader(stream))
        with (directory / "moves.csv").open(newline="", encoding="utf-8-sig") as stream:
            moves = list(csv.DictReader(stream))
        budget = protocol["games"]
        expected_ids = [(color, index) for color in (1, -1) for index in range(budget)]
        require([(int(g["color"]), int(g["game"])) for g in games] == expected_ids, "Missing/duplicate/unordered game IDs")
        grouped = defaultdict(list)
        for row in moves: grouped[(int(row["color"]), int(row["game"]))].append(row)
        require(set(grouped) == set(expected_ids), "Move/game identities differ")
        results, opponent_actions = [], 0
        for color, seed_base in ((1, protocol["seed_black"]), (-1, protocol["seed_white"])):
            outcomes, lengths, durations, reasons, nodes = Counter(), [], [], Counter(), []
            for index in range(budget):
                game = games[index + (0 if color == 1 else budget)]
                seed, trajectory = seed_base + index, grouped[(color, index)]
                require(int(game["seed"]) == seed and game["rules"] == protocol["rules"]
                        and game["split"] == protocol["split"] and game["mode"] == protocol["mode"]
                        and game["opponent"] == protocol["opponent"], "Game protocol identity differs")
                outcome, times, reason, count, reproduced = replay_game(trajectory, color, seed,
                    protocol["rules"], protocol["opponent"], modules)
                require(int(game["outcome"]) == outcome and int(game["moves"]) == len(trajectory), "Game outcome/length differs")
                equal(sum(times), float(game["policy_seconds"]), "summed game inference")
                outcomes[outcome] += 1; lengths.append(len(trajectory)); durations.extend(times)
                reasons.update(reason); nodes.extend(count); opponent_actions += reproduced
            ordered = sorted(durations)
            require(bool(ordered), "Missing candidate inference samples")
            results.append({"color": color, "games": budget, "wins": outcomes[1], "draws": outcomes[0],
                "losses": outcomes[-1], "win_rate": outcomes[1] / budget,
                "score": (outcomes[1] + .5 * outcomes[0]) / budget, "wilson_95": wilson(outcomes[1], budget),
                "mean_game_plies": statistics.mean(lengths), "inference_samples": len(durations),
                "mean_move_ms": 1000 * statistics.mean(durations), "median_move_ms": 1000 * statistics.median(durations),
                "p95_move_ms": 1000 * ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
                "max_move_ms": 1000 * ordered[-1], "tactical_reasons": dict(reasons),
                "mean_search_nodes": statistics.mean(nodes)})
        equal(results, summary["results"], "recalculated summary")
        require(summary["total_games"] == 2 * budget, "Total game count differs")
        mean_score = statistics.mean(row["score"] for row in results)
        mean_win = statistics.mean(row["win_rate"] for row in results)
        equal(mean_score, summary["mean_score"], "macro score")
        equal(mean_win, summary["mean_win_rate"], "macro win rate")
        require(before == {name: digest(directory / name) for name in ARTIFACTS}, "Artifacts changed during audit")
        if checkpoint: require(digest(checkpoint) == checkpoint_sha, "Checkpoint changed during audit")
        if training["training_protocol_path"]:
            require(digest(training["training_protocol_path"]) == training["training_protocol_sha256"], "Training provenance changed")
        if training.get("complete_candidate_checkpoint_path"):
            require(digest(training["complete_candidate_checkpoint_path"]) == training["complete_candidate_checkpoint_sha256"],
                    "Complete candidate companion changed during audit")
        if training.get("demonstration_or_warm_start_origin"):
            origin = training["demonstration_or_warm_start_origin"]
            require(digest(origin["path"]) == origin["sha256"], "Initial demonstration/warm-start source changed during audit")
        check_sources(protocol["source_sha256"], project_root, required)
        if frozen["verified"]: require(digest(frozen["path"]) == frozen["sha256"], "Frozen manifest changed during audit")
        return {"directory": str(directory), "passed": True, "frozen_selection": frozen,
            "role": entry.get("role", "candidate"),
            "training": training, "games": 2 * budget, "legal_moves_replayed": len(moves),
            "opponent_actions_exactly_reproduced": opponent_actions, "artifact_sha256": before,
            "checkpoint_sha256": checkpoint_sha, "identity": frozen["identity"], "split": protocol["split"],
            "results": results, "mean_score": mean_score, "mean_win_rate": mean_win,
            "candidate_actions_recomputed": False}


def seed_statistics(values):
    return {"mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values) >= 2 else None}


def aggregate(items):
    groups = defaultdict(list)
    for item in items:
        identity, training = item["identity"], item["training"]
        # Neither wrapper, color, opponent, rules, model, training recipe, device,
        # split nor search budget may silently be pooled across experiments.
        key = {"role": item["role"], "recipe": training["recipe"], "recipe_signature": training["recipe_signature"],
            "model_spec": training["model_spec"], "split": item["split"],
            **{k: identity[k] for k in IDENTITY_FIELDS if k not in ("seed_black", "seed_white")}}
        groups[json.dumps(key, sort_keys=True)].append(item)
    result = []
    for key, evaluations in groups.items():
        seeds, fingerprints = set(), {}
        for item in evaluations:
            training = item["training"]; seed = training["training_seed"]
            require(seed not in seeds, "Repeated evaluation of the same training seed cannot be a replicate")
            seeds.add(seed)
            fingerprint = training["checkpoint_state_sha256"]
            require(fingerprint not in fingerprints, "Identical model states across claimed different training seeds")
            fingerprints[fingerprint] = seed
        if json.loads(key)["role"] == "contextual_reference":
            require(len(evaluations) == 1, "Contextual reference group must stay a single explicitly labelled measurement")
        actual_seeds = sorted(seed for seed in seeds if seed is not None)
        schedules = sorted({(v["identity"]["seed_black"], v["identity"]["seed_white"], v["identity"]["games"]) for v in evaluations})
        rows = []
        for color in (1, -1, "mean_black_white"):
            metrics = {name: [] for name in ("win_rate", "score", "mean_move_ms", "mean_game_plies")}
            per_seed = []
            for item in sorted(evaluations, key=lambda x: x["training"]["training_seed"] if x["training"]["training_seed"] is not None else -1):
                selected = item["results"] if color == "mean_black_white" else [r for r in item["results"] if r["color"] == color]
                row = {name: statistics.mean(v[name] for v in selected) for name in metrics}
                if color != "mean_black_white":
                    row.update({name: selected[0][name] for name in ("games", "wins", "draws", "losses", "wilson_95")})
                per_seed.append({"training_seed": item["training"]["training_seed"], **row})
                for name in metrics: metrics[name].append(row[name])
            rows.append({"color": color, "training_seeds": actual_seeds, "n_training_seeds": len(actual_seeds),
                "metrics": {name: seed_statistics(values) for name, values in metrics.items()}, "per_seed": per_seed})
        result.append({"group": json.loads(key), "n_training_seeds": len(actual_seeds), "rows": rows,
            "evaluation_seed_schedules": schedules,
            "paired_game_seed_reuse_across_training_seeds": len(evaluations) > 1 and len(schedules) == 1,
            "shared_dataset_sha256": sorted({v["training"]["dataset_sha256"] for v in evaluations if v["training"]["dataset_sha256"]}),
            "inference": "Across actual training seeds, not repeated games/checkpoints; n=1 sample SD is null. Reused opponent/game seeds are paired; no pooled independent-game confidence interval is computed."})
    return result


def random_replay_self_test(modules):
    """Synthetic nonreserved trajectories, including a legal RNG tamper test."""
    Env, _, Random, choose = modules
    counts, tested_tamper = 0, False
    for color, seed in ((1, 6000000), (-1, 6100000)):
        env, opponent = Env(rules="freestyle"), Random(seed=seed)
        candidate_rng, moves = random.Random(seed + 17), []
        while not env.done:
            player = env.current_player; is_candidate = player == color
            action = candidate_rng.choice(env.get_legal_actions()) if is_candidate else choose(opponent, env)
            moves.append({"color": color, "seed": seed, "ply": len(moves) + 1, "player": player,
                "rules": "freestyle", "actor": "candidate" if is_candidate else "random", "action": action,
                "inference_seconds": 0., "tactical_reason": "", "search_nodes": 0})
            env.step(action)
        replayed = replay_game(moves, color, seed, "freestyle", "random", modules)
        require(replayed[0] == (0 if env.winner == 0 else 1 if env.winner == color else -1), "Random synthetic outcome differs")
        counts += replayed[-1]
        bad = [dict(row) for row in moves]
        first = next(i for i, row in enumerate(bad) if row["actor"] == "random")
        check = Env(rules="freestyle")
        for row in bad[:first]: check.step(row["action"])
        bad[first]["action"] = next(a for a in check.get_legal_actions() if a != bad[first]["action"])
        try: replay_game(bad, color, seed, "freestyle", "random", modules)
        except AuditError as error:
            require("Seeded random action differs" in str(error), "Random tamper failed for another reason")
            tested_tamper = True
        else: raise AuditError("Seeded Random tamper was not detected")
    return {"passed": True, "synthetic_games": 2, "reserved_test_results_read": False,
            "random_actions_reproduced": counts, "legal_action_rng_tamper_rejected": tested_tamper}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", required=True)
    p.add_argument("--directory", action="append", default=[], help="Exact evaluation directory; repeat explicitly")
    p.add_argument("--manifest", help="Explicit input manifest; no directory discovery")
    p.add_argument("--output", required=True, help="New audit JSON; existing files are never overwritten")
    p.add_argument("--allow-development", action="store_true", help="Audit nonreserved development cases for smoke only")
    p.add_argument("--random-self-test", action="store_true", help="Also verify tiny synthetic nonreserved Random trajectories")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    project_root, output = Path(args.project_root).resolve(), Path(args.output).resolve()
    require(project_root.is_dir(), "Missing project root")
    require(not output.exists(), "Audit output already exists; use a new file")
    entries = [(dict(directory=path), Path.cwd()) for path in args.directory]
    input_sha = None
    if args.manifest:
        manifest_path = Path(args.manifest).resolve(); manifest = read_json(manifest_path)
        require(manifest.get("format") == "a2c_independent_audit_inputs_v1", "Unsupported input manifest format")
        require(isinstance(manifest.get("evaluations"), list), "Input manifest lacks explicit evaluations")
        entries += [(entry, manifest_path.parent) for entry in manifest["evaluations"]]
        input_sha = digest(manifest_path)
    require(bool(entries) or args.random_self_test, "No explicit inputs or Random self-test")
    directories = [resolve(entry["directory"], base) for entry, base in entries]
    require(len(directories) == len(set(directories)), "Duplicate input directory")
    require(all(not within(output, directory) for directory in directories), "Audit output must not alter an input evaluation directory")
    # No policy constructors, evaluator.run, tactics.analyze or search methods
    # are imported/called. Only the environment and fixed opponents are used.
    sys.path.insert(0, str(project_root))
    import torch
    torch.set_num_threads(1)
    from rl.environment.gomoku import GomokuEnv
    from rl.agents.greedy_agent import GreedyAgent
    from rl.agents.random_agent import RandomAgent
    from rl.training.common import choose
    modules = GomokuEnv, GreedyAgent, RandomAgent, choose
    loaded_sources = {}
    for name in ("rl.environment.gomoku", "rl.environment.rules", "rl.environment.state",
                 "rl.agents.greedy_agent", "rl.agents.q_learning", "rl.agents.random_agent",
                 "rl.agents.base_agent", "rl.training.common"):
        module = sys.modules.get(name)
        if module is None:
            continue
        actual = Path(module.__file__).resolve()
        expected = project_root / (name.replace(".", "/") + ".py")
        require(actual == expected.resolve(), f"Imported module escaped requested source root: {name}")
        loaded_sources[name] = str(actual)
    report = {"format": FORMAT, "started_at": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": digest(__file__), "project_root": str(project_root), "input_manifest_sha256": input_sha,
        "loaded_replay_modules": loaded_sources,
        "allow_development": args.allow_development, "candidate_inference_recomputed": False,
        "training_or_test_execution_performed": False, "evaluations": [], "errors": [],
        "limitations": ["This audit checks recorded candidate moves for legality, not exact neural/search recomputation.",
            "Per-policy Wilson95 describes game win rates only, not training-seed uncertainty or score.",
            "Sample standard deviations use actual training seeds and ddof=1; shared expert data/warm starts remain dependent.",
            "Reused opponent/game seeds across models, wrappers or training seeds are paired; pooled 300-game independence is not assumed.",
            "Episode-zero selections retain demonstration/warm-start provenance and are never labelled A2C training improvement.",
            "Finite Greedy/Random tests do not establish general playing strength.",
            "Sources must match --project-root; mismatches are refused rather than silently importing new rules.",
            "Source verification covers the evaluator's declared hash contract, not the entire Python dependency closure.",
            "Complete local candidate companions use trusted-local torch pickle to verify training lineage."]}
    try:
        if args.random_self_test: report["random_replay_self_test"] = random_replay_self_test(modules)
        for entry, base in entries:
            item = audit_one(entry, base, project_root, args.allow_development, modules)
            report["evaluations"].append(item)
            print(json.dumps({"audited": item["directory"], "games": item["games"],
                "training_seed": item["training"]["training_seed"], "score": item["mean_score"]}), flush=True)
        report["cross_training_seed_statistics"] = aggregate(report["evaluations"])
        if args.manifest: require(digest(manifest_path) == input_sha, "Input manifest changed during audit")
        report["total_games"] = sum(x["games"] for x in report["evaluations"])
        report["legal_moves_replayed"] = sum(x["legal_moves_replayed"] for x in report["evaluations"])
        report["opponent_actions_exactly_reproduced"] = sum(x["opponent_actions_exactly_reproduced"] for x in report["evaluations"])
        report["passed"] = True
    except Exception as error:
        report["passed"] = False; report["errors"].append(repr(error)); report["traceback"] = traceback.format_exc()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"passed": report["passed"], "audit_output": str(output), "errors": report["errors"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as failure:
        print(f"Audit refused before processing: {failure}", file=sys.stderr)
        sys.exit(2)
