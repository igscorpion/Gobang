"""Execute an explicitly frozen final matrix; never select or freeze models.

Only a supplied frozen manifest can authorize jobs. --plan-only validates it
without creating output or running games. There is no automatic development
selection, model discovery, test-budget reduction, or policy tuning here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

from rl.evaluation import a2c_research_eval_v2 as evaluator
from rl.evaluation.eval_a2c_tactical import wilson
from rl.training.common import atomic_json as original_atomic_json, exclusive_run
from rl.training.a2c_research_suite import process_identity, same_process

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "a2c_explicit_frozen_final_queue_v1"
IDENTITY_FIELDS = ("network_kind", "mode", "device", "rules", "opponent", "games",
    "seed_black", "seed_white", "depth", "nodes", "search_candidates", "tactical_candidates",
    "value_weight", "sims", "checkpoint_sha256", "source_sha256")
CONTEXT_FIELDS = ("role", "recipe", "training_seed", "training_run",
                  "training_protocol_sha256", "checkpoint_sha256")
SELF = "rl/evaluation/a2c_research_final_suite.py"


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def shared_bytes(path):
    """Request Windows delete sharing; keep active JSON handle lifetimes short."""
    path = Path(path)
    if os.name != "nt":
        return path.read_bytes()
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    name = str(path.resolve())
    if not name.startswith("\\\\?\\"):
        name = "\\\\?\\" + name
    handle = kernel.CreateFileW(name, 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def read_json(path, default=None):
    try:
        return json.loads(shared_bytes(path).decode("utf-8-sig"))
    except FileNotFoundError:
        return {} if default is None else default


def atomic_json(path, value):
    """Bounded retry of this module's own control writes; core stays frozen."""
    for attempt in range(8):
        try:
            original_atomic_json(path, value)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) != 5 or attempt == 7:
                raise
            time.sleep(min(.01 * 2**attempt, .4))


@contextmanager
def unheld_lock(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1:
        raise RuntimeError(f"Missing/malformed OS lock: {path}")
    with path.open("rb") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Active writer: {path}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def lock_held(path):
    if not Path(path).exists():
        return False
    try:
        with unheld_lock(path):
            return False
    except RuntimeError as error:
        if isinstance(error.__cause__, OSError):
            return True
        raise


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checksum(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def job_command(job, identity, manifest, output):
    command = [sys.executable, "-u", "-m", "rl.evaluation.a2c_research_eval_v2"]
    if job["checkpoint"] is not None:
        command += ["--checkpoint", str(resolve(job["checkpoint"]))]
    for key in IDENTITY_FIELDS[:-2]:
        command += ["--" + key.replace("_", "-"), str(identity[key])]
    command += ["--split", "test", "--frozen-selection", str(Path(manifest).resolve()),
                "--output", str(output)]
    return command


def parsed_job(job, identity, manifest):
    return evaluator.parser().parse_args(job_command(job, identity, manifest, "unused")[4:])


def verify_sources(values):
    require(isinstance(values, dict) and bool(values), "Missing frozen source coverage")
    for relative, expected in values.items():
        path = resolve(relative)
        require(path.is_relative_to(ROOT) and checksum(expected) and path.is_file()
                and digest(path) == expected, f"Frozen source changed: {relative}")


def verify_provenance(job, identity, model):
    """Validate actual local lineage, not directory names or asserted seed alone."""
    import torch
    context = job["context"]
    require(set(CONTEXT_FIELDS) <= set(context), "Job context lacks required provenance fields")
    require(not ({"directory", "frozen_selection"} & set(context)), "Context overrides audit output identity")
    require(context["role"] in ("candidate", "contextual_reference")
            and isinstance(context["recipe"], str) and bool(context["recipe"]), "Missing role/recipe")
    require(context["checkpoint_sha256"] == identity["checkpoint_sha256"], "Context checkpoint hash mismatch")
    if identity["mode"] == "minimax":
        require(job["checkpoint"] is None and model is None and identity["checkpoint_sha256"] is None,
                "Minimax cannot carry model weights")
        require(context["role"] == "contextual_reference" and all(context[k] is None for k in
            ("training_seed", "training_run", "training_protocol_sha256")), "Minimax must not invent training provenance")
        return {"checkpoint": None, "context": context, "training_completion_verified": True}
    require(isinstance(model, dict), "Frozen models lacks checkpoint provenance")
    require(job["checkpoint"] is not None and type(context["training_seed"]) is int,
            "A model needs a checkpoint and actual integer training seed")
    for key in set(context) - {"checkpoint_sha256"}:
        if key != "training_run":
            require(key in model and normalized(context[key]) == normalized(model[key]),
                    f"Job context differs from frozen model: {key}")
    for key in CONTEXT_FIELDS[:-1]:
        require(key in model, f"Frozen model lacks {key}")
        if key == "training_run":
            require(context[key] and model[key] and resolve(context[key]) == resolve(model[key]), "Job/model training_run mismatch")
        else:
            require(context[key] == model[key], f"Job/model context mismatch: {key}")
    checkpoint = resolve(job["checkpoint"])
    require(checksum(identity["checkpoint_sha256"]) and checkpoint.is_file()
            and digest(checkpoint) == identity["checkpoint_sha256"], "Checkpoint SHA mismatch")
    run = resolve(context["training_run"])
    require(checkpoint.is_relative_to(run), "Checkpoint is outside its training run")
    protocol_path = run / "protocol.json"
    legacy = not protocol_path.exists()
    if legacy:
        require(context["role"] == "contextual_reference", "Candidate lacks training protocol")
        protocol_path = run / "config.json"
    require(protocol_path.is_file() and checksum(context["training_protocol_sha256"])
            and digest(protocol_path) == context["training_protocol_sha256"], "Training provenance SHA mismatch")
    training = read_json(protocol_path)
    config = training if legacy else training.get("config", {})
    require(type(config.get("seed")) is int and config["seed"] == context["training_seed"], "Claimed training seed is false")
    require(read_json(run / "status.json").get("state") == "complete", "Selected training run is unfinished")
    if (run / "run.lock").exists():
        require(not lock_held(run / "run.lock"), "Selected training writer is still active")
    else:
        require(context["role"] == "contextual_reference", "Candidate lacks OS run.lock")
    if not legacy:
        verify_sources(training.get("sources"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("spec") is not None:
        require(normalized(payload["spec"]) == normalized(training.get("spec")), "Model/training full spec mismatch")
    require(payload.get("rules", "freestyle") == config.get("rules", "freestyle"), "Training rules mismatch")
    for key in ("model_spec", "selected_episode", "selected_epoch"):
        if key in model:
            actual = payload.get("spec") if key == "model_spec" else payload.get(key.removeprefix("selected_"))
            if key == "model_spec" and actual is None:
                actual = {k: payload.get(k, config.get(k)) for k in ("board_size", "channels", "input_channels", "hidden_units")}
            require(normalized(model[key]) == normalized(actual), f"Frozen {key} differs from actual model")
    state = payload.get("state_dict", payload.get("net"))
    require(isinstance(state, dict) and bool(state), "Checkpoint lacks network weights")
    require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values()),
            "Checkpoint has invalid/nonfinite weights")
    companion_sha, companion_path = None, None
    if checkpoint.parent.name == "candidates":
        companion = checkpoint.with_suffix(".checkpoint")
        require(companion.is_file(), "Immutable candidate lacks full checkpoint")
        complete = torch.load(companion, map_location="cpu", weights_only=False)
        require(complete["protocol"]["config"]["seed"] == context["training_seed"]
                and complete["protocol"]["sources"] == training["sources"]
                and normalized(complete["protocol"]["spec"]) == normalized(training["spec"]), "Companion provenance mismatch")
        require(payload.get("episode") == complete.get("episode") and set(state) == set(complete["net"])
                and all(torch.equal(value, complete["net"][key]) for key, value in state.items()), "Companion weights/episode mismatch")
        companion_sha = digest(companion)
        companion_path = str(companion)
    lineage = {}
    for argument, field in (("initial_model", "initial_sha256"), ("dataset", "dataset_sha256")):
        if config.get(argument):
            path = resolve(config[argument])
            require(path.is_file() and digest(path) == training.get(field), f"Changed training {argument}")
            lineage[argument] = {"path": str(path), "sha256": training[field]}
    if payload.get("episode") == 0:
        require("initial_model" in lineage, "Episode-zero candidate has no recorded CE/warm-start origin")
        initial = torch.load(lineage["initial_model"]["path"], map_location="cpu", weights_only=True)
        require(set(initial["state_dict"]) == set(state) and all(torch.equal(state[k], initial["state_dict"][k]) for k in state),
                "Episode-zero candidate differs from its initial weights")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": identity["checkpoint_sha256"],
        "context": context, "training_protocol": str(protocol_path), "legacy_config": legacy,
        "companion_sha256": companion_sha, "companion_path": companion_path, "lineage": lineage,
        "training_sources": training.get("sources", {}) if not legacy else {},
        "training_completion_verified": True,
        "selected_has_A2C_updates": None if payload.get("episode") is None else payload["episode"] > 0}


def validate_manifest(path):
    path = Path(path).resolve()
    require(path.is_file(), "Explicit frozen-selection file is required")
    manifest = read_json(path)
    require(manifest.get("status") == "frozen" and manifest.get("selected_using") == "development only",
            "Manifest must be frozen and selected_using must equal 'development only'")
    require(manifest.get("expected_runner_sha256") == digest(ROOT / SELF), "Frozen runner SHA mismatch")
    evaluations, jobs, models = (manifest.get(key) for key in ("evaluations", "jobs", "models"))
    require(isinstance(evaluations, list) and evaluations and isinstance(jobs, list)
            and len(jobs) == len(evaluations) and isinstance(models, dict), "Missing complete jobs/evaluations/models")
    require(all(isinstance(row, dict) and set(row) == set(IDENTITY_FIELDS) for row in evaluations), "Each identity must have exactly the original 16 fields")
    require(len({stable(row) for row in evaluations}) == len(evaluations), "Duplicate evaluation identity")
    require(all(isinstance(job, dict) and set(job) == {"name", "checkpoint", "identity_index", "context"} for job in jobs), "Job schema differs")
    require(all(isinstance(job["name"], str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,99}", job["name"]) for job in jobs), "Unsafe job name")
    require(len({job["name"] for job in jobs}) == len(jobs), "Duplicate job name")
    require(all(type(job["identity_index"]) is int for job in jobs)
            and sorted(job["identity_index"] for job in jobs) == list(range(len(evaluations))), "Jobs must cover every identity exactly once")
    provenance, all_sources = {}, {SELF: digest(ROOT / SELF),
        "rl/training/a2c_research_suite.py": digest(ROOT / "rl/training/a2c_research_suite.py")}
    for job in jobs:
        row = evaluations[job["identity_index"]]
        require(row["device"] == "cpu" and type(row["games"]) is int and row["games"] == 100, "Final jobs require CPU and exactly 100 games per color")
        require(row["rules"] in ("freestyle", "renju") and row["opponent"] in ("greedy", "random"), "Unsupported final rules/opponent")
        seeds = (8100000, 8200000) if row["rules"] == "freestyle" else (8300000, 8400000)
        require((row["seed_black"], row["seed_white"]) == seeds, "Final reserved seeds differ")
        require(isinstance(job["context"], dict) and (job["checkpoint"] is None or isinstance(job["checkpoint"], str)), "Malformed job checkpoint/context")
        args = parsed_job(job, row, path)
        require(row["source_sha256"] == evaluator.source_hashes(args), "Exact v2 evaluation source identity differs")
        verify_sources(row["source_sha256"])
        if row["checkpoint_sha256"] is not None:
            require(checksum(row["checkpoint_sha256"]), "Malformed checkpoint SHA")
        provenance[job["name"]] = verify_provenance(job, row, models.get(row["checkpoint_sha256"]))
        all_sources.update(row["source_sha256"])
        all_sources.update(provenance[job["name"]].get("training_sources", {}))
    return manifest, {"manifest_path": str(path), "manifest_sha256": digest(path),
        "sources": all_sources, "jobs": jobs, "provenance": provenance,
        "scope": "Explicit frozen evaluation only; no selection, tuning or automatic freeze"}


def rows(path):
    return list(csv.DictReader(io.StringIO(shared_bytes(path).decode("utf-8-sig"))))


def close(actual, expected):
    return isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8)


def validate_completed(directory, identity, manifest, manifest_sha):
    directory = Path(directory)
    with unheld_lock(directory / "eval.lock"):
        require(read_json(directory / "status.json").get("state") == "complete", "Evaluation is incomplete")
        summary, protocol = read_json(directory / "summary.json"), read_json(directory / "protocol.json")
        require(summary.get("protocol") == protocol and {k: protocol.get(k) for k in IDENTITY_FIELDS} == identity,
                "Completed evaluation identity differs")
        frozen = protocol.get("frozen_selection", {})
        require(protocol.get("split") == "test" and isinstance(frozen, dict)
                and Path(frozen.get("path", "")).resolve() == Path(manifest).resolve()
                and frozen.get("sha256") == manifest_sha and frozen.get("identity") == identity
                and frozen.get("selected_using") == read_json(manifest)["selected_using"], "Recorded frozen manifest differs")
        require(summary.get("sources_and_weights_unchanged") is True and summary.get("total_games") == 200,
                "Missing 200-game unchanged-source result")
        require(not protocol.get("training_performed") and not protocol.get("selection_performed"), "Evaluation performed training/selection")
        if identity["checkpoint_sha256"]:
            require(digest(resolve(protocol["checkpoint"])) == identity["checkpoint_sha256"], "Evaluated checkpoint changed")
        games, moves = rows(directory / "games.csv"), rows(directory / "moves.csv")
        require(len(games) == 200 and len({(int(g["color"]), int(g["game"])) for g in games}) == 200, "Incomplete/duplicate game CSV")
        grouped = defaultdict(list)
        for move in moves:
            grouped[(int(move["color"]), int(move["game"]))].append(move)
        require(set(grouped) == {(int(g["color"]), int(g["game"])) for g in games}, "Game/move key mismatch")
        timing, lengths, reasons, nodes = defaultdict(list), defaultdict(list), defaultdict(Counter), defaultdict(list)
        for game in games:
            color, index, seed, outcome = (int(game[k]) for k in ("color", "game", "seed", "outcome"))
            require(color in (1, -1) and 0 <= index < 100 and outcome in (-1, 0, 1), "Invalid game row")
            require(seed == identity["seed_black" if color == 1 else "seed_white"] + index
                and all(game[k] == identity[k] for k in ("mode", "opponent", "rules"))
                and game["split"] == "test", "Game seed/protocol differs")
            sequence = grouped[(color, index)]
            require(1 <= len(sequence) == int(game["moves"]) <= 81, "Move count differs")
            actions, policy_seconds = set(), 0.
            for ply, move in enumerate(sequence, 1):
                action, player, elapsed = int(move["action"]), int(move["player"]), float(move["inference_seconds"])
                require(int(move["ply"]) == ply and int(move["seed"]) == seed and move["rules"] == identity["rules"]
                    and player == (1 if ply % 2 else -1) and 0 <= action < 81 and action not in actions
                    and math.isfinite(elapsed) and elapsed >= 0, "Malformed move sequence")
                actions.add(action)
                own = player == color
                require(move["actor"] == ("candidate" if own else identity["opponent"]), "Move actor differs")
                if own:
                    timing[color].append(elapsed); policy_seconds += elapsed
                    reasons[color][move["tactical_reason"]] += 1
                    nodes[color].append(int(move["search_nodes"]))
            require(close(float(game["policy_seconds"]), policy_seconds), "Game inference sum differs")
            lengths[color].append(len(sequence))
        results = summary.get("results", [])
        require(len(results) == 2 and {r["color"] for r in results} == {1, -1}, "Missing summary color")
        for result in results:
            color = result["color"]; group = [g for g in games if int(g["color"]) == color]
            require(len(group) == 100 and {int(g["game"]) for g in group} == set(range(100)), "Color budget differs")
            counts = Counter(int(g["outcome"]) for g in group)
            require(result["games"] == 100 and all(result[k] == counts[o] for k, o in (("wins", 1), ("draws", 0), ("losses", -1))), "WDL differs")
            expected = {"win_rate": counts[1]/100, "score": (counts[1]+.5*counts[0])/100,
                "mean_game_plies": statistics.mean(lengths[color]), "inference_samples": len(timing[color]),
                "mean_move_ms": 1000*statistics.mean(timing[color]), "median_move_ms": 1000*statistics.median(timing[color]),
                "p95_move_ms": 1000*sorted(timing[color])[min(len(timing[color])-1, int(.95*len(timing[color])))],
                "max_move_ms": 1000*max(timing[color]), "mean_search_nodes": statistics.mean(nodes[color])}
            require(all(close(result[k], value) for k, value in expected.items()), "Summary score/latency/count differs")
            require(len(result["wilson_95"]) == 2 and all(close(a, b) for a, b in zip(result["wilson_95"], wilson(counts[1], 100)))
                and result["tactical_reasons"] == dict(reasons[color]), "Wilson/reason counts differ")
        require(close(summary["mean_score"], statistics.mean(r["score"] for r in results))
                and close(summary["mean_win_rate"], statistics.mean(r["win_rate"] for r in results)), "Mean score/rate differs")
        return {"directory": str(directory), "identity": identity, "results": results,
            "artifacts": {name: digest(directory/name) for name in ("protocol.json", "summary.json", "status.json", "games.csv", "moves.csv")},
            "validated": "complete identity, source declaration and CSV statistics; independent legality/RNG replay still required"}


class FinalSuite:
    def __init__(self, args):
        self.args = args
        self.manifest_path, self.output = Path(args.frozen_selection).resolve(), resolve(args.output)
        self.manifest, self.snapshot = validate_manifest(self.manifest_path)
        self.processes, self.completed = {}, {}

    def publish(self, state, **details):
        atomic_json(self.output / "suite_status.json", {"format": FORMAT, "state": state,
            "process": process_identity(os.getpid()), "updated_at": now(), "completed_jobs": len(self.completed),
            "expected_jobs": len(self.manifest["jobs"]), "selection_performed": False, **details})

    def guard(self):
        require(digest(self.manifest_path) == self.snapshot["manifest_sha256"], "Frozen manifest changed")
        verify_sources(self.snapshot["sources"])
        for job in self.manifest["jobs"]:
            identity = self.manifest["evaluations"][job["identity_index"]]
            if job["checkpoint"] is not None:
                require(digest(resolve(job["checkpoint"])) == identity["checkpoint_sha256"], "Frozen model changed")
                info = self.snapshot["provenance"][job["name"]]
                require(digest(info["training_protocol"]) == job["context"]["training_protocol_sha256"], "Training provenance changed")
                training_run = resolve(job["context"]["training_run"])
                require(not lock_held(training_run / "run.lock"), "Training writer became active")
                require(read_json(training_run / "status.json").get("state") == "complete", "Training no longer complete")
                if info.get("companion_path"):
                    require(digest(info["companion_path"]) == info["companion_sha256"], "Complete candidate checkpoint changed")
                for source in info.get("lineage", {}).values():
                    require(digest(source["path"]) == source["sha256"], "Frozen lineage changed")

    def base(self, job):
        return self.output / "jobs" / job["name"]

    def observe(self, job):
        base = self.base(job)
        attempts = [path for path in sorted(base.glob("attempt_*")) if path.is_dir()]
        # Observe EVERY known child/lock before accepting any completed attempt.
        # An older complete directory cannot hide a later still-active writer.
        for attempt in attempts:
            if not attempt.is_dir():
                continue
            launcher = read_json(attempt / "launcher.json")
            actual = process_identity(launcher.get("pid")); process = self.processes.get(str(attempt))
            alive = process.poll() is None if process is not None else same_process(launcher, actual)
            # Missing/denied identity information is not evidence of death.
            # A known DIFFERENT birth is an identifiable recycled PID instead.
            unknown = bool(launcher.get("pid") and (
                actual.get("alive") is None or (actual.get("alive") is True and
                (not launcher.get("birth") or not actual.get("birth")))))
            if lock_held(attempt / "eval.lock") or alive or unknown:
                return "active", {"directory": str(attempt), "launcher_process": actual,
                    "runtime_process": process_identity(read_json(attempt / "status.json").get("pid")),
                    "writer_lock_held": lock_held(attempt / "eval.lock")}
            if launcher and launcher.get("exit_code") is None:
                launcher.update(exit_code=process.returncode if process is not None else "stopped; exact exit unavailable",
                    finished_observed_at=now(), process=actual)
                atomic_json(attempt / "launcher.json", launcher)
        for attempt in attempts:
            if read_json(attempt / "status.json").get("state") == "complete":
                identity = self.manifest["evaluations"][job["identity_index"]]
                result = validate_completed(attempt, identity, self.manifest_path, self.snapshot["manifest_sha256"])
                self.completed[job["name"]] = result
                atomic_json(base / "accepted_result.json", result)
                return "complete", result
        require(len(list(base.glob("attempt_*"))) < self.args.max_attempts,
                f"{job['name']}: operational retry limit reached; preserve evidence for review")
        return "pending", None

    def launch(self, job):
        self.guard()
        base = self.base(job)
        attempt = base / f"attempt_{len(list(base.glob('attempt_*')))+1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        identity = self.manifest["evaluations"][job["identity_index"]]
        command = job_command(job, identity, self.manifest_path, attempt)
        log_path = attempt / "console.log"
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                 subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        self.processes[str(attempt)] = process
        actual = process_identity(process.pid)
        atomic_json(attempt / "launcher.json", {"pid": process.pid, "birth": actual.get("birth"),
            "process": actual, "command": command, "cwd": str(ROOT), "started_at": now(),
            "log": str(log_path), "frozen_manifest_sha256": self.snapshot["manifest_sha256"],
            "identity": identity, "context": job["context"],
            "lifecycle": "hidden child; Codex OS job membership may still apply"})
        return {"directory": str(attempt), "launcher_process": actual}

    def run(self):
        configuration = {"format": FORMAT, **self.snapshot, "workers": self.args.workers,
            "max_attempts": self.args.max_attempts, "model_selection_allowed": False,
            "automatic_freezing_allowed": False}
        previous = read_json(self.output / "suite_protocol.json")
        require(not previous or previous == configuration, "Queue manifest/source/plan changed; preserve for explicit review")
        atomic_json(self.output / "suite_protocol.json", configuration)
        while True:
            self.guard()
            active, pending = [], []
            for job in self.manifest["jobs"]:
                if job["name"] in self.completed:
                    continue
                state, observed = self.observe(job)
                if state == "active": active.append({"name": job["name"], **observed})
                elif state == "pending": pending.append(job)
            for job in pending[:max(0, self.args.workers-len(active))]:
                active.append({"name": job["name"], **self.launch(job)})
            atomic_json(self.output / "completed_jobs.json", self.completed)
            if len(self.completed) == len(self.manifest["jobs"]):
                self.guard()
                audit = []
                for job in self.manifest["jobs"]:
                    identity = self.manifest["evaluations"][job["identity_index"]]
                    result = validate_completed(self.completed[job["name"]]["directory"], identity,
                        self.manifest_path, self.snapshot["manifest_sha256"])
                    require(result == self.completed[job["name"]], "Completed evaluation evidence changed")
                    context = dict(job["context"])
                    if context.get("training_run") is not None:
                        context["training_run"] = str(resolve(context["training_run"]))
                    audit.append({"directory": str(Path(result["directory"]).resolve()), **context,
                                  "frozen_selection": str(self.manifest_path)})
                atomic_json(self.output / "audit_inputs.json", {"format": "a2c_independent_audit_inputs_v1",
                    "evaluations": audit})
                self.publish("frozen_evaluations_complete", audit_inputs=str(self.output / "audit_inputs.json"),
                    independent_audit_required=True, model_selection_performed=False)
                return
            self.publish("running", active=active, pending_jobs=len(pending))
            time.sleep(self.args.poll_seconds)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--plan-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    require(0 < args.poll_seconds <= 60, "poll-seconds must be in (0,60]")
    suite = FinalSuite(args)
    if args.plan_only:
        print(json.dumps({"format": FORMAT, **suite.snapshot, "execution_started": False}, indent=2))
        return
    with exclusive_run(suite.output / "final_suite.lock"):
        try:
            suite.run()
        except BaseException as error:
            suite.publish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=repr(error), traceback=traceback.format_exc(),
                child_policy="Preserve children, checkpoints and partial evidence; reattach by OS lock and PID birth")
            raise


if __name__ == "__main__":
    main()
