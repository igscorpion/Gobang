"""Training-only A2C confirmation; preserve all original 10k ablation runs.

Three residual/tactical10 training seeds share the same audited AZ400 expert
dataset. Seed 22 resumes a byte-verified COPY of its full 10k checkpoint; seeds
11/33 start fresh random initialization plus CE20. Flat/tactical10 seed22 is a
secondary 30k extension, after the three residual runs. No external policy
evaluation or final test is launched here. The unchanged trainer retains its
ordinary pure-policy development validation and immutable 5k candidates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from rl.training.common import atomic_json, exclusive_run
from rl.training.a2c_research_suite import digest, now, process_identity, read_json, same_process

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "a2c_confirmation_training_queue_v1"
DATASET_SHA256 = "ca566f90aaf7695ad8fd4b82e997ff221bb483859825ab6e5e589942266307e3"


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


@contextmanager
def readonly_writer_guard(path):
    """Acquire an existing run lock without creating or changing source files."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Source lacks its expected run.lock: {path}")
    with path.open("r+b") as handle:
        if path.stat().st_size < 1:
            raise RuntimeError(f"Refuse malformed zero-length source lock: {path}")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Writer lock held: {path}") from error
        try:
            yield handle
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def lock_held(path):
    path = Path(path)
    if not path.exists():
        return False
    try:
        with readonly_writer_guard(path):
            return False
    except RuntimeError as error:
        if isinstance(error.__cause__, OSError):
            return True
        raise


def directory_hashes(directory, lock_bytes=None):
    directory = Path(directory)
    return {str(path.relative_to(directory)): (
                hashlib.sha256(lock_bytes).hexdigest()
                if path == directory / "run.lock" and lock_bytes is not None else digest(path))
            for path in sorted(directory.rglob("*")) if path.is_file()}


@dataclass(frozen=True)
class Stage:
    name: str
    output: str
    command: tuple[str, ...]
    kind: str
    target: int
    copy_from: str | None = None


def build_plan(args):
    research, output = resolve(args.research_root), resolve(args.output)
    dataset = research / "stage3/expert_dataset/demonstrations.pt"
    stages = []
    for architecture, seed in (("residual", 22), ("residual", 11),
                               ("residual", 33), ("flat", 22)):
        label = f"{architecture}_tactical10_seed{seed}"
        destination = output / label
        original = research / "stage4" / label
        common = ("--device", args.device, "--seed", str(seed), "--rules", "freestyle",
                  "--architecture", architecture, "--features", "tactical10",
                  "--dataset", str(dataset), "--learning-rate", "0.0003",
                  "--max-gradient-norm", "5")
        if seed != 22:
            command = (sys.executable, "-u", "-m", "rl.training.a2c_research", "pretrain",
                       "--output", str(destination / "pretrain"), *common,
                       "--epochs", "20", "--batch-size", "256")
            stages.append(Stage(label + "_pretrain", str(destination / "pretrain"),
                                command, "pretrain", 20))
        initial = (original if seed == 22 else destination) / "pretrain/best.pt"
        command = (sys.executable, "-u", "-m", "rl.training.a2c_research", "train",
                   "--output", str(destination / "train"), *common,
                   "--episodes", "30000", "--num-envs", "8", "--rollout-steps", "5",
                   "--gamma", "0.99", "--value-coef", "0.5", "--entropy-coef", "0.01",
                   "--imitation-coef", "0.1", "--imitation-batch-size", "64",
                   "--save-every", "250", "--eval-every", "5000", "--candidate-every", "5000",
                   "--validation-games", "40", "--validation-seed", "3000000",
                   "--skip-final", "--initial-model", str(initial))
        stages.append(Stage(label, str(destination / "train"), command, "train", 30000,
                            str(original / "train") if seed == 22 else None))
    return stages


def training_arguments(stage):
    from rl.training.a2c_research import parser as training_parser
    return training_parser().parse_args(list(stage.command[4:]))


def suite_sources():
    from rl.training.a2c_research import source_hashes
    return {**source_hashes(), **{name: digest(ROOT / name) for name in (
        "rl/training/a2c_confirmation_suite.py", "rl/training/a2c_research_suite.py")}}


def verify_inputs(stage, checkpoint=False):
    from rl.training.a2c_research import source_hashes
    path = Path(stage.output)
    saved = read_json(path / "protocol.json")
    if not saved:
        if checkpoint:
            raise RuntimeError("Durable checkpoint lacks protocol.json")
        return
    if saved.get("sources") != source_hashes():
        raise RuntimeError(f"{stage.name}: source changed; preserve this run")
    expected, actual = vars(training_arguments(stage)), saved["config"]
    for key, value in expected.items():
        if key in ("output", "resume"):
            continue
        old = actual.get(key)
        if key in ("initial_model", "dataset") and (old or value):
            if not old or not value or resolve(old) != resolve(value):
                raise RuntimeError(f"{stage.name}: changed {key}")
        elif key in ("episodes", "epochs"):
            if old is None or not 0 < old <= value:
                raise RuntimeError(f"{stage.name}: budget shrank or malformed")
        elif old != value:
            raise RuntimeError(f"{stage.name}: changed {key}: {old!r} -> {value!r}")
    for argument, field in (("initial_model", "initial_sha256"), ("dataset", "dataset_sha256")):
        value = expected.get(argument)
        if saved.get(field) != (digest(resolve(value)) if value else None):
            raise RuntimeError(f"{stage.name}: changed {argument} hash")


def copy_checkpoint(stage, record_path):
    """Copy a completed 10k run under its OS lock; never overwrite a target."""
    import torch
    source, destination = Path(stage.copy_from), Path(stage.output)
    record_path = Path(record_path)
    existing = read_json(record_path)
    if destination.exists():
        if not existing or not existing.get("copy_verified"):
            raise RuntimeError(f"Destination exists without a verified copy: {destination}")
        if existing.get("source") != str(source) or existing.get("destination") != str(destination):
            raise RuntimeError("Copy provenance path mismatch")
        if directory_hashes(source) != existing["source_file_sha256"]:
            raise RuntimeError("Original 10k run changed after copying")
        return existing
    with readonly_writer_guard(source / "run.lock") as lock_handle:
        lock_handle.seek(0)
        lock_bytes = lock_handle.read()
        status = read_json(source / "status.json")
        if status.get("state") != "complete" or status.get("episode") != 10000:
            raise RuntimeError("Only a completed 10k source can be extended")
        old_stage = Stage(stage.name, str(source), stage.command, stage.kind, stage.target)
        verify_inputs(old_stage, checkpoint=True)
        durable = torch.load(source / "latest.checkpoint", map_location="cpu", weights_only=False)
        durable_metadata = json.loads(json.dumps(durable["protocol"]))
        if durable["episode"] != 10000 or durable_metadata != read_json(source / "protocol.json"):
            raise RuntimeError("Original durable checkpoint mismatches 10k protocol/status")
        for key in ("optimizer", "rng", "worker_rngs", "imitation_rng", "agent_rng"):
            if key not in durable:
                raise RuntimeError(f"Missing complete resume state: {key}")
        source_files = directory_hashes(source, lock_bytes)
        record = {"source": str(source), "destination": str(destination),
                  "source_episode": 10000, "target_episode": 30000,
                  "source_file_sha256": source_files, "copy_started_at": now(),
                  "copy_verified": False, "original_preserved": True}
        atomic_json(record_path, record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Windows locks forbid a second handle reading the locked byte, even in
        # this process. Copy that unchanged lock content from its owning handle.
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("run.lock"))
        (destination / "run.lock").write_bytes(lock_bytes)
        if directory_hashes(source, lock_bytes) != source_files or directory_hashes(destination) != source_files:
            raise RuntimeError("Source changed during copying or copy is not byte-identical")
        record.update(copy_verified=True, copy_finished_at=now(),
                      checkpoint_sha256=digest(destination / "latest.checkpoint"))
        atomic_json(record_path, record)
        return record


class ConfirmationSuite:
    def __init__(self, args):
        self.args, self.output = args, resolve(args.output)
        self.stages = build_plan(args)
        self.processes, self.completed = {}, []

    def publish(self, state, **details):
        atomic_json(self.output / "suite_status.json", {"format": FORMAT, "state": state,
            "process": process_identity(os.getpid()), "updated_at": now(),
            "completed_stages": self.completed, "final_tests_started": False,
            "external_evaluations_started": False, **details})

    def jobs(self, stage):
        return self.output / "jobs" / stage.name

    def assert_sources(self):
        frozen = read_json(self.output / "suite_protocol.json")
        if frozen and frozen["sources"] != suite_sources():
            raise RuntimeError("Confirmation source changed; preserve queue for explicit review")
        if digest(resolve(self.args.research_root) / "stage3/expert_dataset/demonstrations.pt") != DATASET_SHA256:
            raise RuntimeError("Expert dataset changed")

    def observe(self, stage):
        record = read_json(self.jobs(stage) / "launcher.json")
        actual = process_identity(record.get("pid"))
        process = self.processes.get(stage.name)
        alive = process.poll() is None if process is not None else same_process(record, actual)
        status = read_json(Path(stage.output) / "status.json")
        runtime = process_identity(status.get("pid"))
        return {"writer_lock_held": lock_held(Path(stage.output) / "run.lock"),
                "owned_process_alive": alive,
                "owned_process_unknown": bool(record.get("birth") and actual.get("alive") is None),
                "launcher_process": actual, "runtime_process": runtime, "status": status}

    def spawn(self, stage, command):
        self.assert_sources()
        jobs = self.jobs(stage)
        jobs.mkdir(parents=True, exist_ok=True)
        attempt = read_json(jobs / "launcher.json").get("attempt", 0) + 1
        log_path = jobs / f"attempt_{attempt:03d}.log"
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                 subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        self.processes[stage.name] = process
        identity = process_identity(process.pid)
        record = {"pid": process.pid, "birth": identity.get("birth"), "process": identity,
            "command": command, "cwd": str(ROOT), "output": stage.output,
            "attempt": attempt, "started_at": now(), "log": str(log_path),
            "creationflags": flags, "lifecycle": "hidden child; may remain inside Codex OS job"}
        atomic_json(jobs / "launcher.json", record)
        atomic_json(jobs / f"attempt_{attempt:03d}.json", record)
        return process

    def verify_complete(self, stage, status):
        import torch
        directory = Path(stage.output)
        field = "episode" if stage.kind == "train" else "epoch"
        durable = torch.load(directory / "latest.checkpoint", map_location="cpu", weights_only=False)
        if status.get(field) != stage.target or durable.get(field) != stage.target:
            raise RuntimeError(f"{stage.name}: incomplete target budget")
        required = ["current.pt", "final.pt", "best_pure.pt" if stage.kind == "train" else "best.pt"]
        if stage.kind == "train":
            required += [f"candidates/episode_{i:06d}{suffix}"
                for i in range(0, 30001, 5000) for suffix in (".pt", ".checkpoint")]
        if any(not (directory / name).is_file() for name in required):
            raise RuntimeError(f"{stage.name}: missing final artifacts or immutable candidates")

    def ensure_training(self, stage):
        directory = Path(stage.output)
        if stage.copy_from:
            copy_checkpoint(stage, self.jobs(stage) / "copy_provenance.json")
        while True:
            self.assert_sources()
            observed = self.observe(stage)
            if any(observed[name] for name in ("writer_lock_held", "owned_process_alive", "owned_process_unknown")):
                self.publish("waiting_existing_writer", active_stage=stage.name, observed=observed)
                time.sleep(self.args.poll_seconds)
                continue
            checkpoint = directory / "latest.checkpoint"
            verify_inputs(stage, checkpoint.exists())
            field = "episode" if stage.kind == "train" else "epoch"
            status = observed["status"]
            # A copied 10k complete state MUST resume to 30k, never count as done.
            if status.get("state") == "complete" and status.get(field) == stage.target:
                self.verify_complete(stage, status)
                self.completed.append(stage.name)
                self.publish("stage_complete", active_stage=stage.name)
                return
            if directory.exists() and not checkpoint.exists() and any(p.name != "run.lock" for p in directory.iterdir()):
                raise RuntimeError(f"{stage.name}: nonempty output lacks full checkpoint; preserve it")
            command = list(stage.command) + (["--resume"] if checkpoint.exists() else [])
            self.publish("launching", active_stage=stage.name, command=command)
            process = self.spawn(stage, command)
            while process.poll() is None:
                self.publish("running_child", active_stage=stage.name, observed=self.observe(stage))
                time.sleep(self.args.poll_seconds)
            record = read_json(self.jobs(stage) / "launcher.json")
            record.update(exit_code=process.returncode, finished_at=now(),
                          process=process_identity(process.pid))
            atomic_json(self.jobs(stage) / "launcher.json", record)
            atomic_json(self.jobs(stage) / f"attempt_{record['attempt']:03d}.json", record)
            if process.returncode and not lock_held(directory / "run.lock"):
                raise RuntimeError(f"{stage.name}: exited {process.returncode}; see {self.jobs(stage)}")

    def validate_dataset(self):
        from rl.training.a2c_research import load_demonstrations
        directory = resolve(self.args.research_root) / "stage3/expert_dataset"
        status, manifest = read_json(directory / "status.json"), read_json(directory / "manifest.json")
        if status.get("state") != "complete" or status.get("games") != 576 or lock_held(directory / "run.lock"):
            raise RuntimeError("Formal 576-game expert dataset is not finished and unlocked")
        path = directory / "demonstrations.pt"
        if digest(path) != DATASET_SHA256 or manifest.get("sha256") != DATASET_SHA256:
            raise RuntimeError("Formal expert dataset SHA mismatch")
        config = manifest["protocol"]["config"]
        for key, value in {"train_games": 512, "validation_games": 64, "train_seed": 5000000,
                           "validation_seed": 5100000, "sims": 400}.items():
            if config.get(key) != value:
                raise RuntimeError(f"Dataset protocol mismatch: {key}")
        expert = manifest["protocol"]["expert"]
        if digest(resolve(expert["checkpoint"])) != expert["checkpoint_sha256"]:
            raise RuntimeError("Teacher weight hash changed")
        data = load_demonstrations(path, "freestyle")
        return {"path": str(path), "sha256": DATASET_SHA256,
                "manifest_sha256": digest(directory / "manifest.json"),
                "shared_teacher": expert, "shared_data_across_seeds": True,
                "train_samples": len(data["train"]["actions"]),
                "validation_samples": len(data["validation"]["actions"])}

    def run(self):
        dataset = self.validate_dataset()
        configuration = json.loads(json.dumps({"format": FORMAT,
            "stages": [asdict(stage) for stage in self.stages], "sources": suite_sources(),
            "expert_dataset": dataset, "initialization": "fresh random + CE20, not production A2C warm-start",
            "seed22_extension": "byte-identical copy of full 10k state; extend only in confirmation",
            "opponent_episode_probabilities": {"greedy": .6, "random": .2, "selfplay": .2},
            "pure_validation": "unchanged trainer, 40/color; candidate selection external",
            "final_tests_allowed": False, "external_policy_evaluations_allowed": False}))
        previous = read_json(self.output / "suite_protocol.json")
        if previous and previous != configuration:
            raise RuntimeError("Confirmation source/plan/input changed; preserve queue for review")
        atomic_json(self.output / "suite_protocol.json", configuration)
        self.publish("starting")
        for stage in self.stages:
            self.ensure_training(stage)
        self.publish("confirmation_training_complete",
            next_step="External guarded/pure development candidate validation and independent frozen final testing")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-root", default="runs/a2c_research_20261020")
    p.add_argument("--output", default="runs/a2c_research_20261020/confirmation")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--poll-seconds", type=float, default=20)
    p.add_argument("--plan-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if not 0 < args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be in (0, 60]")
    if args.plan_only:
        print(json.dumps({"format": FORMAT, "stages": [asdict(s) for s in build_plan(args)],
                          "final_tests_allowed": False}, indent=2))
        return
    suite = ConfirmationSuite(args)
    with exclusive_run(suite.output / "confirmation_suite.lock"):
        try:
            suite.run()
        except BaseException as error:
            suite.publish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=repr(error), traceback=traceback.format_exc(),
                          child_policy="Preserve live children; rerun supervisor to reattach via OS locks and birth")
            raise


if __name__ == "__main__":
    main()
