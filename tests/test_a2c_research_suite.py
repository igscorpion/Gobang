"""Small orchestration checks; never launch model training or expert generation."""

import json
import os
from pathlib import Path
import sys

import pytest

from rl.training import a2c_research_suite as module
from rl.training.common import atomic_json, exclusive_run


def arguments(tmp_path):
    return module.parser().parse_args(["--output", str(tmp_path / "queue"),
                                      "--device", "cpu", "--poll-seconds", "0.01"])


def test_plan_preserves_budgets_and_separates_warm_and_fresh_ablations(tmp_path):
    args = arguments(tmp_path)
    stages = module.build_plan(args)
    assert len(stages) == 12
    assert [s.name for s in stages[:4]] == ["strong_only_seed22", "warm_imitation_seed22",
                                           "demo_a2c_seed22", "demo_regularized_seed22"]
    for stage in stages:
        configuration = vars(module.training_arguments(stage))
        assert configuration["seed"] == 22
        assert configuration["rules"] == "freestyle"
        if stage.kind == "train":
            assert configuration["candidate_every"] == 5000
            assert configuration["validation_games"] == 40
            assert configuration["episodes"] == (10000 if "stage4" in stage.output else 30000)
        else:
            assert configuration["epochs"] == 20
        if stage.name.endswith("_pretrain"):
            assert configuration["initial_model"] is None
    suite = module.ResearchSuite(args)
    command = suite.evaluation_command("checkpoint.pt", "tactical", tmp_path / "evaluation")
    assert command[command.index("--split")+1] == "validation"
    assert command[command.index("--seed-black")+1] == "3100000"
    assert command[command.index("--seed-white")+1] == "3200000"
    assert "--phase" not in command and "test" not in command


def test_actual_os_lock_and_process_birth_are_observed(tmp_path):
    path = tmp_path / "run.lock"
    assert not module.lock_held(path)
    with exclusive_run(path):
        assert module.lock_held(path)
    assert not module.lock_held(path)
    actual = module.process_identity(os.getpid())
    assert actual["alive"] is True and actual["birth"]
    assert module.same_process(actual, actual)
    assert not module.same_process({**actual, "birth": "old-recycled-pid"}, actual)


def test_existing_writer_waits_without_starting_duplicate_child(tmp_path, monkeypatch):
    suite = module.ResearchSuite(arguments(tmp_path))
    stage = suite.stages[0]
    output = Path(stage.output)
    output.mkdir(parents=True)
    atomic_json(output / "status.json", {"state": "training", "pid": os.getpid()})
    launches = []
    monkeypatch.setattr(suite, "spawn", lambda *args: launches.append(args))

    class WaitObserved(Exception):
        pass

    def stop_after_observation(_):
        raise WaitObserved

    monkeypatch.setattr(module.time, "sleep", stop_after_observation)
    with exclusive_run(output / "run.lock"):
        with pytest.raises(WaitObserved):
            suite.ensure_training(stage)
    assert not launches
    status = module.read_json(suite.output / "suite_status.json")
    assert status["state"] == "waiting_existing_writer"
    assert status["observed"]["writer_lock_held"]


def test_child_pid_logs_and_exit_are_recorded_outside_training_output(tmp_path):
    suite = module.ResearchSuite(arguments(tmp_path))
    destination = tmp_path / "untouched_training_output"
    command = [sys.executable, "-u", "-c", "import time; print('orchestration probe', flush=True); time.sleep(.2)"]
    child = suite.spawn("probe_only", command, destination)
    record = module.read_json(suite.job_directory("probe_only") / "launcher.json")
    assert record["pid"] == child.pid and record["birth"]
    assert suite.wait_child("probe_only") == 0
    finished = module.read_json(suite.job_directory("probe_only") / "launcher.json")
    assert finished["exit_code"] == 0 and finished["finished_at"]
    assert "orchestration probe" in Path(finished["log"]).read_text()
    assert not destination.exists()
    status = module.read_json(suite.output / "suite_status.json")
    assert status["pid"] == os.getpid() == status["process"]["pid"]
    assert status["child_process"]["pid"] == child.pid


def test_source_change_blocks_a_fresh_child_before_process_or_log_creation(tmp_path, monkeypatch):
    suite = module.ResearchSuite(arguments(tmp_path))
    atomic_json(suite.output / "suite_protocol.json", {"sources": {"frozen": "old"}})
    monkeypatch.setattr(module, "suite_sources", lambda: {"frozen": "changed"})
    with pytest.raises(RuntimeError, match="source changed before child launch"):
        suite.spawn("must_not_start", [sys.executable, "-c", "raise RuntimeError('launched')"], tmp_path / "unused")
    assert not suite.job_directory("must_not_start").exists()


def test_changed_source_on_stopped_checkpoint_blocks_without_launch(tmp_path, monkeypatch):
    from rl.training import a2c_research
    suite = module.ResearchSuite(arguments(tmp_path))
    stage = suite.stages[0]
    destination = Path(stage.output)
    destination.mkdir(parents=True)
    (destination / "latest.checkpoint").write_bytes(b"not-loaded-before-provenance-check")
    atomic_json(destination / "status.json", {"state": "failed", "pid": 0})
    atomic_json(destination / "protocol.json", {
        "sources": {"old": "source"}, "config": vars(module.training_arguments(stage))})
    monkeypatch.setattr(a2c_research, "source_hashes", lambda: {"new": "source"})
    launches = []
    monkeypatch.setattr(suite, "spawn", lambda *args: launches.append(args))
    with pytest.raises(RuntimeError, match="source changed"):
        suite.ensure_training(stage)
    assert not launches


def test_failed_evaluation_is_preserved_and_next_attempt_is_new(tmp_path, monkeypatch):
    suite = module.ResearchSuite(arguments(tmp_path))
    base = suite.output / "development/group/episode_005000/tactical"
    failed = base / "attempt_001"
    failed.mkdir(parents=True)
    (failed / "moves.csv").write_text("preserve partial evidence")
    atomic_json(failed / "status.json", {"state": "failed"})
    launched = []

    class Launched(Exception):
        pass

    def capture(name, command, destination):
        launched.append((name, command, destination))
        raise Launched

    monkeypatch.setattr(suite, "spawn", capture)
    with pytest.raises(Launched):
        suite.ensure_evaluation("group", "episode_005000", "weights.pt", "tactical")
    assert launched[0][2] == base / "attempt_002"
    assert (failed / "moves.csv").read_text() == "preserve partial evidence"


def test_completed_empty_plan_can_resume_same_serialized_protocol(tmp_path, monkeypatch):
    suite = module.ResearchSuite(arguments(tmp_path))
    suite.stages = []  # Exercise supervisor persistence, with no actual work.
    monkeypatch.setattr(module, "suite_sources", lambda: {"frozen": "hash"})
    suite.run()
    suite.run()
    status = module.read_json(suite.output / "suite_status.json")
    assert status["state"] == "development_queue_complete"
    assert status["final_tests_started"] is False
    frozen = module.read_json(suite.output / "suite_protocol.json")
    assert not frozen["final_tests_allowed"]


def test_tactical_selection_tie_is_earlier_and_separate_from_pure_best(tmp_path, monkeypatch):
    import torch
    suite = module.ResearchSuite(arguments(tmp_path))
    stage = suite.stages[0]
    object.__setattr__(stage, "target", 5000)
    destination = Path(stage.output)
    (destination / "candidates").mkdir(parents=True)
    torch.save({"episode": 5000}, destination / "best_pure.pt")
    atomic_json(destination / "status.json", {"best_pure_score": 0.1})
    for episode in (0, 5000):
        (destination / "candidates" / f"episode_{episode:06d}.pt").write_bytes(str(episode).encode())
    monkeypatch.setattr(suite, "ensure_evaluation", lambda *args: {
        "output": "development_only", "summary": {"mean_score": .8, "mean_win_rate": .6,
                                                    "results": []}})
    suite.record_selection(stage)
    record = suite.selection["groups"][stage.name]
    assert record["best_pure"]["episode"] == 5000
    assert record["best_tactical_development"]["label"] == "episode_000000"
    assert record["selection_complete"]
    assert not suite.selection["final_tests_started"]
