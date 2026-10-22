"""Confirmation orchestration integrity; no real model training or test games."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from rl.training import a2c_confirmation_suite as module
from rl.training.common import atomic_json, exclusive_run


def arguments(tmp_path):
    return module.parser().parse_args(["--research-root", str(tmp_path / "research"),
        "--output", str(tmp_path / "confirmation"), "--device", "cpu", "--poll-seconds", ".01"])


def test_plan_keeps_three_residual_seeds_then_secondary_flat_and_no_tests(tmp_path):
    args = arguments(tmp_path)
    stages = module.build_plan(args)
    assert [s.name for s in stages] == ["residual_tactical10_seed22",
        "residual_tactical10_seed11_pretrain", "residual_tactical10_seed11",
        "residual_tactical10_seed33_pretrain", "residual_tactical10_seed33",
        "flat_tactical10_seed22"]
    assert len({module.training_arguments(s).dataset for s in stages}) == 1
    for stage in stages:
        config = vars(module.training_arguments(stage))
        assert config["features"] == "tactical10" and config["rules"] == "freestyle"
        assert config["seed"] in (11, 22, 33)
        assert config["learning_rate"] == .0003 and config["max_gradient_norm"] == 5
        assert str(tmp_path / "confirmation") in stage.output
        assert "evaluate" not in stage.command and "--phase" not in stage.command
        if stage.kind == "pretrain":
            assert config["epochs"] == 20 and config["initial_model"] is None
            assert not stage.copy_from
        else:
            assert config["episodes"] == 30000 and config["imitation_coef"] == .1
            assert config["num_envs"] == 8 and config["rollout_steps"] == 5
            assert config["candidate_every"] == config["eval_every"] == 5000
            assert config["skip_final"] and config["validation_games"] == 40
            if config["seed"] == 22:
                assert "stage4" in config["initial_model"] and stage.copy_from


def test_existing_source_lock_is_nonmutating_and_excludes_duplicate_writer(tmp_path):
    path = tmp_path / "run.lock"
    path.write_bytes(b"original source lock")
    before = path.read_bytes()
    assert not module.lock_held(path)
    assert path.read_bytes() == before
    with module.readonly_writer_guard(path):
        assert module.lock_held(path)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="lacks"):
        with module.readonly_writer_guard(tmp_path / "missing.lock"):
            pass
    assert not (tmp_path / "missing.lock").exists()


def make_copy_source(tmp_path, monkeypatch):
    stage = module.build_plan(arguments(tmp_path))[0]
    source = Path(stage.copy_from)
    source.mkdir(parents=True)
    (source / "run.lock").write_bytes(b"0")
    config = vars(module.training_arguments(stage)).copy()
    config.update(output=str(source), episodes=10000)
    # Torch checkpoint retains tuples; protocol.json normalizes them to lists.
    saved = {"sources": {}, "config": config, "spec": {"channels": (32, 64, 64)}}
    atomic_json(source / "protocol.json", saved)
    atomic_json(source / "status.json", {"state": "complete", "episode": 10000})
    torch.save({"episode": 10000, "protocol": saved, "optimizer": {}, "rng": {},
        "worker_rngs": ["complete"], "imitation_rng": "state", "agent_rng": "state"},
        source / "latest.checkpoint")
    (source / "training.csv").write_text("immutable original evidence")
    monkeypatch.setattr(module, "verify_inputs", lambda *args, **kwargs: None)
    return stage


def test_copy_is_byte_exact_and_resume_does_not_overwrite_evolved_destination(tmp_path, monkeypatch):
    stage = make_copy_source(tmp_path, monkeypatch)
    before = module.directory_hashes(stage.copy_from)
    record = tmp_path / "copy_provenance.json"
    result = module.copy_checkpoint(stage, record)
    assert result["copy_verified"] and result["source_episode"] == 10000
    assert module.directory_hashes(stage.copy_from) == before == module.directory_hashes(stage.output)
    (Path(stage.output) / "training.csv").write_text("continued to 30k")
    module.copy_checkpoint(stage, record)
    assert (Path(stage.output) / "training.csv").read_text() == "continued to 30k"
    assert module.directory_hashes(stage.copy_from) == before
    (Path(stage.copy_from) / "training.csv").write_text("unauthorized source mutation")
    with pytest.raises(RuntimeError, match="changed"):
        module.copy_checkpoint(stage, record)


def test_copy_refuses_existing_unverified_target_and_active_source(tmp_path, monkeypatch):
    stage = make_copy_source(tmp_path, monkeypatch)
    with exclusive_run(Path(stage.copy_from) / "run.lock"):
        with pytest.raises(RuntimeError, match="Writer lock held"):
            module.copy_checkpoint(stage, tmp_path / "record.json")
    Path(stage.output).mkdir(parents=True)
    (Path(stage.output) / "preserve.txt").write_text("preserve")
    with pytest.raises(RuntimeError, match="without a verified copy"):
        module.copy_checkpoint(stage, tmp_path / "record.json")
    assert (Path(stage.output) / "preserve.txt").read_text() == "preserve"


def test_copied_complete_10k_does_not_skip_30k_training(tmp_path, monkeypatch):
    suite = module.ConfirmationSuite(arguments(tmp_path))
    stage = suite.stages[0]
    directory = Path(stage.output)
    directory.mkdir(parents=True)
    (directory / "latest.checkpoint").write_bytes(b"complete checkpoint stub")
    monkeypatch.setattr(module, "copy_checkpoint", lambda *args: None)
    monkeypatch.setattr(module, "verify_inputs", lambda *args: None)
    monkeypatch.setattr(suite, "assert_sources", lambda: None)
    monkeypatch.setattr(suite, "observe", lambda stage: {"writer_lock_held": False,
        "owned_process_alive": False, "owned_process_unknown": False,
        "status": {"state": "complete", "episode": 10000}})
    commands = []
    class Spawned(Exception):
        pass
    def capture(stage, command):
        commands.append(command)
        raise Spawned
    monkeypatch.setattr(suite, "spawn", capture)
    with pytest.raises(Spawned):
        suite.ensure_training(stage)
    assert commands[0][-1] == "--resume"
    assert commands[0][commands[0].index("--episodes") + 1] == "30000"
    assert not suite.completed


def test_source_guard_prevents_new_child_log_or_process(tmp_path, monkeypatch):
    suite = module.ConfirmationSuite(arguments(tmp_path))
    atomic_json(suite.output / "suite_protocol.json", {"sources": {"old": "sha"}})
    monkeypatch.setattr(module, "suite_sources", lambda: {"new": "sha"})
    with pytest.raises(RuntimeError, match="source changed"):
        suite.spawn(suite.stages[0], [sys.executable, "-c", "raise Exception('must not launch')"])
    assert not suite.jobs(suite.stages[0]).exists()


def test_child_identity_birth_logs_and_status_preserve_supervisor_identity(tmp_path, monkeypatch):
    suite = module.ConfirmationSuite(arguments(tmp_path))
    monkeypatch.setattr(suite, "assert_sources", lambda: None)
    stage = suite.stages[0]
    process = suite.spawn(stage, [sys.executable, "-u", "-c", "print('hidden child probe')"])
    assert process.wait(timeout=10) == 0
    record = module.read_json(suite.jobs(stage) / "launcher.json")
    assert record["pid"] == process.pid and record["birth"]
    assert "hidden child probe" in Path(record["log"]).read_text()
    suite.publish("probe_complete")
    status = module.read_json(suite.output / "suite_status.json")
    assert status["process"]["pid"] == os.getpid() != process.pid
    assert not status["final_tests_started"] and not status["external_evaluations_started"]
