"""Synthetic orchestration tests only: no policy inference or final games.

Reserved-seed identities are dictionaries only. Synthetic CSV fixtures use
6.1m/6.2m; subprocess probes only print text and do not call an evaluator.
"""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from rl.evaluation import a2c_research_final_suite as module
from rl.training.common import exclusive_run


@pytest.fixture
def frozen(tmp_path):
    run = tmp_path / "synthetic_training"
    (run / "candidates").mkdir(parents=True)
    (run / "run.lock").write_bytes(b"0")
    spec = {"architecture": "residual", "features": "tactical10", "board_size": 9,
        "channels": [32, 64, 64], "hidden_units": 256, "residual_width": 64, "residual_blocks": 4}
    sources = {"rl/training/a2c_math.py": module.digest(module.ROOT / "rl/training/a2c_math.py")}
    training = {"config": {"seed": 11, "rules": "freestyle", "command": "train"},
        "sources": sources, "spec": spec}
    module.atomic_json(run / "protocol.json", training)
    module.atomic_json(run / "status.json", {"state": "complete", "episode": 30000})
    checkpoint = run / "candidates/episode_005000.pt"
    weights = {"synthetic_layer": torch.tensor([1., 2.])}
    torch.save({"spec": spec, "state_dict": weights, "episode": 5000, "rules": "freestyle"}, checkpoint)
    torch.save({"protocol": training, "net": weights, "episode": 5000}, checkpoint.with_suffix(".checkpoint"))
    args = module.evaluator.parser().parse_args(["--checkpoint", str(checkpoint), "--mode", "guarded", "--output", "unused"])
    identity = {key: getattr(args, key) for key in module.IDENTITY_FIELDS[:-2]}
    identity.update(games=100, seed_black=8100000, seed_white=8200000,
        checkpoint_sha256=module.digest(checkpoint), source_sha256=module.evaluator.source_hashes(args))
    context = {"role": "candidate", "recipe": "synthetic_no_games", "training_seed": 11,
        "training_run": str(run), "training_protocol_sha256": module.digest(run / "protocol.json"),
        "checkpoint_sha256": module.digest(checkpoint)}
    manifest = {"status": "frozen", "selected_using": "development only",
        "expected_runner_sha256": module.digest(module.ROOT / module.SELF),
        "models": {identity["checkpoint_sha256"]: {**context, "model_spec": spec, "selected_episode": 5000}},
        "evaluations": [identity], "jobs": [{"name": "synthetic_seed11", "checkpoint": str(checkpoint),
            "identity_index": 0, "context": context}]}
    path = tmp_path / "synthetic_frozen.json"
    module.atomic_json(path, manifest)
    return path, manifest, run


def arguments(path, output):
    return module.parser().parse_args(["--frozen-selection", str(path), "--output", str(output), "--poll-seconds", ".01"])


def test_plan_only_validates_exact_schema_without_output_or_children(frozen, tmp_path, monkeypatch, capsys):
    path, manifest, _ = frozen
    output = tmp_path / "must_not_be_created"
    monkeypatch.setattr(sys, "argv", ["final_suite", "--frozen-selection", str(path), "--output", str(output), "--plan-only"])
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("plan-only spawned a child"))
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    module.main()
    assert not output.exists()
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert json.loads(capsys.readouterr().out)["execution_started"] is False
    command = module.job_command(manifest["jobs"][0], manifest["evaluations"][0], path, "unused")
    parsed = module.evaluator.parser().parse_args(command[4:])
    assert parsed.split == "test" and parsed.games == 100 and parsed.device == "cpu"
    assert (parsed.seed_black, parsed.seed_white) == (8100000, 8200000)
    assert parsed.frozen_selection == str(path)


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.update(status="planned"), "must be frozen"),
    (lambda m: m.update(selected_using="test results"), "development only"),
    (lambda m: m.update(expected_runner_sha256="0"*64), "runner SHA"),
    (lambda m: m["evaluations"][0].update(recipe="forbidden context"), "16 fields"),
    (lambda m: m["evaluations"][0].update(games=40), "100 games"),
    (lambda m: m["evaluations"][0].update(seed_black=3100000), "reserved seeds"),
    (lambda m: m["evaluations"][0].update(device="cuda"), "CPU"),
    (lambda m: m["jobs"][0].update(identity_index=1), "cover every identity"),
    (lambda m: m["jobs"][0].update(name="../escape"), "Unsafe job"),
    (lambda m: m["jobs"][0]["context"].update(training_seed=22), "context mismatch|context differs"),
    (lambda m: m["models"].clear(), "lacks checkpoint provenance"),
])
def test_invalid_freeze_is_rejected_before_any_output(frozen, tmp_path, mutation, match):
    path, manifest, _ = frozen
    mutation(manifest)
    module.atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match=match):
        module.FinalSuite(arguments(path, tmp_path / "unused"))
    assert not (tmp_path / "unused").exists()


def test_duplicate_identity_job_and_missing_job_are_rejected(frozen):
    path, manifest, _ = frozen
    duplicate = deepcopy(manifest)
    duplicate["evaluations"].append(deepcopy(duplicate["evaluations"][0]))
    duplicate["jobs"].append({**duplicate["jobs"][0], "name": "another", "identity_index": 1})
    module.atomic_json(path, duplicate)
    with pytest.raises(RuntimeError, match="Duplicate evaluation"):
        module.validate_manifest(path)
    duplicate["evaluations"][1]["opponent"] = "random"
    duplicate["jobs"][1]["name"] = duplicate["jobs"][0]["name"]
    module.atomic_json(path, duplicate)
    with pytest.raises(RuntimeError, match="Duplicate job"):
        module.validate_manifest(path)
    duplicate["jobs"].pop()
    module.atomic_json(path, duplicate)
    with pytest.raises(RuntimeError, match="complete jobs"):
        module.validate_manifest(path)


def test_minimax_requires_reference_context_without_model_or_fabricated_seed(frozen):
    path, manifest, _ = frozen
    job, identity = manifest["jobs"][0], manifest["evaluations"][0]
    identity.update(mode="minimax", checkpoint_sha256=None)
    identity["source_sha256"] = module.evaluator.source_hashes(module.parsed_job({**job, "checkpoint": None}, identity, path))
    job.update(checkpoint=None, context={"role": "contextual_reference", "recipe": "minimax_depth2_candidates8",
        "training_seed": None, "training_run": None, "training_protocol_sha256": None, "checkpoint_sha256": None})
    manifest["models"] = {}
    module.atomic_json(path, manifest)
    accepted, snapshot = module.validate_manifest(path)
    assert snapshot["provenance"][job["name"]]["checkpoint"] is None
    job["context"]["training_seed"] = 11
    module.atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match="invent training"):
        module.validate_manifest(path)


def test_live_training_changed_companion_and_changed_manifest_block_launch(frozen, tmp_path):
    path, manifest, run = frozen
    with exclusive_run(run / "run.lock"):
        with pytest.raises(RuntimeError, match="active"):
            module.FinalSuite(arguments(path, tmp_path / "queue"))
    suite = module.FinalSuite(arguments(path, tmp_path / "queue"))
    companion = Path(manifest["jobs"][0]["checkpoint"]).with_suffix(".checkpoint")
    companion.write_bytes(b"changed complete checkpoint")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        suite.guard()
    manifest["selected_using"] = "changed declaration"
    module.atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match="manifest changed"):
        suite.guard()
    assert not (tmp_path / "queue").exists()


def test_legacy_reference_without_episode_retains_unknown_update_metadata(frozen, tmp_path):
    path, manifest, _ = frozen
    run = tmp_path / "synthetic_legacy_reference"; run.mkdir()
    config = {"seed": 11, "rules": "freestyle", "channels": [32, 64, 64], "hidden_units": 256}
    module.atomic_json(run / "config.json", config)
    module.atomic_json(run / "status.json", {"state": "complete"})
    (run / "run.lock").write_bytes(b"0")
    checkpoint = run / "best.pt"
    torch.save({"state_dict": {"synthetic_layer": torch.ones(2)}, "channels": [32, 64, 64],
        "input_channels": 4, "hidden_units": 256}, checkpoint)
    identity = manifest["evaluations"][0]
    identity.update(network_kind="standard", mode="tactical", checkpoint_sha256=module.digest(checkpoint))
    job = manifest["jobs"][0]
    context = {"role": "contextual_reference", "recipe": "legacy_A2C_fixture", "training_seed": 11,
        "training_run": str(run), "training_protocol_sha256": module.digest(run / "config.json"),
        "checkpoint_sha256": identity["checkpoint_sha256"]}
    job.update(checkpoint=str(checkpoint), context=context)
    identity["source_sha256"] = module.evaluator.source_hashes(module.parsed_job(job, identity, path))
    manifest["models"] = {identity["checkpoint_sha256"]: context}
    module.atomic_json(path, manifest)
    _, snapshot = module.validate_manifest(path)
    provenance = snapshot["provenance"][job["name"]]
    assert provenance["legacy_config"] is True and provenance["selected_has_A2C_updates"] is None


def synthetic_result(directory, identity, manifest):
    """Pure CSV fixture at NON-reserved seeds; no game engine or agent runs."""
    identity = {**identity, "seed_black": 6100000, "seed_white": 6200000}
    directory.mkdir(parents=True)
    (directory / "eval.lock").write_bytes(b"0")
    protocol = {**identity, "split": "test", "training_performed": False, "selection_performed": False,
        "checkpoint": str(manifest.parent / "synthetic_training/candidates/episode_005000.pt"),
        "frozen_selection": {"path": str(manifest), "sha256": module.digest(manifest),
            "identity": identity, "selected_using": "development only"}, "synthetic_fixture_no_games_played": True}
    game_rows, move_rows, results = [], [], []
    actions = [0, 9, 1, 10, 2, 11, 3, 12, 4]
    for color in (1, -1):
        count = 5 if color == 1 else 4
        for game in range(100):
            seed = identity["seed_black" if color == 1 else "seed_white"] + game
            game_rows.append({"split": "test", "mode": identity["mode"], "opponent": "greedy", "rules": "freestyle",
                "color": color, "game": game, "seed": seed, "outcome": color, "moves": 9, "policy_seconds": count*.001})
            for ply, action in enumerate(actions, 1):
                player = 1 if ply % 2 else -1
                move_rows.append({"rules": "freestyle", "color": color, "game": game, "seed": seed,
                    "ply": ply, "player": player, "actor": "candidate" if color == player else "greedy",
                    "action": action, "inference_seconds": .001, "tactical_reason": "", "search_nodes": 0})
        results.append({"color": color, "games": 100, "wins": 100 if color == 1 else 0,
            "draws": 0, "losses": 0 if color == 1 else 100, "win_rate": 1. if color == 1 else 0.,
            "score": 1. if color == 1 else 0., "wilson_95": module.wilson(100 if color == 1 else 0, 100),
            "mean_game_plies": 9., "inference_samples": count*100, "mean_move_ms": 1.,
            "median_move_ms": 1., "p95_move_ms": 1., "max_move_ms": 1.,
            "tactical_reasons": {"": count*100}, "mean_search_nodes": 0.})
    for name, values in [("games.csv", game_rows), ("moves.csv", move_rows)]:
        with (directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=values[0].keys()); writer.writeheader(); writer.writerows(values)
    module.atomic_json(directory / "protocol.json", protocol)
    module.atomic_json(directory / "summary.json", {"protocol": protocol, "results": results, "total_games": 200,
        "sources_and_weights_unchanged": True, "mean_score": .5, "mean_win_rate": .5})
    module.atomic_json(directory / "status.json", {"state": "complete", "games": 200})
    return identity


def test_complete_csv_reuse_checks_statistics_identity_and_partial(frozen, tmp_path):
    path, manifest, _ = frozen
    directory = tmp_path / "non8m_csv_fixture"
    identity = synthetic_result(directory, manifest["evaluations"][0], path)
    result = module.validate_completed(directory, identity, path, module.digest(path))
    assert result["artifacts"]["games.csv"] and len(result["results"]) == 2
    with exclusive_run(directory / "eval.lock"):
        with pytest.raises(RuntimeError, match="Active writer"):
            module.validate_completed(directory, identity, path, module.digest(path))
    summary = module.read_json(directory / "summary.json")
    summary["results"][1]["wins"] = 100
    module.atomic_json(directory / "summary.json", summary)
    with pytest.raises(RuntimeError, match="WDL"):
        module.validate_completed(directory, identity, path, module.digest(path))


def test_partial_attempt_preserved_and_hidden_probe_does_not_run_evaluator(frozen, tmp_path, monkeypatch):
    path, _, _ = frozen
    suite = module.FinalSuite(arguments(path, tmp_path / "queue"))
    job = suite.manifest["jobs"][0]
    partial = suite.base(job) / "attempt_001"
    partial.mkdir(parents=True)
    (partial / "games.csv").write_text("preserve incomplete evidence")
    module.atomic_json(partial / "status.json", {"state": "failed"})
    monkeypatch.setattr(module, "job_command", lambda *a: [sys.executable, "-u", "-c", "print('scheduler probe only; no games')"])
    result = suite.launch(job)
    process = suite.processes[result["directory"]]
    assert process.wait(timeout=10) == 0
    directory = Path(result["directory"])
    assert directory.name == "attempt_002"
    assert (partial / "games.csv").read_text() == "preserve incomplete evidence"
    launcher = module.read_json(directory / "launcher.json")
    assert launcher["pid"] == process.pid and launcher["birth"]
    assert "scheduler probe only" in (directory / "console.log").read_text()
    assert not (directory / "games.csv").exists()


def test_active_later_attempt_prevents_reuse_of_older_complete(frozen, tmp_path, monkeypatch):
    path, _, _ = frozen
    suite = module.FinalSuite(arguments(path, tmp_path / "queue")); job = suite.manifest["jobs"][0]
    first, second = (suite.base(job)/name for name in ("attempt_001", "attempt_002"))
    first.mkdir(parents=True); second.mkdir()
    module.atomic_json(first / "status.json", {"state": "complete"})
    monkeypatch.setattr(module, "validate_completed", lambda *a: pytest.fail("accepted old result before checking every writer"))
    with exclusive_run(second / "eval.lock"):
        state, _ = suite.observe(job)
    assert state == "active" and not suite.completed


@pytest.mark.parametrize("birth,actual,expected", [
    (None, {"pid": 123456, "alive": True, "birth": "actual"}, "active"),
    ("recorded", {"pid": 123456, "alive": True, "birth": None}, "active"),
    (None, {"pid": 123456, "alive": None, "birth": None}, "active"),
    ("recorded", {"pid": 123456, "alive": True, "birth": "different-recycled"}, "pending"),
])
def test_missing_process_identity_waits_but_known_recycled_pid_does_not(frozen, tmp_path, monkeypatch, birth, actual, expected):
    path, _, _ = frozen
    suite = module.FinalSuite(arguments(path, tmp_path / "queue")); job = suite.manifest["jobs"][0]
    attempt = suite.base(job) / "attempt_001"; attempt.mkdir(parents=True)
    module.atomic_json(attempt / "launcher.json", {"pid": 123456, "birth": birth})
    monkeypatch.setattr(module, "process_identity", lambda pid: actual)
    state, _ = suite.observe(job)
    assert state == expected


def test_complete_mock_queue_resumes_without_spawn_and_exports_absolute_audit_context(frozen, tmp_path, monkeypatch):
    path, manifest, run = frozen
    relative = os.path.relpath(run, module.ROOT)
    manifest["jobs"][0]["context"]["training_run"] = relative
    manifest["models"][manifest["evaluations"][0]["checkpoint_sha256"]]["training_run"] = relative
    module.atomic_json(path, manifest)
    suite = module.FinalSuite(arguments(path, tmp_path / "queue")); job = suite.manifest["jobs"][0]
    attempt = suite.base(job) / "attempt_001"; attempt.mkdir(parents=True)
    (attempt / "eval.lock").write_bytes(b"0")
    module.atomic_json(attempt / "status.json", {"state": "complete"})
    result = {"directory": str(attempt), "identity": manifest["evaluations"][0], "mock_no_games": True}
    monkeypatch.setattr(module, "validate_completed", lambda *a: result)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("completed mock queue spawned"))
    suite.run()
    resumed = module.FinalSuite(arguments(path, tmp_path / "queue")); resumed.run()
    audit = module.read_json(suite.output / "audit_inputs.json")
    assert audit["format"] == "a2c_independent_audit_inputs_v1"
    row = audit["evaluations"][0]
    assert Path(row["directory"]).is_absolute() and Path(row["training_run"]) == run
    assert row["frozen_selection"] == str(path) and row["training_seed"] == 11
    assert module.read_json(suite.output / "suite_status.json")["independent_audit_required"] is True


def test_local_atomic_winerror5_retry_is_bounded_and_other_errors_propagate(tmp_path, monkeypatch):
    original, attempts, delays = module.original_atomic_json, [], []
    def sometimes(path, value):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError("synthetic Windows access conflict"); error.winerror = 5; raise error
        original(path, value)
    monkeypatch.setattr(module, "original_atomic_json", sometimes)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    module.atomic_json(tmp_path / "state.json", {"ok": True})
    assert len(attempts) == 3 and delays == [.01, .02]
    def always(*a):
        attempts.append(1); error = PermissionError("persistent synthetic conflict"); error.winerror = 5; raise error
    attempts.clear(); delays.clear(); monkeypatch.setattr(module, "original_atomic_json", always)
    with pytest.raises(PermissionError): module.atomic_json(tmp_path / "failed.json", {})
    assert len(attempts) == 8 and len(delays) == 7
    attempts.clear(); delays.clear()
    def other_error(*a):
        attempts.append(1); error = PermissionError("non-WinError5 failure"); error.winerror = 32; raise error
    monkeypatch.setattr(module, "original_atomic_json", other_error)
    with pytest.raises(PermissionError): module.atomic_json(tmp_path / "different.json", {})
    assert len(attempts) == 1 and not delays


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing request")
def test_shared_reader_requests_delete_sharing_and_closes_its_handle(tmp_path, monkeypatch):
    import ctypes
    path, replacement = tmp_path / "active.json", tmp_path / "replacement.json"
    path.write_bytes(b"old complete snapshot"); replacement.write_bytes(b"new complete snapshot")
    from ctypes import wintypes
    real_dll = ctypes.WinDLL("kernel32", use_last_error=True)
    real_dll.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    real_dll.CreateFileW.restype = wintypes.HANDLE
    requested = []
    class Proxy:
        CloseHandle = real_dll.CloseHandle
        @staticmethod
        def CreateFileW(*args):
            requested.append(args[2])
            return real_dll.CreateFileW(*args)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: Proxy())
    assert module.shared_bytes(path) == b"old complete snapshot"
    assert requested == [7]  # FILE_SHARE_READ | WRITE | DELETE.
    replacement.replace(path)  # Reader has closed its handle before returning.
    assert path.read_bytes() == b"new complete snapshot"
