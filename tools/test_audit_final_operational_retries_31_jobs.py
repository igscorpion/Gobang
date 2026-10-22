"""Non-8m synthetic tests. Never imports project modules or opens real runs."""
from contextlib import contextmanager
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_final_operational_retries_31_jobs as tool


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def csv_bytes(rows):
    result = io.StringIO(newline="")
    writer = csv.DictWriter(result, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return result.getvalue().encode()


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        (self.root/"SYNTHETIC_NONRESERVED_FIXTURE").write_text("Synthetic contract only", encoding="utf-8")
        self.final = self.root/"final"
        self.results = self.final/"results"
        self.results.mkdir(parents=True)
        self.manifest = self.final/"final_selection.json"
        self.audit = self.final/"synthetic_a0a_report.json"
        self.identities = []
        self.jobs = []
        self.completed = {}
        self.entries = []
        self.audit_items = []
        self.attempts = []
        for index in range(2):
            identity = {"network_kind": "standard", "mode": "minimax", "device": "cpu",
                "rules": "freestyle", "opponent": "greedy", "games": 2,
                "seed_black": 6000000 + 20000*index, "seed_white": 6100000 + 20000*index,
                "depth": 2, "nodes": 2000, "search_candidates": 8, "tactical_candidates": 12,
                "value_weight": 0.0, "sims": 400, "checkpoint_sha256": None, "source_sha256": {}}
            self.identities.append(identity)
            self.jobs.append({"name": f"synthetic_{index}", "checkpoint": None, "identity_index": index,
                "context": {"role": "contextual_reference", "recipe": "synthetic", "training_seed": None,
                    "training_run": None, "training_protocol_sha256": None, "checkpoint_sha256": None}})
        dump(self.manifest, {"status": "frozen", "selected_using": "development only",
            "jobs": self.jobs, "evaluations": self.identities, "models": {}})
        self.contract = tool.Contract(jobs=2, games_per_color=2, frozen_sha=tool.digest(self.manifest), synthetic=True)
        for index, job in enumerate(self.jobs):
            base = self.results/"jobs"/job["name"]
            identity = self.identities[index]
            games = [{"color": str(color), "game": str(n), "seed": str(seed+n), "mode": "minimax",
                "opponent": "greedy", "rules": "freestyle", "split": "test", "outcome": str((n%3)-1),
                "moves": "3", "policy_seconds": "0.003"}
                for color, seed in ((1, identity["seed_black"]), (-1, identity["seed_white"])) for n in range(2)]
            moves = [{"color": g["color"], "game": g["game"], "seed": g["seed"], "rules": "freestyle",
                "ply": str(ply), "player": str(1 if ply%2 else -1), "action": str(ply-1),
                "actor": "candidate" if str(1 if ply%2 else -1) == g["color"] else "greedy",
                "inference_seconds": "0.001", "tactical_reason": "synthetic", "search_nodes": "0"}
                for g in games for ply in range(1, 4)]
            attempts = []
            for number in range(1, 3 if index == 0 else 2):
                directory = base/f"attempt_{number:03d}"
                directory.mkdir(parents=True)
                complete = index == 1 or number == 2
                runtime = 1000 + 100*index+number
                protocol = {**identity, "pid": runtime, "started_at": f"synthetic-{number}", "output": str(directory),
                    "checkpoint": None, "split": "test", "training_performed": False, "selection_performed": False,
                    "frozen_selection": {"path": str(self.manifest), "sha256": self.contract.frozen_sha,
                        "identity": identity, "selected_using": "development only"}}
                launcher = {"pid": runtime+10000, "birth": str(runtime), "exit_code": 0 if complete else 1,
                    "command": ["python", "-u", "-m", "synthetic", "--output", str(directory)],
                    "cwd": str(self.root), "identity": identity, "context": job["context"],
                    "frozen_manifest_sha256": self.contract.frozen_sha}
                dump(directory/"protocol.json", protocol)
                dump(directory/"launcher.json", launcher)
                dump(directory/"status.json", {"state": "complete" if complete else "failed"})
                (directory/"eval.lock").write_bytes(b"x")
                (directory/"console.log").write_text("complete" if complete else
                    "Traceback (most recent call last):\n os.replace(status.json.tmp, status.json)\nPermissionError: [WinError 5] denied\n", encoding="utf-8")
                (directory/"games.csv").write_bytes(csv_bytes(games if complete else games[:1]))
                # Include an interrupted partially written second game.
                (directory/"moves.csv").write_bytes(csv_bytes(moves if complete else moves[:4]))
                if complete:
                    dump(directory/"summary.json", {"synthetic": True})
                attempts.append(directory)
            self.attempts.append(attempts)
            selected = attempts[-1]
            artifacts = {name: tool.digest(selected/name) for name in tool.ARTIFACTS}
            self.completed[job["name"]] = {"directory": str(selected), "identity": identity, "artifacts": artifacts}
            dump(base/"accepted_result.json", self.completed[job["name"]])
            self.entries.append({"directory": str(selected), **job["context"], "frozen_selection": str(self.manifest)})
            self.audit_items.append({"directory": str(selected), "passed": True, "games": 4, "split": "test",
                "identity": identity, "frozen_selection": {"verified": True, "path": str(self.manifest), "sha256": self.contract.frozen_sha},
                "artifact_sha256": artifacts})
        dump(self.results/"completed_jobs.json", self.completed)
        dump(self.results/"audit_inputs.json", {"format": "a2c_independent_audit_inputs_v1", "evaluations": self.entries})
        dump(self.results/"suite_protocol.json", {"manifest_sha256": self.contract.frozen_sha,
            "manifest_path": str(self.manifest), "jobs": self.jobs})
        dump(self.results/"suite_status.json", {"state": "frozen_evaluations_complete", "completed_jobs": 2,
            "expected_jobs": 2, "active": [], "process": {"pid": 101, "birth": "runtime-101"}})
        dump(self.final/"supervisor.launcher.json", {"pid": 100, "birth": "launcher-100"})
        (self.results/"final_suite.lock").write_bytes(b"x")
        self.write_audit()

    def write_audit(self, **extra):
        dump(self.audit, {"format": "a2c_independent_final_audit_v1", "passed": True, "errors": [],
            "allow_development": False, "tool_sha256": tool.A0A_SHA, "project_root": str(self.root),
            "input_manifest_sha256": tool.digest(self.results/"audit_inputs.json"),
            "training_or_test_execution_performed": False, "candidate_inference_recomputed": False,
            "finished_at": "synthetic-finished", "total_games": 8, "evaluations": self.audit_items, **extra})

    def run(self, **extra):
        @contextmanager
        def no_real_locks(path):
            assert Path(path).name in {"final_suite.lock", "eval.lock"}
            yield
        return tool._verify(self.root, self.final, self.audit, contract=self.contract,
            inspect_process=extra.pop("inspect_process", lambda pid: {"pid": pid, "alive": False, "birth": None}),
            acquire_lock=extra.pop("acquire_lock", no_real_locks), **extra)


class RetryTests(unittest.TestCase):
    def setUp(self):
        # All writes stay in a visualization-local TemporaryDirectory, not a run.
        self.temporary = tempfile.TemporaryDirectory(prefix="retry_non8m_", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(self.temporary.name)

    def assert_gate_refusal_without_data_read(self, expected, **kwargs):
        original = tool.shared_bytes
        sensitive = []
        def observed(path):
            if Path(path).name in tool.SENSITIVE:
                sensitive.append(str(path))
            return original(path)
        with patch.object(tool, "shared_bytes", observed):
            with self.assertRaisesRegex(tool.Refusal, expected):
                self.fixture.run(**kwargs)
        self.assertEqual([], sensitive)

    def test_complete_retry_and_trailing_move_prefix_pass(self):
        report = self.fixture.run()
        self.assertTrue(report["passed"])
        self.assertTrue(report["synthetic_test_only"])
        self.assertEqual(1, report["retry_job_count"])
        self.assertEqual(1, report["retry_jobs"][0]["failed_attempts"][0]["prefix"]["trailing_moves_after_last_game_row"])
        self.assertEqual([], report["gate"]["training_lock_paths_opened"])

    def test_running_queue_blocks_before_sensitive_reads(self):
        path = self.fixture.results/"suite_status.json"
        value = tool.read_json(path); value["state"] = "running"; dump(path, value)
        self.assert_gate_refusal_without_data_read("Queue is not complete")

    def test_active_array_blocks_before_sensitive_reads(self):
        path = self.fixture.results/"suite_status.json"
        value = tool.read_json(path); value["active"] = [{"name": "still-running"}]; dump(path, value)
        self.assert_gate_refusal_without_data_read("Queue is not complete")

    def test_live_related_process_blocks_before_sensitive_reads(self):
        self.assert_gate_refusal_without_data_read("related process still alive",
            inspect_process=lambda pid: {"pid": pid, "alive": True, "birth": "launcher-100"})

    def test_unknown_process_blocks_before_sensitive_reads(self):
        self.assert_gate_refusal_without_data_read("unknown", inspect_process=lambda pid: {"pid": pid, "alive": None, "birth": None})

    def test_archived_supervisor_runtime_alive_blocks_before_sensitive_reads(self):
        history = self.fixture.final/"recoveries"/"synthetic"/"suite_status.before.json"
        dump(history, {"state": "failed", "process": {"pid": 200, "birth": "historical-runtime"}})
        self.assert_gate_refusal_without_data_read("related process still alive",
            inspect_process=lambda pid: {"pid": pid, "alive": True, "birth": "historical-runtime"}
                if pid == 200 else {"pid": pid, "alive": False, "birth": None})

    def test_archived_supervisor_runtime_unknown_blocks_before_sensitive_reads(self):
        history = self.fixture.final/"recoveries"/"synthetic"/"suite_status.failed.json"
        dump(history, {"state": "failed", "process": {"pid": 200, "birth": "historical-runtime"}})
        self.assert_gate_refusal_without_data_read("unknown",
            inspect_process=lambda pid: {"pid": pid, "alive": None, "birth": None}
                if pid == 200 else {"pid": pid, "alive": False, "birth": None})

    def test_live_pid_known_different_birth_is_reuse(self):
        report = self.fixture.run(inspect_process=lambda pid: {"pid": pid, "alive": True, "birth": "other"}
            if pid == 100 else {"pid": pid, "alive": False, "birth": None})
        self.assertTrue(any(p["disposition"] == "PID_reused_known_different_birth" for p in report["gate"]["processes"]))

    def test_unheld_lock_gate_precedes_sensitive_reads(self):
        @contextmanager
        def held(path):
            raise tool.Refusal("Active synthetic writer lock")
            yield
        self.assert_gate_refusal_without_data_read("Active synthetic writer", acquire_lock=held)

    def test_missing_or_failed_full_audit_blocks_data_reads(self):
        self.fixture.write_audit(passed=False)
        self.assert_gate_refusal_without_data_read("Full independent final audit")

    def test_incomplete_full_audit_blocks_data_reads(self):
        self.fixture.write_audit(total_games=7)
        self.assert_gate_refusal_without_data_read("complete frozen budget")

    def test_full_audit_fingerprint_mismatch_blocks_data_reads(self):
        self.fixture.audit_items[0]["artifact_sha256"]["games.csv"] = "0"*64
        self.fixture.write_audit()
        self.assert_gate_refusal_without_data_read("fingerprints differ")

    def test_partial_action_tamper_rejected(self):
        path = self.fixture.attempts[0][0]/"moves.csv"
        rows = tool.read_rows(path); rows[0]["action"] = "79"; path.write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(tool.Refusal, "move/action/reason/node prefix differs"):
            self.fixture.run()

    def test_partial_outcome_tamper_rejected(self):
        path = self.fixture.attempts[0][0]/"games.csv"
        rows = tool.read_rows(path); rows[0]["outcome"] = "1"; path.write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(tool.Refusal, "seed/outcome/length prefix differs"):
            self.fixture.run()

    def test_timing_only_prefix_changes_allowed(self):
        for name, key in (("games.csv", "policy_seconds"), ("moves.csv", "inference_seconds")):
            path = self.fixture.attempts[0][0]/name
            rows = tool.read_rows(path); rows[0][key] = "1000.123"; path.write_bytes(csv_bytes(rows))
        self.assertTrue(self.fixture.run()["passed"])

    def test_command_budget_tamper_rejected(self):
        path = self.fixture.attempts[0][0]/"launcher.json"
        value = tool.read_json(path); value["command"] += ["--games", "1"]; dump(path, value)
        with self.assertRaisesRegex(tool.Refusal, "command/protocol differs"):
            self.fixture.run()

    def test_no_traceback_is_not_external_cause_evidence(self):
        (self.fixture.attempts[0][0]/"console.log").write_text("progress only\n", encoding="utf-8")
        with self.assertRaisesRegex(tool.Refusal, "explicit external-stop evidence"):
            self.fixture.run()

    def test_explicit_unknown_exit_metadata_is_linked(self):
        directory = self.fixture.attempts[0][0]
        (directory/"console.log").write_text("progress only\n", encoding="utf-8")
        incident = self.fixture.final/"incident.json"
        launcher = tool.read_json(directory/"launcher.json")
        protocol = tool.read_json(directory/"protocol.json")
        dump(incident, {"format": "a2c_frozen_queue_operational_recovery_v1", "observed_at_utc": "synthetic-time",
            "process_and_lock_observations": [{"directory": str(directory), "status": {"state": "evaluating"},
                "eval_lock_held": False, "recorded_launcher": {"pid": launcher["pid"], "birth": launcher["birth"]},
                "launcher_actual": {"pid": launcher["pid"], "alive": False},
                "runtime_actual": {"pid": protocol["pid"], "alive": False}}]})
        evidence = self.fixture.final/"stop_evidence.json"
        dump(evidence, {"format": "a2c_external_stop_evidence_v1", "stops": [{"directory": str(directory),
            "reason": "unexpected_process_exit_mechanism_unknown", "observed_at": "synthetic-time",
            "explanation": "Runtime absent; no claim about the termination mechanism",
            "evidence": [{"path": str(incident), "sha256": tool.digest(incident)}]}]})
        report = self.fixture.run(external_stop_evidence=evidence)
        self.assertEqual("unexpected_process_exit_mechanism_unknown_no_Python_traceback",
            report["retry_jobs"][0]["failed_attempts"][0]["failure"]["category"])

    def test_no_synthetic_budget_override_in_cli(self):
        self.assertFalse(any(action.dest in {"synthetic", "games", "jobs", "allow_development"} for action in tool.parser()._actions))

    def test_training_lock_is_forbidden(self):
        with self.assertRaisesRegex(tool.Refusal, "Forbidden lock name"):
            with tool.unheld_lock(Path(self.temporary.name)/"run.lock"):
                self.fail("Training lock was entered")

    def test_case_insensitive_sensitive_metadata_alias_refused_before_read(self):
        for name in ("Summary.json", "summary.JSON", "GaMeS.CsV", "moves.CSV"):
            with self.subTest(name=name):
                with patch.object(tool, "read_json", side_effect=AssertionError("Metadata gate opened a sensitive alias")):
                    with self.assertRaisesRegex(tool.Refusal, "never games/moves/summary"):
                        tool.safe_metadata(Path(self.temporary.name)/name)

    def test_full_audit_artifact_drift_is_rejected(self):
        path = self.fixture.attempts[1][0]/"summary.json"
        path.write_text('{"synthetic": "changed after full audit"}', encoding="utf-8")
        with self.assertRaisesRegex(tool.Refusal, "artifact changed"):
            self.fixture.run()

    def test_null_launcher_without_retirement_refused(self):
        path = self.fixture.attempts[1][0]/"launcher.json"
        value = tool.read_json(path); value["pid"] = None; dump(path, value)
        self.assert_gate_refusal_without_data_read("Null launcher PID")

    def test_retired_launcher_binds_before_after_sha_and_distinct_birth(self):
        path = self.fixture.attempts[1][0]/"launcher.json"
        before = tool.read_json(path)
        incident_dir = self.fixture.final/"recoveries"/"synthetic_retirement"
        incident_dir.mkdir(parents=True)
        backup = incident_dir/f"completed_launcher_pid{before['pid']}.before.json"
        backup.write_bytes(path.read_bytes())
        incident = incident_dir/"incident.json"
        other = {"pid": before["pid"], "birth": "different-current-birth", "image_name": "synthetic_service"}
        after = {**before, "pid": None, "historical_pid": before["pid"], "identity_retirement": {
            "evidence": str(incident), "original_birth": before["birth"], "unrelated_current_process": other}}
        dump(path, after)
        dump(incident, {"completed_launcher_identity_retirement": {"path": str(path),
            "before_sha256": tool.digest(backup), "after_sha256": tool.digest(path), "current_unrelated_process": other}})
        report = self.fixture.run(inspect_process=lambda pid: {"pid": pid, "alive": True, "birth": other["birth"]}
            if pid == before["pid"] else {"pid": pid, "alive": False, "birth": None})
        self.assertTrue(any(item["recorded"].get("retirement_verified") for item in report["gate"]["processes"]))
        document = tool.read_json(incident); document["completed_launcher_identity_retirement"]["before_sha256"] = "f"*64; dump(incident, document)
        self.assert_gate_refusal_without_data_read("before-SHA differs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
