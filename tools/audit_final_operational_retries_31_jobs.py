"""Verify operational-retry prefixes only AFTER the complete final replay audit.

This standalone standard-library tool never imports the experiment or a0a,
never opens training run.lock, and never recomputes candidate inference.
The CLI accepts only the fixed 32-job / 6,400-game frozen final contract.
It refuses all games/moves/summary reads until the queue, every recorded
process, every final/eval lock, and the complete passed a0a report pass gates.
No output or input artifact is overwritten. See the sibling handoff document.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback

sys.dont_write_bytecode = True
FORMAT = "a2c_post_final_operational_retry_prefix_audit_v1"
A0A_SHA = "a0a5268cd09f301a03a8786d381fc6fcee900a2d4398e81781e3422b2e6a73ce"
FROZEN_SHA = "4d0d0acc5a49fa604d1d773b5730ea352c08b77f935dfda592d83017f8721c7a"
IDENTITY = ("network_kind", "mode", "device", "rules", "opponent", "games",
    "seed_black", "seed_white", "depth", "nodes", "search_candidates",
    "tactical_candidates", "value_weight", "sims", "checkpoint_sha256", "source_sha256")
ARTIFACTS = ("protocol.json", "summary.json", "status.json", "games.csv", "moves.csv")
SENSITIVE = {"games.csv", "moves.csv", "summary.json"}


class Refusal(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def shared_bytes(path):
    """Short read handles with Windows FILE_SHARE_DELETE, including JSON."""
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


def read_json(path):
    return json.loads(shared_bytes(path).decode("utf-8-sig"))


def digest(path):
    return hashlib.sha256(shared_bytes(path)).hexdigest()


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve(path, base):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def within(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def process_identity(pid):
    """Independent OS query; permission errors mean unknown, never stopped."""
    require(type(pid) is int and pid > 0, "Missing/malformed recorded PID")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 5:
                # Protected unrelated services may deny OpenProcess. This
                # read-only system enumeration can still establish PID birth.
                observed = enumerate_windows_identity(pid)
                if observed is not None:
                    return observed
            return {"pid": pid, "alive": False if error == 87 else None,
                    "birth": None, "query_error": error}
        try:
            code = wintypes.DWORD()
            birth, end, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or not kernel.GetProcessTimes(
                handle, ctypes.byref(birth), ctypes.byref(end), ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return {"pid": pid, "alive": None, "birth": None, "query_error": ctypes.get_last_error()}
            return {"pid": pid, "alive": code.value == 259,
                "birth": str((birth.dwHighDateTime << 32) | birth.dwLowDateTime),
                "exit_code": None if code.value == 259 else code.value}
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        fields = stat.read_text().rsplit(")", 1)[1].split() if stat.exists() else []
        return {"pid": pid, "alive": not fields or fields[0] != "Z",
                "birth": fields[19] if fields else None}
    except ProcessLookupError:
        return {"pid": pid, "alive": False, "birth": None}
    except PermissionError:
        return {"pid": pid, "alive": None, "birth": None}


def enumerate_windows_identity(pid):
    """Read-only x64 SYSTEM_PROCESS_INFORMATION; unknown on unsupported layout."""
    import ctypes
    from ctypes import wintypes
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        return None
    query = ctypes.WinDLL("ntdll").NtQuerySystemInformation
    query.argtypes = (wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG))
    query.restype = wintypes.LONG
    size = 262144
    for _ in range(8):
        data, needed = ctypes.create_string_buffer(size), wintypes.ULONG()
        result = query(5, data, size, ctypes.byref(needed))
        if result == 0:
            offset = 0
            while offset + 88 <= size:
                base = ctypes.addressof(data) + offset
                next_entry = ctypes.c_uint32.from_address(base).value
                found_pid = ctypes.c_uint64.from_address(base + 80).value
                if found_pid == pid:
                    birth = ctypes.c_int64.from_address(base + 32).value
                    length = ctypes.c_uint16.from_address(base + 56).value
                    pointer = ctypes.c_void_p.from_address(base + 64).value
                    return {"pid": pid, "alive": True, "birth": str(birth) if birth > 0 else None,
                        "image_name": ctypes.wstring_at(pointer, length//2) if pointer and length else "",
                        "method": "NtQuerySystemInformation/SystemProcessInformation/read-only"}
                if not next_entry:
                    return {"pid": pid, "alive": False, "birth": None,
                        "method": "NtQuerySystemInformation/SystemProcessInformation/read-only"}
                if next_entry < 88 or offset + next_entry >= size:
                    return None
                offset += next_entry
            return None
        if result & 0xffffffff != 0xc0000004:
            return None
        size = max(size*2, needed.value+65536)
    return None


def stopped(record, actual, label):
    require(type(record.get("pid")) is int and record["pid"] > 0, f"{label}: missing PID")
    require(actual.get("pid") == record["pid"], f"{label}: OS query PID differs")
    if actual.get("alive") is False:
        return "exited"
    if actual.get("alive") is True and record.get("birth") and actual.get("birth"):
        require(str(record["birth"]) != str(actual["birth"]), f"{label}: related process still alive")
        return "PID_reused_known_different_birth"
    raise Refusal(f"{label}: process liveness/birth is unknown; refuse conservatively")


@contextmanager
def unheld_lock(path):
    """Only final_suite.lock or eval.lock; never touch training locks."""
    path = Path(path)
    require(path.name in {"final_suite.lock", "eval.lock"}, "Forbidden lock name")
    require(path.is_file() and path.stat().st_size > 0, f"Missing/malformed OS lock: {path}")
    with path.open("rb") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise Refusal(f"Active final/eval writer lock: {path}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class Contract:
    jobs: int = 32
    games_per_color: int = 100
    frozen_sha: str = FROZEN_SHA
    auditor_sha: str = A0A_SHA
    synthetic: bool = False


def metadata_records(value, origin):
    """Find recorded historical process identities in explicit metadata only."""
    if isinstance(value, dict):
        if type(value.get("pid")) is int and value["pid"] > 0:
            yield {"pid": value["pid"], "birth": value.get("birth"), "origin": origin}
        for key, child in value.items():
            yield from metadata_records(child, f"{origin}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from metadata_records(child, f"{origin}[{index}]")


def safe_metadata(path):
    path = Path(path).resolve()
    require(path.name.casefold() not in SENSITIVE and path.suffix.lower() == ".json",
            "Process/recovery evidence must be JSON metadata, never games/moves/summary")
    return read_json(path)


def identity_from(protocol):
    require(set(IDENTITY) <= set(protocol), "Attempt lacks full frozen identity")
    return {key: protocol[key] for key in IDENTITY}


def normalized_command(command):
    require(isinstance(command, list) and command and all(isinstance(x, str) for x in command),
            "Malformed launcher command")
    require(command.count("--output") == 1, "Launcher has ambiguous output option")
    result = list(command)
    index = result.index("--output")
    require(index + 1 < len(result), "Launcher output lacks value")
    result[index + 1] = "<operational_attempt_directory>"
    return result


def read_rows(path):
    return list(csv.DictReader(io.StringIO(shared_bytes(path).decode("utf-8-sig"))))


def compare_prefix(partial_games, partial_moves, accepted_games, accepted_moves):
    require(len(partial_games) < len(accepted_games), "Failed attempt is not a partial game budget")
    require(len(partial_moves) <= len(accepted_moves), "Failed moves exceed complete budget")
    def omit(rows, key):
        return [{k: v for k, v in row.items() if k != key} for row in rows]
    require(omit(partial_games, "policy_seconds") == omit(accepted_games[:len(partial_games)], "policy_seconds"),
            "Failed completed-game seed/outcome/length prefix differs")
    require(omit(partial_moves, "inference_seconds") == omit(accepted_moves[:len(partial_moves)], "inference_seconds"),
            "Failed move/action/reason/node prefix differs")
    covered = sum(int(row["moves"]) for row in partial_games)
    require(covered <= len(partial_moves), "Failed complete game lacks recorded moves")
    extra = len(partial_moves) - covered
    if extra:
        require(len(partial_games) < len(accepted_games), "Trailing moves have no next accepted game")
        require(extra <= int(accepted_games[len(partial_games)]["moves"]), "Trailing moves span unreported games")
    return {"completed_games_compared": len(partial_games), "all_moves_compared": len(partial_moves),
        "trailing_moves_after_last_game_row": extra,
        "all_completed_game_seeds_outcomes_lengths_identical": True,
        "all_recorded_actions_reasons_nodes_identical": True,
        "ignored_fields": ["policy_seconds", "inference_seconds"]}


def failure_reason(directory, launcher, protocol, external):
    log_path = directory / "console.log"
    log = shared_bytes(log_path).decode("utf-8", errors="replace")
    if "PermissionError: [WinError 5]" in log and "status.json.tmp" in log and "os.replace" in log:
        return {"category": "Windows_status_atomic_replace_WinError5",
            "exception_recorded": True, "external_holder_identified": False,
            "console_sha256": digest(log_path)}
    entry = external.get(str(directory.resolve()))
    require(entry is not None, "Unclassified failure: explicit external-stop evidence is required")
    require("Traceback (most recent call last)" not in log and not re.search(r"^\w*(?:Error|Exception):", log, re.M),
            "External-stop classification conflicts with a Python exception log")
    require(entry.get("reason") == "unexpected_process_exit_mechanism_unknown" and bool(entry.get("explanation"))
            and isinstance(entry.get("observed_at"), str), "Malformed external-stop explanation")
    require(launcher.get("exit_code") not in (None, 0), "External-stop launcher has no recorded nonzero/stopped exit")
    evidence = entry.get("evidence")
    require(isinstance(evidence, list) and evidence, "External stop lacks immutable supporting metadata")
    verified = []
    stopped_observations = []
    for item in evidence:
        path = Path(item["path"]).resolve()
        require(path.name.casefold() not in SENSITIVE and path.suffix.lower() in {".json", ".md", ".log"},
                "Stop evidence must be control metadata/log/narrative")
        require(digest(path) == item["sha256"], "External-stop supporting evidence SHA differs")
        verified.append({"path": str(path), "sha256": item["sha256"]})
        if path.suffix.lower() == ".json":
            document = safe_metadata(path)
            if document.get("format") == "a2c_frozen_queue_operational_recovery_v1":
                for observation in document.get("process_and_lock_observations", []):
                    if resolve(observation.get("directory", ""), path.parent) != directory.resolve():
                        continue
                    require(document.get("observed_at_utc") == entry["observed_at"]
                        and observation.get("eval_lock_held") is False
                        and observation.get("status", {}).get("state") != "complete",
                        "Recovery observations lack partial status/empty eval lock/time binding")
                    recorded = observation.get("recorded_launcher", {})
                    expected_pid = launcher.get("pid", launcher.get("historical_pid"))
                    require(recorded.get("pid") == expected_pid
                        and recorded.get("birth") == launcher.get("birth"), "Recovery launcher identity differs")
                    launcher_disposition = stopped(recorded, observation.get("launcher_actual", {}), "recovery launcher")
                    runtime_actual = observation.get("runtime_actual", {})
                    require(runtime_actual.get("pid") == protocol["pid"], "Recovery runtime PID differs")
                    runtime_record = {"pid": protocol["pid"], "birth": None}
                    for active in document.get("prior_state", {}).get("active", []):
                        if resolve(active.get("directory", ""), path.parent) == directory.resolve():
                            runtime_record = active.get("runtime_process", runtime_record)
                    runtime_disposition = stopped(runtime_record, runtime_actual, "recovery runtime")
                    stopped_observations.append({"evidence_path": str(path), "observed_at": entry["observed_at"],
                        "launcher_disposition": launcher_disposition, "runtime_disposition": runtime_disposition,
                        "eval_lock_was_unheld": True})
    require(stopped_observations, "Unexpected-exit evidence lacks exact stopped runtime/launcher and empty-lock observations")
    return {"category": "unexpected_process_exit_mechanism_unknown_no_Python_traceback",
        "basis": "explicit recovery observations of stopped identities/empty lock; termination mechanism remains unknown",
        "recorded_exit_code": launcher["exit_code"], "observed_at": entry["observed_at"],
        "explanation": entry["explanation"], "evidence": verified,
        "recovery_process_observations": stopped_observations, "console_sha256": digest(log_path)}


def launcher_record(launcher, path):
    """Retired reused PID still requires immutable before/after birth evidence."""
    if launcher.get("pid") is not None:
        require(type(launcher["pid"]) is int and launcher["pid"] > 0, "Malformed launcher PID")
        return {"pid": launcher["pid"], "birth": launcher.get("birth"), "origin": str(path)}, []
    retired = launcher.get("identity_retirement")
    require(isinstance(retired, dict) and type(launcher.get("historical_pid")) is int,
            "Null launcher PID lacks explicit identity retirement")
    incident_path = Path(retired["evidence"]).resolve()
    incident = safe_metadata(incident_path)
    assertion = incident.get("completed_launcher_identity_retirement", {})
    require(resolve(assertion["path"], incident_path.parent) == Path(path).resolve()
        and assertion.get("after_sha256") == digest(path), "Retired launcher after-SHA/path differs")
    before_path = incident_path.parent / f"completed_launcher_pid{launcher['historical_pid']}.before.json"
    require(digest(before_path) == assertion.get("before_sha256"), "Retired launcher before-SHA differs")
    before = safe_metadata(before_path)
    reconstructed = {k: v for k, v in launcher.items() if k not in {"historical_pid", "identity_retirement"}}
    reconstructed["pid"] = launcher["historical_pid"]
    require(before == reconstructed and before.get("exit_code") == 0,
            "Retirement changed command/context/history or lacks successful completed exit")
    original = retired.get("original_birth")
    unrelated = retired.get("unrelated_current_process", {})
    require(original and str(before.get("birth")) == str(original) and launcher.get("birth") == original
        and unrelated == assertion.get("current_unrelated_process")
        and unrelated.get("pid") == before["pid"] and unrelated.get("birth")
        and str(unrelated["birth"]) != str(original), "Retirement lacks proven distinct process births")
    return {"pid": before["pid"], "birth": original, "origin": str(path),
            "retirement_verified": True, "before_sha256": assertion["before_sha256"],
            "after_sha256": assertion["after_sha256"]}, [incident_path, before_path]


def _verify(project_root, final_dir, final_audit, process_evidence=(), external_stop_evidence=None,
            contract=Contract(), inspect_process=process_identity, acquire_lock=unheld_lock, delivery_scope=None):
    """Private test seam; CLI always supplies the immutable real Contract()."""
    project_root, final_dir, final_audit = (Path(p).resolve() for p in (project_root, final_dir, final_audit))
    require(within(final_dir, project_root), "Final directory must stay inside project")
    if contract.synthetic:
        require((project_root / "SYNTHETIC_NONRESERVED_FIXTURE").is_file(), "Synthetic test marker missing")
    manifest_path, result_dir = final_dir / "final_selection.json", final_dir / "results"
    manifest = read_json(manifest_path)
    require(digest(manifest_path) == contract.frozen_sha, "Frozen manifest SHA differs")
    require(manifest.get("status") == "frozen" and manifest.get("selected_using") == "development only",
            "Selection is not the development-only frozen selection")
    original_jobs, evaluations = manifest.get("jobs"), manifest.get("evaluations")
    scope_contract = None
    if delivery_scope is not None:
        from a2c_user_delivery_scope import load_scope
        scope_contract = load_scope(project_root, delivery_scope)
    jobs = scope_contract["jobs"] if scope_contract else original_jobs
    require(isinstance(jobs, list) and isinstance(evaluations, list)
            and len(jobs) == contract.jobs and len(evaluations) == (32 if scope_contract else contract.jobs), "Frozen final job count differs")
    require(len({job["name"] for job in jobs}) == contract.jobs
            and len({job["identity_index"] for job in jobs}) == contract.jobs, "Duplicate/missing frozen jobs")
    names = {job["name"] for job in jobs}
    for row in evaluations:
        require(set(row) == set(IDENTITY) and row["games"] == contract.games_per_color, "Frozen identity/budget differs")
        if not contract.synthetic:
            seeds = (8100000, 8200000) if row["rules"] == "freestyle" else (8300000, 8400000)
            require(row["rules"] in {"freestyle", "renju"} and (row["seed_black"], row["seed_white"]) == seeds,
                    "Reserved final seed protocol differs")
        else:
            require(all(seed < 8000000 for seed in (row["seed_black"], row["seed_white"])), "Synthetic fixtures use reserved seeds")
    suite_path = result_dir / "suite_status.json"
    suite = read_json(suite_path)
    if scope_contract:
        delivery = read_json(final_dir / "delivery_status_31_jobs.json")
        require(delivery.get("state") in {"included_evaluations_complete_audit_pending", "included_evaluations_complete_audit_passed"}
                and delivery.get("completed_jobs") == delivery.get("included_jobs") == 31
                and delivery.get("included_games") == 6200
                and delivery.get("deferred_jobs") == ["alphazero_width128_alphazero_renju_greedy"],
                "User-scoped delivery is not complete")
    else:
        require(suite.get("state") == "frozen_evaluations_complete"
                and suite.get("completed_jobs") == suite.get("expected_jobs") == contract.jobs
                and suite.get("active", []) == [] and suite.get("pending_jobs", 0) == 0,
                "Queue is not complete with all jobs and no active/pending entries")
    completed_path = result_dir / "completed_jobs.json"
    completed = read_json(completed_path)
    require(set(completed) == names, "Completed jobs differ from frozen jobs")
    inputs_path = final_dir / "audit_inputs_31_jobs.json" if scope_contract else result_dir / "audit_inputs.json"
    inputs = read_json(inputs_path)
    require(inputs.get("format") == "a2c_independent_audit_inputs_v1"
            and len(inputs.get("evaluations", [])) == contract.jobs, "Missing full final audit inputs")
    protocol_path = result_dir / "suite_protocol.json"
    suite_protocol = read_json(protocol_path)
    require(suite_protocol.get("manifest_sha256") == contract.frozen_sha
            and resolve(suite_protocol["manifest_path"], project_root) == manifest_path
            and suite_protocol.get("jobs") == original_jobs, "Suite protocol/frozen jobs differ")

    # Metadata-only inventory. No games/moves/summary opens or hashes yet.
    control_paths = {suite_path, final_dir / "supervisor.launcher.json", completed_path,
        inputs_path, protocol_path, manifest_path}
    control_paths.update(final_dir.glob("supervisor*.launcher.json"))
    for recovery in (final_dir / "recoveries").glob("**/*supervisor*launcher*.json"):
        control_paths.add(recovery)
    control_paths.update((final_dir / "recoveries").glob("**/*suite_status*.json"))
    attempt_meta = {}
    lock_paths = [result_dir / "final_suite.lock"]
    if scope_contract:
        control_paths.update((scope_contract["scope_path"], scope_contract["user_request_path"], final_dir / "delivery_status_31_jobs.json"))
    for job in original_jobs:
        base = result_dir / "jobs" / job["name"]
        if job["name"] in names: control_paths.add(base / "accepted_result.json")
        attempts = sorted(p for p in base.glob("attempt_*") if p.is_dir())
        require(attempts and len(attempts) <= 3, "Missing attempt or retry budget exceeded")
        require([p.name for p in attempts] == [f"attempt_{i:03d}" for i in range(1, len(attempts)+1)],
                "Attempt numbering is not contiguous")
        for directory in attempts:
            protocol = read_json(directory / "protocol.json")
            launcher = read_json(directory / "launcher.json")
            status = read_json(directory / "status.json")
            attempt_meta[str(directory)] = {"protocol": protocol, "launcher": launcher, "status": status}
            control_paths.update((directory / "launcher.json", directory / "protocol.json", directory / "status.json"))
            lock_paths.append(directory / "eval.lock")
    explicit_evidence = [Path(p).resolve() for p in process_evidence]
    control_paths.update(explicit_evidence)
    require(all(p.is_file() for p in control_paths), "Missing lifecycle control evidence")
    controls = {str(path): safe_metadata(path) for path in sorted(control_paths)}
    records = []
    for path, value in list(controls.items()):
        source_path = Path(path)
        control_name = source_path.name.casefold()
        if control_name == "launcher.json" or "supervisor" in control_name and "launcher" in control_name:
            record, retirement_files = launcher_record(value, source_path)
            records.append(record)
            for evidence_path in retirement_files:
                control_paths.add(evidence_path)
                controls[str(evidence_path)] = safe_metadata(evidence_path)
        elif "suite_status" in control_name:
            require(isinstance(value.get("process"), dict), "Supervisor status lacks actual runtime identity")
            records.append({**value["process"], "origin": str(source_path)+".process"})
    for directory, meta in attempt_meta.items():
        require(type(meta["protocol"].get("pid")) is int, "Attempt protocol lacks runtime PID")
        records.append({"pid": meta["protocol"]["pid"], "birth": None, "origin": directory+"/protocol.json.pid"})
    for path in explicit_evidence:
        value = controls[str(path)]
        if isinstance(value.get("prior_state", {}).get("process"), dict):
            records.append({**value["prior_state"]["process"], "origin": str(path)+".prior_state.process"})
        elif isinstance(value.get("process"), dict):
            records.append({**value["process"], "origin": str(path)+".process"})
    before_controls = {str(path): digest(path) for path in sorted(control_paths)}
    require(any(r["pid"] == suite.get("process", {}).get("pid") and r.get("birth") for r in records),
            "Final supervisor runtime lacks recorded birth identity")
    births = {}
    for record in [r for path, value in controls.items() for r in metadata_records(value, path)]:
        if record.get("birth"):
            births.setdefault(record["pid"], set()).add(str(record["birth"]))
    observations = []
    for record in records:
        # A bare runtime PID may be augmented only by a unique historical birth
        # for that same PID in the explicitly inventoried metadata.
        if not record.get("birth") and len(births.get(record["pid"], set())) == 1:
            record = {**record, "birth": next(iter(births[record["pid"]])), "birth_from_explicit_metadata": True}
        actual = inspect_process(record["pid"])
        observations.append({"recorded": record, "actual": actual,
            "disposition": stopped(record, actual, record["origin"])})

    with ExitStack() as stack:
        for path in lock_paths:
            stack.enter_context(acquire_lock(path))
        audit = read_json(final_audit)
        audit_sha = digest(final_audit)
        require(audit.get("format") == "a2c_independent_final_audit_v1" and audit.get("passed") is True
                and audit.get("allow_development") is False and audit.get("errors") == []
                and audit.get("tool_sha256") == contract.auditor_sha
                and resolve(audit["project_root"], project_root) == project_root
                and audit.get("input_manifest_sha256") == digest(inputs_path)
                and audit.get("training_or_test_execution_performed") is False
                and audit.get("candidate_inference_recomputed") is False
                and isinstance(audit.get("finished_at"), str), "Full independent final audit is missing/not passed/not exact")
        audited = audit.get("evaluations", [])
        expected_games = 2 * contract.games_per_color
        require(len(audited) == contract.jobs and audit.get("total_games") == expected_games * contract.jobs,
                "Full independent audit does not cover complete frozen budget")
        by_dir = {str(resolve(item["directory"], project_root)): item for item in audited}
        require(len(by_dir) == contract.jobs, "Duplicate directories in full independent audit")
        accepted_dirs = {str(resolve(completed[name]["directory"], project_root)) for name in names}
        require(set(by_dir) == accepted_dirs
                and {str(resolve(x["directory"], inputs_path.parent)) for x in inputs["evaluations"]} == accepted_dirs,
                "Accepted/audit input/audited directory sets differ")
        # Complete every metadata identity/fingerprint cross-link before opening
        # even one games/moves/summary artifact (including SHA reads).
        for job in jobs:
            identity = evaluations[job["identity_index"]]
            selected = resolve(completed[job["name"]]["directory"], project_root)
            require(selected.parent == result_dir / "jobs" / job["name"], "Accepted directory escapes its job")
            item = by_dir[str(selected)]
            accepted = read_json(selected.parent / "accepted_result.json")
            require(accepted == completed[job["name"]], "Per-job accepted result differs from queue completion")
            require(item.get("passed") is True and item.get("games") == expected_games and item.get("split") == "test"
                    and item.get("identity") == identity and item.get("frozen_selection", {}).get("verified") is True
                    and item["frozen_selection"].get("sha256") == contract.frozen_sha
                    and resolve(item["frozen_selection"]["path"], project_root) == manifest_path,
                    "Full-audit item lacks exact completed final identity")
            require(set(item.get("artifact_sha256", {})) == set(ARTIFACTS)
                    and accepted.get("artifacts") == item["artifact_sha256"], "Accepted/a0a artifact fingerprints differ")
            require(accepted.get("identity") == identity, "Accepted frozen identity differs")
        gate = {"all_checks_passed_before_any_game_move_summary_read": True,
            "queue_jobs_complete": contract.jobs, "total_full_audit_games": audit["total_games"],
            "final_and_eval_locks_held_uncontended_for_entire_read": len(lock_paths),
            "training_lock_paths_opened": [], "processes": observations,
            "final_audit": {"path": str(final_audit), "sha256": audit_sha, "tool_sha256": contract.auditor_sha}}

        # Data access begins only here, after the complete safety/final-audit gate.
        for directory, item in by_dir.items():
            require(item["artifact_sha256"] == {name: digest(Path(directory)/name) for name in ARTIFACTS},
                    "Passed full-audit artifact changed before retry audit")
        external = {}
        external_sha = None
        external_paths_sha = {}
        if external_stop_evidence:
            external_path = Path(external_stop_evidence).resolve()
            external_doc = safe_metadata(external_path)
            external_sha = digest(external_path)
            external_paths_sha[str(external_path)] = external_sha
            require(external_doc.get("format") == "a2c_external_stop_evidence_v1", "Unsupported external-stop evidence")
            for entry in external_doc.get("stops", []):
                key = str(resolve(entry["directory"], external_path.parent))
                require(key not in external, "Duplicate external-stop entry")
                external[key] = entry
                for evidence in entry.get("evidence", []):
                    external_paths_sha[str(Path(evidence["path"]).resolve())] = evidence["sha256"]
        reports = []
        for job in jobs:
            base = result_dir / "jobs" / job["name"]
            attempts = sorted(p for p in base.glob("attempt_*") if p.is_dir())
            if len(attempts) == 1:
                continue
            selected = resolve(completed[job["name"]]["directory"], project_root)
            statuses = [attempt_meta[str(p)]["status"].get("state") for p in attempts]
            complete = [p for p, state in zip(attempts, statuses) if state == "complete"]
            require(complete == [selected] and selected == attempts[-1],
                    "Accepted attempt is not the unique first complete attempt with no later attempts")
            identity = evaluations[job["identity_index"]]
            accepted_games, accepted_moves = read_rows(selected/"games.csv"), read_rows(selected/"moves.csv")
            require(len(accepted_games) == expected_games, "Accepted game row count differs")
            frozen_protocol = None
            frozen_command = None
            evidence_before = {}
            for directory in attempts:
                evidence_before[str(directory)] = {p.name: digest(p) for p in directory.iterdir()
                    if p.is_file() and p.name != "eval.lock"}
                meta = attempt_meta[str(directory)]
                protocol, launcher = meta["protocol"], meta["launcher"]
                require(identity_from(protocol) == identity, "Retry frozen identity differs")
                frozen = protocol.get("frozen_selection", {})
                require(protocol.get("split") == "test" and frozen.get("sha256") == contract.frozen_sha
                    and resolve(frozen["path"], project_root) == manifest_path and frozen.get("identity") == identity
                    and frozen.get("selected_using") == "development only", "Retry frozen manifest declaration differs")
                require(launcher.get("identity") == identity and launcher.get("context") == job["context"]
                    and launcher.get("frozen_manifest_sha256") == contract.frozen_sha
                    and resolve(launcher["cwd"], project_root) == project_root, "Retry launcher identity/context differs")
                require(resolve(protocol["output"], project_root) == directory
                    and resolve(launcher["command"][launcher["command"].index("--output")+1], project_root) == directory,
                    "Retry output path differs")
                current_protocol = {k: v for k, v in protocol.items() if k not in {"output", "pid", "started_at"}}
                current_command = normalized_command(launcher["command"])
                if frozen_protocol is None:
                    frozen_protocol, frozen_command = current_protocol, current_command
                require(current_protocol == frozen_protocol and current_command == frozen_command,
                        "Retry command/protocol differs beyond operational metadata")
                require(not protocol.get("training_performed") and not protocol.get("selection_performed"),
                        "Retry performed training or selection")
                if identity["checkpoint_sha256"] is not None:
                    require(digest(resolve(protocol["checkpoint"], project_root)) == identity["checkpoint_sha256"],
                            "Retry checkpoint SHA changed")
                for relative, sha in identity["source_sha256"].items():
                    path = resolve(relative, project_root)
                    require(within(path, project_root) and digest(path) == sha, "Retry source SHA changed")
            failures = []
            for directory in attempts[:-1]:
                require(not (directory/"summary.json").exists(), "Failed attempt unexpectedly has a summary; manual boundary review required")
                reason = failure_reason(directory, attempt_meta[str(directory)]["launcher"],
                    attempt_meta[str(directory)]["protocol"], external)
                games = read_rows(directory/"games.csv") if (directory/"games.csv").is_file() else []
                moves = read_rows(directory/"moves.csv") if (directory/"moves.csv").is_file() else []
                comparison = compare_prefix(games, moves, accepted_games, accepted_moves)
                failures.append({"directory": str(directory), "failure": reason, "prefix": comparison,
                    "partial_rows_counted_in_final_total": 0})
            require(evidence_before == {str(directory): {p.name: digest(p) for p in directory.iterdir()
                    if p.is_file() and p.name != "eval.lock"} for directory in attempts}, "Retry evidence changed during read")
            reports.append({"job": job["name"], "identity": identity, "failed_attempts": failures,
                "accepted_attempt": str(selected), "accepted_games": expected_games,
                "only_first_complete_attempt_counted": True, "partial_games_merged": False,
                "accepted_artifacts_match_full_a0a_audit": True, "evidence_sha256": evidence_before})
        require(not (set(external) - {x["directory"] for r in reports for x in r["failed_attempts"]}),
                "External-stop evidence names an unverified/nonfailed attempt")
        require(before_controls == {path: digest(path) for path in before_controls}, "Lifecycle metadata changed during audit")
        require(external_paths_sha == {path: digest(path) for path in external_paths_sha},
                "Unexpected-exit evidence changed during audit")
        require(digest(final_audit) == audit_sha and digest(manifest_path) == contract.frozen_sha,
                "Full audit/frozen manifest changed during retry audit")
        return {"format": FORMAT, "passed": True, "synthetic_test_only": contract.synthetic,
            "started_after_complete_final_audit": True, "gate": gate, "retry_jobs": reports,
            "retry_job_count": len(reports), "failed_attempt_count": sum(len(r["failed_attempts"]) for r in reports),
            "frozen_manifest_sha256": contract.frozen_sha, "external_stop_evidence_sha256": external_sha,
            "delivery_scope_sha256": scope_contract["scope_sha256"] if scope_contract else None,
            "deferred_jobs_not_counted": scope_contract["scope"]["deferred_jobs"] if scope_contract else [],
            "lifecycle_metadata_sha256": before_controls, "external_evidence_sha256": external_paths_sha,
            "candidate_inference_recomputed": False, "policy_or_wrapper_selection_performed": False,
            "full_game_legality_and_RNG_audit_reused_from": str(final_audit),
            "limitations": ["This verifies retry identity and recorded prefixes; complete accepted-game legality/RNG/statistics remain the linked a0a audit's responsibility.",
                "Unexpected-exit classification is based on explicit recovery observations. Its mechanism remains unknown; absence of a Python traceback alone does not identify a cause.",
                "Timing fields alone are excluded from prefix equality; moves, outcomes, reasons and node declarations must match exactly."]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--delivery-scope", required=True, help="SHA-bound explicit human-requested31-job scope")
    result.add_argument("--project-root", required=True)
    result.add_argument("--final-directory", required=True)
    result.add_argument("--final-audit", required=True)
    result.add_argument("--process-evidence", action="append", default=[],
        help="Explicit historical process-identity JSON; repeat if needed for known PID reuse")
    result.add_argument("--external-stop-evidence", help="Explicit immutable external-stop evidence JSON")
    result.add_argument("--output", required=True, help="New JSON output outside all input attempt directories")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    output, final_dir = Path(args.output).resolve(), Path(args.final_directory).resolve()
    require(not output.exists(), "Output exists; never overwrite evidence")
    require(not within(output, final_dir/"results"), "Output cannot alter final queue input directory")
    started_at = now()
    report = _verify(args.project_root, final_dir, args.final_audit, args.process_evidence, args.external_stop_evidence,
        contract=Contract(jobs=31), delivery_scope=args.delivery_scope)
    report.update(started_at=started_at, finished_at=now(), tool_sha256=digest(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"passed": True, "retry_jobs": report["retry_job_count"], "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Retry audit refused: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
