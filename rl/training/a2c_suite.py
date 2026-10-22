"""Independent resumable A2C control queue; preserves the existing experiment suite."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rl.evaluation.report_a2c import publish, write_atomic
from rl.training.common import exclusive_run

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 22, 33)


def stage_status(directory):
    path = directory / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_status(out, state, **details):
    payload = {"state": state, "suite_pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    write_atomic(out / "a2c_suite_status.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def writer_active(directory):
    try:
        with exclusive_run(directory / "run.lock"):
            return False
    except RuntimeError:
        return True


def update_report(out):
    try:
        publish(out)
    except (OSError, ValueError, RuntimeError) as exc:
        # A report collision must not kill a healthy long-running trainer.
        print(f"A2C report postponed: {exc}", flush=True)


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        publish(out)
        return
    with exclusive_run(out / "a2c_suite.lock"):
        update_report(out)
        for seed in SEEDS:
            name = f"a2c4_seed{seed}"
            directory = out / name
            directory.mkdir(parents=True, exist_ok=True)
            # A suite may have died while its trainer survived. Wait for the
            # trainer's OS-held lock rather than creating a second writer.
            while writer_active(directory):
                previous = stage_status(directory)
                save_status(out, "waiting_existing_writer", stage=name, seed=seed, child_pid=previous.get("pid", previous.get("trainer_pid")), episode=previous.get("episode"))
                update_report(out)
                time.sleep(30)
            if stage_status(directory).get("state") == "complete":
                update_report(out)
                continue
            command = [sys.executable, "-u", "-m", "rl.training.local_a2c", "--output", str(directory), "--seed", str(seed), "--episodes", "30000", "--device", args.device, "--resume"]
            save_status(out, "starting", stage=name, seed=seed, child_pid=None, command=command)
            try:
                with (directory / "console.log").open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    save_status(out, "running", stage=name, seed=seed, child_pid=process.pid, command=command)
                    while process.poll() is None:
                        time.sleep(30)
                        update_report(out)
                    exit_code = process.returncode
                state = stage_status(directory)
                if exit_code or state.get("state") != "complete":
                    save_status(out, "failed", stage=name, seed=seed, child_pid=process.pid, child_running=False, exit_code=exit_code, trainer_state=state.get("state"), command=command)
                    update_report(out)
                    raise SystemExit(exit_code or 1)
                update_report(out)
            except OSError as exc:
                save_status(out, "failed", stage=name, seed=seed, child_pid=None, error=str(exc), command=command)
                update_report(out)
                raise
        save_status(out, "complete", child_pid=None, completed_seeds=list(SEEDS))
        update_report(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/production")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--report-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
