"""CPU-only development selection while isolated confirmation training runs.

Evaluate every immutable 0/5k/.../30k candidate with policy and guarded wrappers,
40 Greedy games per color. Exact v2 protocol matches may reuse prior development
cases. Reuse never creates another training seed. Final test seeds are forbidden.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from rl.training.common import atomic_json, exclusive_run
from rl.training.a2c_confirmation_suite import (ROOT, digest, lock_held, now,
    process_identity, read_json, resolve, same_process)
from rl.evaluation import a2c_research_eval_v2 as evaluator

FORMAT = "a2c_confirmation_development_selection_v1"
MODES = ("policy", "guarded")
RECIPES = ("residual_tactical10_seed11", "residual_tactical10_seed22",
           "residual_tactical10_seed33", "flat_tactical10_seed22")
IDENTITY_KEYS = ("network_kind", "mode", "device", "rules", "opponent", "split",
    "games", "seed_black", "seed_white", "depth", "nodes", "search_candidates",
    "tactical_candidates", "value_weight", "sims", "checkpoint_sha256", "source_sha256")


def sources():
    args = evaluator.parser().parse_args(["--output", "unused", "--mode", "guarded"])
    result = evaluator.source_hashes(args)
    for name in ("rl/evaluation/a2c_confirmation_select.py",
                 "rl/training/a2c_confirmation_suite.py", "rl/training/a2c_research_suite.py"):
        result[name] = digest(ROOT / name)
    return result


def case_plan(confirmation):
    confirmation = resolve(confirmation)
    return [{"key": f"{recipe}/episode_{episode:06d}/{mode}", "recipe": recipe,
             "training_seed": int(recipe.rsplit("seed", 1)[1]),
             "episode": episode, "mode": mode,
             "checkpoint": str(confirmation / recipe / "train/candidates" / f"episode_{episode:06d}.pt")}
            for recipe in RECIPES for episode in range(0, 30001, 5000) for mode in MODES]


def command(case, destination):
    return [sys.executable, "-u", "-m", "rl.evaluation.a2c_research_eval_v2",
        "--checkpoint", case["checkpoint"], "--network-kind", "research", "--mode", case["mode"],
        "--device", "cpu", "--rules", "freestyle", "--opponent", "greedy", "--split", "validation",
        "--games", "40", "--seed-black", "3100000", "--seed-white", "3200000",
        "--output", str(destination)]


def identity(case):
    args = evaluator.parser().parse_args(command(case, "unused")[4:])
    protocol = vars(args)
    protocol.update(checkpoint_sha256=digest(case["checkpoint"]), source_sha256=evaluator.source_hashes(args))
    return {key: protocol[key] for key in IDENTITY_KEYS}


def validate_summary(directory, expected):
    directory = Path(directory)
    if lock_held(directory / "eval.lock") or read_json(directory / "status.json").get("state") != "complete":
        raise ValueError("Evaluation is not completed and unlocked")
    summary = read_json(directory / "summary.json")
    protocol = summary.get("protocol", {})
    if protocol != read_json(directory / "protocol.json"):
        raise ValueError("Summary and protocol file disagree")
    if {key: protocol.get(key) for key in IDENTITY_KEYS} != expected:
        raise ValueError("Full evaluator identity mismatch")
    if not summary.get("sources_and_weights_unchanged") or summary.get("total_games") != 80:
        raise ValueError("Evaluation lacks source guard or 80 complete games")
    if protocol.get("training_performed") or protocol.get("selection_performed"):
        raise ValueError("Evaluation unexpectedly trained or selected a model")
    if digest(resolve(protocol["checkpoint"])) != expected["checkpoint_sha256"]:
        raise ValueError("Original evaluated checkpoint changed")
    rows = summary.get("results", [])
    if len(rows) != 2 or {row["color"] for row in rows} != {1, -1}:
        raise ValueError("Missing development color")
    with (directory / "games.csv").open(encoding="utf-8", newline="") as handle:
        games = list(csv.DictReader(handle))
    if len(games) != 80:
        raise ValueError("Game CSV does not contain 80 games")
    from rl.evaluation.eval_a2c_tactical import wilson
    for row in rows:
        color = row["color"]
        group = [game for game in games if int(game["color"]) == color]
        seed = expected["seed_black" if color == 1 else "seed_white"]
        if (len(group) != 40 or {int(g["game"]) for g in group} != set(range(40)) or
                any(int(g["seed"]) != seed + int(g["game"]) for g in group) or
                any(int(g["outcome"]) not in (-1, 0, 1) for g in group) or
                any(g[field] != expected[field] for g in group
                    for field in ("split", "mode", "opponent", "rules"))):
            raise ValueError("Game CSV seeds/counts mismatch")
        counts = {outcome: sum(int(g["outcome"]) == outcome for g in group) for outcome in (1, 0, -1)}
        for name, outcome in (("wins", 1), ("draws", 0), ("losses", -1)):
            if row[name] != counts[outcome]:
                raise ValueError("CSV and WDL disagree")
        if (row["games"] != 40 or sum(row[k] for k in ("wins", "draws", "losses")) != 40 or
                abs(row["score"] - (row["wins"] + .5*row["draws"])/40) > 1e-12):
            raise ValueError("Development score/count mismatch")
        if abs(row["win_rate"] - row["wins"]/40) > 1e-12:
            raise ValueError("Development win rate mismatch")
        if len(row["wilson_95"]) != 2 or any(abs(a-b) > 1e-12 for a, b in zip(row["wilson_95"], wilson(row["wins"], 40))):
            raise ValueError("Development Wilson interval mismatch")
    if abs(summary["mean_score"] - statistics.mean(row["score"] for row in rows)) > 1e-12:
        raise ValueError("Mean score mismatch")
    if abs(summary["mean_win_rate"] - statistics.mean(row["win_rate"] for row in rows)) > 1e-12:
        raise ValueError("Mean win rate mismatch")
    return summary


def select_best(cases):
    return max(cases, key=lambda item: (item["mean_score"], item["mean_win_rate"], -item["episode"]))


def selection_report(completed):
    groups = {}
    for recipe in RECIPES:
        groups[recipe] = {}
        for mode in MODES:
            rows = [row for row in completed.values() if row["recipe"] == recipe and row["mode"] == mode]
            groups[recipe][mode] = {"complete": len(rows) == 7, "evaluated_candidates": len(rows),
                "best": select_best(rows) if rows else None}
    common = {}
    main_recipes = RECIPES[:3]
    for mode in MODES:
        if all(groups[recipe][mode]["complete"] for recipe in main_recipes):
            chosen = [groups[recipe][mode]["best"] for recipe in main_recipes]
            scores, wins = ([row[field] for row in chosen] for field in ("mean_score", "mean_win_rate"))
            common[mode] = {"mean_score": statistics.mean(scores), "sample_sd_score": statistics.stdev(scores),
                "mean_win_rate": statistics.mean(wins), "sample_sd_win_rate": statistics.stdev(wins),
                "training_seeds": [11, 22, 33], "per_seed": chosen,
                "shared_expert_data": True, "independent_training_seeds": True}
    chosen_wrapper = (max(MODES, key=lambda mode: (common[mode]["mean_score"],
        common[mode]["mean_win_rate"], -MODES.index(mode))) if len(common) == 2 else None)
    return {"format": FORMAT, "split": "validation", "groups": groups,
            "common_wrapper_comparison": common, "selected_common_wrapper": chosen_wrapper,
            "wrapper_selection_complete": chosen_wrapper is not None,
            "selection": "Per seed/mode mean Greedy score; ties win rate then earlier episode. Common wrapper by three-seed mean score, ties mean win rate then policy.",
            "initialization": "Fresh random initialization per seed + CE20; shared formal AZ400 expert data. Seed22 extends complete copied 10k run.",
            "flat22_role": "secondary representation reference, not another residual training seed",
            "final_testing_started": False}


class Selector:
    def __init__(self, args):
        self.args, self.output = args, resolve(args.output)
        self.cases = case_plan(args.confirmation)
        self.processes, self.completed = {}, {}
        self.ready_cache = {}
        self.reuse_directories = []

    def publish(self, state, **details):
        atomic_json(self.output / "selector_status.json", {"format": FORMAT, "state": state,
            "process": process_identity(os.getpid()), "updated_at": now(), "completed_cases": len(self.completed),
            "expected_cases": len(self.cases), "final_testing_started": False, **details})

    def base(self, case):
        return self.output / "cases" / case["key"]

    def assert_sources(self):
        protocol = read_json(self.output / "selector_protocol.json")
        if protocol and protocol["sources"] != sources():
            raise RuntimeError("Frozen selector/evaluator source changed; preserve this queue")

    def ready(self, case):
        import torch
        path = Path(case["checkpoint"])
        if not path.is_file() or not path.with_suffix(".checkpoint").is_file():
            return False
        if case["key"] in self.ready_cache:
            if digest(path) != self.ready_cache[case["key"]]:
                raise RuntimeError("Immutable model candidate changed")
            return True
        model = torch.load(path, map_location="cpu", weights_only=True)
        complete = torch.load(path.with_suffix(".checkpoint"), map_location="cpu", weights_only=False)
        if model.get("episode") != case["episode"] or complete.get("episode") != case["episode"]:
            raise RuntimeError("Candidate episode mismatch")
        protocol = complete["protocol"]
        if protocol["config"]["seed"] != case["training_seed"]:
            raise RuntimeError("Candidate training seed mismatch")
        expected_architecture = case["recipe"].split("_", 1)[0]
        spec = model.get("spec", {})
        if (spec.get("architecture") != expected_architecture or spec.get("features") != "tactical10" or
                json.loads(json.dumps(spec)) != json.loads(json.dumps(protocol.get("spec"))) or
                model.get("rules") != "freestyle" or protocol["config"].get("rules") != "freestyle"):
            raise RuntimeError("Candidate architecture/features/rules provenance mismatch")
        from rl.training.a2c_research import source_hashes as training_sources
        if (protocol.get("dataset_sha256") != "ca566f90aaf7695ad8fd4b82e997ff221bb483859825ab6e5e589942266307e3" or
                protocol.get("sources") != training_sources()):
            raise RuntimeError("Candidate shared expert/training source provenance mismatch")
        if set(model["state_dict"]) != set(complete["net"]) or any(
                not torch.equal(value, complete["net"][key]) for key, value in model["state_dict"].items()):
            raise RuntimeError("Candidate weights mismatch complete checkpoint")
        self.ready_cache[case["key"]] = digest(path)
        return True

    def record_result(self, case, directory, expected, reuse=False):
        summary = validate_summary(directory, expected)
        result = {**case, "checkpoint_sha256": expected["checkpoint_sha256"], "identity": expected,
            "evaluation_output": str(directory), "reused_exact_development_case": reuse,
            "mean_score": summary["mean_score"], "mean_win_rate": summary["mean_win_rate"],
            "results": summary["results"], "summary_sha256": digest(Path(directory) / "summary.json"),
            "games_sha256": digest(Path(directory) / "games.csv"),
            "moves_sha256": digest(Path(directory) / "moves.csv")}
        self.completed[case["key"]] = result
        atomic_json(self.base(case) / "result.json", result)
        return result

    def existing(self, case, expected):
        base = self.base(case)
        saved = read_json(base / "result.json")
        if saved:
            return "complete", self.record_result(case, saved["evaluation_output"], expected,
                saved.get("reused_exact_development_case", False))
        for attempt in sorted(base.glob("attempt_*"), reverse=True):
            if not attempt.is_dir():
                continue
            launcher = read_json(attempt / "launcher.json")
            process = self.processes.get(str(attempt))
            actual = process_identity(launcher.get("pid"))
            alive = process.poll() is None if process is not None else same_process(launcher, actual)
            unknown = bool(launcher.get("birth") and actual.get("alive") is None)
            if lock_held(attempt / "eval.lock") or alive or unknown:
                return "active", {"output": str(attempt), "launcher_process": actual,
                    "runtime_process": process_identity(read_json(attempt / "status.json").get("pid")),
                    "writer_lock_held": lock_held(attempt / "eval.lock")}
            if read_json(attempt / "status.json").get("state") == "complete":
                return "complete", self.record_result(case, attempt, expected)
            if launcher and launcher.get("exit_code") is None:
                launcher.update(exit_code=process.returncode if process is not None else "unknown stopped",
                                finished_observed_at=now(), process=actual)
                atomic_json(attempt / "launcher.json", launcher)
        if len(list(base.glob("attempt_*"))) >= 3:
            raise RuntimeError(f"Three evaluation attempts stopped for {case['key']}; preserve evidence for review")
        for directory in self.reuse_directories:
            protocol = read_json(directory / "protocol.json")
            if {key: protocol.get(key) for key in IDENTITY_KEYS} == expected:
                return "complete", self.record_result(case, directory, expected, reuse=True)
        return "pending", None

    def launch(self, case):
        self.assert_sources()
        base = self.base(case)
        attempt = base / f"attempt_{len(list(base.glob('attempt_*')))+1:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        log_path = attempt / "console.log"
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                 subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
        cmd = command(case, attempt)
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        self.processes[str(attempt)] = process
        actual = process_identity(process.pid)
        atomic_json(attempt / "launcher.json", {"pid": process.pid, "birth": actual.get("birth"),
            "process": actual, "command": cmd, "cwd": str(ROOT), "started_at": now(),
            "log": str(log_path), "lifecycle": "hidden process; Codex OS job membership may apply"})
        return {"output": str(attempt), "launcher_process": actual, "writer_lock_held": False}

    def run(self):
        configuration = {"format": FORMAT, "sources": sources(), "cases": self.cases,
            "max_workers": self.args.workers, "games_per_color": 40, "device": "cpu",
            "split": "validation", "seeds": [3100000, 3200000],
            "modes": list(MODES), "shared_expert_sha256": "ca566f90aaf7695ad8fd4b82e997ff221bb483859825ab6e5e589942266307e3",
            "reuse_directory": str(resolve(self.args.reuse_directory)), "final_tests_allowed": False}
        previous = read_json(self.output / "selector_protocol.json")
        if previous and previous != configuration:
            raise RuntimeError("Selector source/plan/input changed; preserve queue for review")
        atomic_json(self.output / "selector_protocol.json", configuration)
        self.reuse_directories = [path.parent for path in resolve(self.args.reuse_directory).glob("*/summary.json")]
        while True:
            self.assert_sources()
            active, pending = [], []
            for case in self.cases:
                if case["key"] in self.completed:
                    continue
                if not self.ready(case):
                    continue
                state, observed = self.existing(case, identity(case))
                if state == "active":
                    active.append({"key": case["key"], **observed})
                elif state == "pending":
                    pending.append(case)
            for case in pending[:max(0, self.args.workers-len(active))]:
                active.append({"key": case["key"], **self.launch(case)})
            report = selection_report(self.completed)
            report.update(updated_at=now(), completed_cases=len(self.completed), expected_cases=len(self.cases))
            atomic_json(self.output / "selection.json", report)
            atomic_json(self.output / "completed_cases.json", self.completed)
            if len(self.completed) == len(self.cases):
                for case in self.cases:
                    result = self.completed[case["key"]]
                    if digest(case["checkpoint"]) != result["checkpoint_sha256"]:
                        raise RuntimeError("Completed candidate weights changed")
                    for name, field in (("summary.json", "summary_sha256"),
                                        ("games.csv", "games_sha256"), ("moves.csv", "moves_sha256")):
                        if digest(Path(result["evaluation_output"]) / name) != result[field]:
                            raise RuntimeError("Completed development evidence changed")
                self.publish("development_selection_complete", common_wrapper=report["selected_common_wrapper"])
                return
            training = read_json(resolve(self.args.confirmation) / "suite_status.json")
            if training.get("state") == "confirmation_training_complete" and not active and not pending:
                raise RuntimeError("Completed training lacks expected immutable candidates")
            self.publish("running" if active else "waiting_training_candidates", active=active,
                ready_pending_cases=len(pending), training_state=training.get("state"))
            time.sleep(self.args.poll_seconds)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirmation", default="runs/a2c_research_20261020/confirmation")
    p.add_argument("--output", default="runs/a2c_research_20261020/confirmation/selection")
    p.add_argument("--reuse-directory", default="runs/a2c_research_20261020/stage5/guarded_policy_study")
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--plan-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if not 0 < args.poll_seconds <= 60:
        raise ValueError("poll-seconds must be in (0,60]")
    if args.plan_only:
        print(json.dumps({"format": FORMAT, "cases": case_plan(args.confirmation),
                          "final_tests_allowed": False}, indent=2))
        return
    selector = Selector(args)
    with exclusive_run(selector.output / "selector.lock"):
        try:
            selector.run()
        except BaseException as error:
            selector.publish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=repr(error), traceback=traceback.format_exc(),
                child_policy="Preserve children and evidence; rerun to reattach by OS locks and birth")
            raise


if __name__ == "__main__":
    main()
