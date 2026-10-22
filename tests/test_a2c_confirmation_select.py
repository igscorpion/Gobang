"""Selection protocol/reuse/scheduling integrity; no real evaluation games."""
import csv
from pathlib import Path
import sys

import pytest
import torch

from rl.evaluation import a2c_confirmation_select as module
from rl.evaluation.eval_a2c_tactical import wilson
from rl.training.common import atomic_json, exclusive_run


def arguments(tmp_path):
    return module.parser().parse_args(["--confirmation", str(tmp_path / "confirmation"),
        "--output", str(tmp_path / "selection"), "--reuse-directory", str(tmp_path / "prior"),
        "--poll-seconds", ".01"])


def test_all_56_candidates_have_fixed_cpu_development_budget(tmp_path):
    cases = module.case_plan(tmp_path / "confirmation")
    assert len(cases) == len({case["key"] for case in cases}) == 56
    for recipe in module.RECIPES:
        rows = [case for case in cases if case["recipe"] == recipe]
        assert len(rows) == 14
        assert {row["episode"] for row in rows} == set(range(0, 30001, 5000))
        assert {row["mode"] for row in rows} == {"policy", "guarded"}
        for row in rows:
            args = module.evaluator.parser().parse_args(module.command(row, tmp_path / "evaluation")[4:])
            assert args.split == "validation" and args.games == 40
            assert args.device == "cpu" and args.opponent == "greedy" and args.rules == "freestyle"
            assert (args.seed_black, args.seed_white) == (3100000, 3200000)


def example_evaluation(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    case = {"checkpoint": str(checkpoint), "mode": "guarded"}
    expected = module.identity(case)
    directory = tmp_path / "completed_eval"
    directory.mkdir()
    (directory / "eval.lock").write_bytes(b"0")
    (directory / "moves.csv").write_text("preserved replay evidence")
    games = [{"color": color, "game": game, "split": "validation", "mode": "guarded",
        "opponent": "greedy", "rules": "freestyle",
        "seed": (3100000 if color == 1 else 3200000)+game,
        "outcome": 1 if game < 20 else 0 if game < 30 else -1}
        for color in (1, -1) for game in range(40)]
    with (directory / "games.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=games[0].keys())
        writer.writeheader()
        writer.writerows(games)
    protocol = {**expected, "checkpoint": str(checkpoint),
        "training_performed": False, "selection_performed": False}
    atomic_json(directory / "protocol.json", protocol)
    summary = {"protocol": protocol, "total_games": 80, "sources_and_weights_unchanged": True,
        "mean_score": .625, "mean_win_rate": .5,
        "results": [{"color": color, "games": 40, "wins": 20, "draws": 10,
                     "losses": 10, "score": .625, "win_rate": .5,
                     "wilson_95": wilson(20, 40)} for color in (1, -1)]}
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "status.json", {"state": "complete"})
    return directory, expected


def test_reuse_requires_exact_hash_mode_device_rules_seeds_and_source(tmp_path):
    directory, expected = example_evaluation(tmp_path)
    assert module.validate_summary(directory, expected)["total_games"] == 80
    for key, value in (("mode", "policy"), ("device", "cuda"), ("rules", "renju"),
                       ("games", 20), ("seed_black", 3100001),
                       ("checkpoint_sha256", "other"), ("source_sha256", {"changed": "source"})):
        with pytest.raises(ValueError, match="identity"):
            module.validate_summary(directory, {**expected, key: value})
    with exclusive_run(directory / "eval.lock"):
        with pytest.raises(ValueError, match="unlocked"):
            module.validate_summary(directory, expected)


def test_wdl_csv_corruption_prevents_summary_reuse(tmp_path):
    directory, expected = example_evaluation(tmp_path)
    text = (directory / "games.csv").read_text()
    (directory / "games.csv").write_text(text.replace("1,0,validation,guarded,greedy,freestyle,3100000,1",
        "1,0,validation,guarded,greedy,freestyle,3100000,-1"))
    with pytest.raises(ValueError, match="WDL"):
        module.validate_summary(directory, expected)


def test_same_wrapper_is_selected_across_all_training_seeds(tmp_path):
    cases = module.case_plan(tmp_path / "confirmation")
    scores = {11: {"policy": .90, "guarded": .70},
              22: {"policy": .60, "guarded": .95},
              33: {"policy": .50, "guarded": .80}}
    completed = {case["key"]: {**case,
        "mean_score": scores.get(case["training_seed"], scores[22])[case["mode"]],
        "mean_win_rate": .5} for case in cases}
    report = module.selection_report(completed)
    assert report["selected_common_wrapper"] == "guarded"
    selected = report["common_wrapper_comparison"]["guarded"]["per_seed"]
    assert {row["mode"] for row in selected} == {"guarded"}
    assert {row["training_seed"] for row in selected} == {11, 22, 33}
    assert all(row["episode"] == 0 for row in selected)  # ties choose earlier.
    assert report["groups"]["residual_tactical10_seed11"]["policy"]["best"]["mean_score"] == .9
    del completed["residual_tactical10_seed33/episode_030000/guarded"]
    assert module.selection_report(completed)["selected_common_wrapper"] is None


def test_candidate_requires_matching_full_checkpoint_and_training_seed(tmp_path):
    selector = module.Selector(arguments(tmp_path))
    case = selector.cases[0]
    path = Path(case["checkpoint"])
    path.parent.mkdir(parents=True)
    weight = {"layer": torch.tensor([1.])}
    spec = {"architecture": "residual", "features": "tactical10", "channels": (32, 64, 64)}
    torch.save({"episode": 0, "state_dict": weight, "spec": spec, "rules": "freestyle"}, path)
    assert not selector.ready(case)
    from rl.training.a2c_research import source_hashes
    protocol = {"config": {"seed": 22, "rules": "freestyle"}, "spec": spec,
        "dataset_sha256": "ca566f90aaf7695ad8fd4b82e997ff221bb483859825ab6e5e589942266307e3", "sources": source_hashes()}
    complete = {"episode": 0, "net": weight, "protocol": protocol}
    torch.save(complete, path.with_suffix(".checkpoint"))
    with pytest.raises(RuntimeError, match="seed"):
        selector.ready(case)
    protocol["config"]["seed"] = 11
    protocol["dataset_sha256"] = "wrong dataset"
    torch.save(complete, path.with_suffix(".checkpoint"))
    with pytest.raises(RuntimeError, match="shared expert"):
        selector.ready(case)
    protocol["dataset_sha256"] = "ca566f90aaf7695ad8fd4b82e997ff221bb483859825ab6e5e589942266307e3"
    torch.save(complete, path.with_suffix(".checkpoint"))
    assert selector.ready(case)
    torch.save({"episode": 0, "state_dict": {"layer": torch.tensor([2.])}}, path)
    with pytest.raises(RuntimeError, match="changed"):
        selector.ready(case)


def test_existing_held_case_is_observed_before_any_reuse(tmp_path):
    selector = module.Selector(arguments(tmp_path))
    case = selector.cases[0]
    attempt = selector.base(case) / "attempt_001"
    attempt.mkdir(parents=True)
    with exclusive_run(attempt / "eval.lock"):
        state, observation = selector.existing(case, {})
    assert state == "active" and observation["writer_lock_held"]
    assert not selector.completed


def test_source_change_stops_before_spawn(tmp_path, monkeypatch):
    selector = module.Selector(arguments(tmp_path))
    atomic_json(selector.output / "selector_protocol.json", {"sources": {"old": "sha"}})
    monkeypatch.setattr(module, "sources", lambda: {"changed": "sha"})
    with pytest.raises(RuntimeError, match="source changed"):
        selector.launch(selector.cases[0])
    assert not selector.base(selector.cases[0]).exists()


def test_partial_attempt_is_preserved_and_new_hidden_child_has_birth(tmp_path, monkeypatch):
    selector = module.Selector(arguments(tmp_path))
    case = selector.cases[0]
    failed = selector.base(case) / "attempt_001"
    failed.mkdir(parents=True)
    (failed / "moves.csv").write_text("preserve partial games")
    atomic_json(failed / "status.json", {"state": "failed"})
    monkeypatch.setattr(selector, "assert_sources", lambda: None)
    monkeypatch.setattr(module, "command", lambda *_: [sys.executable, "-u", "-c", "print('eval scheduling probe')"])
    result = selector.launch(case)
    new = Path(result["output"])
    process = selector.processes[str(new)]
    assert process.wait(timeout=10) == 0
    launcher = module.read_json(new / "launcher.json")
    assert new.name == "attempt_002" and launcher["birth"] and launcher["pid"] == process.pid
    assert (failed / "moves.csv").read_text() == "preserve partial games"
    assert "eval scheduling probe" in (new / "console.log").read_text()
