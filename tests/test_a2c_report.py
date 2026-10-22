"""Independent tests cannot select a model or become results before completion."""

import csv
import json

import pytest

from rl.evaluation.report_a2c import publish


def fixture_run(out, family, seed, validation=0.5, wins=40, state="complete"):
    directory = out / f"{family}_seed{seed}"
    directory.mkdir(parents=True)
    records = {
        "status.json": {"state": state, "episode": 30000, "target_episodes": 30000},
        "config.json": {"seed": seed, "episodes": 30000, "parameters": 1405042},
        "final_evaluation_model.json": {"checkpoint": "best.pt", "test_seed": 800000, "selection": "validation greedy score only"},
        "final_evaluation.json": [
            {"opponent": opponent, "color": color, "games": 100, "wins": wins, "draws": 10, "losses": 90 - wins,
             # Deliberately stale derived fields: report must recompute from counts.
             "win_rate": 0.999, "score": 0.999, "wilson_low": 0.9, "wilson_high": 1}
            for opponent in ("random", "greedy") for color in (1, -1)
        ],
    }
    for name, value in records.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    (directory / "best.pt").write_bytes(b"checkpoint")
    with (directory / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode", "opponent", "color", "score"))
        writer.writeheader()
        for episode, score in ((0, 1.0), (1000, validation), (2000, validation)):
            for color in (1, -1):
                writer.writerow({"episode": episode, "opponent": "greedy", "color": color, "score": score})
    return directory


def test_pending_result_never_exposes_initial_or_partial_test(tmp_path):
    out = tmp_path / "runs/production"
    fixture_run(out, "a2c4", 11, state="training")
    result = publish(out, root=tmp_path)
    item = result["experiments"][0]
    assert "test" not in item
    assert result["a2c_complete"] is False
    assert json.loads((out / "a2c_models.json").read_text()) == {}
    assert "99.9%" not in (out / "a2c_dqn_comparison.md").read_text(encoding="utf-8")


def test_selection_uses_validation_and_earliest_tie_not_independent_test(tmp_path):
    out = tmp_path / "runs/production"
    fixture_run(out, "a2c4", 11, validation=0.8, wins=20)
    fixture_run(out, "a2c4", 22, validation=0.7, wins=85)
    fixture_run(out, "a2c4", 33, validation=0.8, wins=70)
    result = publish(out, root=tmp_path)
    model = json.loads((out / "a2c_models.json").read_text())["a2c4"]
    assert result["a2c_complete"] is True
    assert model["seed"] == 11 and model["selected_episode"] == 1000
    assert model["checkpoint"] == "runs/production/a2c4_seed11/best.pt"
    test = result["experiments"][0]["test"][0]
    assert test["win_rate"] == 0.2 and test["score"] == 0.25
    assert 0 < test["wilson_low"] < 0.2 < test["wilson_high"] < 1
    assert not (out / "models.json").exists()


def test_across_seeds_and_paired_differences_do_not_treat_absent_seed_as_zero(tmp_path):
    out = tmp_path / "runs/production"
    for seed, a2c, dqn in ((11, 30, 20), (22, 50, 30)):
        fixture_run(out, "a2c4", seed, wins=a2c)
        fixture_run(out, "dqn4", seed, wins=dqn)
    result = publish(out, root=tmp_path)
    summary = result["across_seeds"][0]["win_rate"]
    assert summary["n"] == 2 and summary["mean"] == pytest.approx(0.4)
    assert summary["sample_sd"] == pytest.approx(0.2 / 2**0.5)
    pair = result["paired_a2c_minus_dqn"][0]
    assert [row["seed"] for row in pair["pairs"]] == [11, 22]
    assert pair["win_rate_difference_percentage_points"]["mean"] == pytest.approx(15)
    assert pair["win_rate_difference_percentage_points"]["sample_sd"] == pytest.approx(10 / 2**0.5)


def test_incomplete_final_evidence_blocks_publication_and_registration(tmp_path):
    out = tmp_path / "runs/production"
    directory = fixture_run(out, "a2c4", 11)
    records = json.loads((directory / "final_evaluation.json").read_text())
    (directory / "final_evaluation.json").write_text(json.dumps(records[:3]))
    result = publish(out, root=tmp_path)
    item = result["experiments"][0]
    assert item["issues"] and "test" not in item
    assert json.loads((out / "a2c_models.json").read_text()) == {}
