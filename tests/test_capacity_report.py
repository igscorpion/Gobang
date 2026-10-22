"""Keep warm-start validation separate from final capacity comparison results."""

import csv
import json

from rl.evaluation.report_capacity import publish


def fixture_run(out, width, complete=True):
    directory = out / f"az_width{width}_fixed"
    directory.mkdir(parents=True)
    config = {
        "seed": 11,
        "episodes": 1000,
        "sims": 400,
        "fixed_ratio": 0.6,
        "rules": "freestyle",
        "batch_size": 512,
        "eval_every": 250,
        "validation_games": 40,
        "final_games": 100,
    }
    data = {
        "status.json": {
            "state": "complete" if complete else "training",
            "episode": 1000 if complete else 150,
        },
        "config.json": config,
        "final_evaluation_model.json": {"checkpoint": "best.pt", "test_seed": 800000},
        # Realistic initial validation is deliberately higher than the test score.
        "initial_validation.json": [{"opponent": "greedy", "win_rate": 0.99}],
        "final_evaluation.json": [
            {
                "opponent": "greedy",
                "color": color,
                "games": 100,
                "wins": wins,
                "draws": 5,
                "losses": 95 - wins,
                "win_rate": wins / 100,
                "score": (wins + 2.5) / 100,
                "wilson_low": 0.5,
                "wilson_high": 0.95,
            }
            for color, wins in ((1, 80 + width // 64), (-1, 60 + width // 64))
        ],
    }
    for name, value in data.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    (directory / "best.pt").write_bytes(b"fixture checkpoint")
    with (directory / "validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "opponent", "score"])
        writer.writeheader()
        for episode in (0, 250):
            for _ in range(2):
                writer.writerow(
                    {"episode": episode, "opponent": "greedy", "score": 0.8}
                )


def test_incomplete_training_never_publishes_validation_as_final_test(tmp_path):
    out = tmp_path / "runs/production"
    fixture_run(out, 64)
    fixture_run(out, 128, complete=False)
    result = publish(out, root=tmp_path)
    assert not result["complete"] and "greedy" not in result["experiments"][1]
    text = (out / "alphazero_capacity_comparison.md").read_text(encoding="utf-8")
    assert "99.0%" not in text and "当前不作扩容效果结论" in text


def test_completed_comparison_uses_test_rates_and_keeps_first_validation_tie(tmp_path):
    out = tmp_path / "runs/production"
    fixture_run(out, 64)
    fixture_run(out, 128)
    (tmp_path / "README.md").write_text("# Original user content\n", encoding="utf-8")
    result = publish(out, root=tmp_path)
    assert result["complete"]
    assert all(
        abs(v - 1.0) < 1e-9 for v in result["win_rate_differences_percentage_points"]
    )
    assert all(item["best_episode"] == 0 for item in result["experiments"])
    publish(out, root=tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert text.startswith("# Original user content")
    assert text.count("<!-- alphazero-capacity-results:start -->") == 1
