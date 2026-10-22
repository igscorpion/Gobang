"""A drained A2C barrier must resume the exact next optimizer trajectory."""

import csv

import numpy as np
import torch

from rl.training.local_a2c import parser, run


def same_nested(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_nested(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for item, other in zip(left, right):
            same_nested(item, other)
    else:
        assert left == right


def arguments(directory, episodes, resume=False):
    flags = [
        "--output", str(directory), "--episodes", str(episodes),
        "--device", "cpu", "--seed", "17", "--num-envs", "2",
        "--rollout-steps", "3", "--save-every", "4", "--eval-every", "4",
        "--history-every", "4", "--validation-games", "1", "--final-games", "1",
    ]
    if resume:
        flags.append("--resume")
    return parser().parse_args(flags)


def read_rows(path, omit=()):
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: value for key, value in row.items() if key not in omit}
            for row in csv.DictReader(handle)
        ]


def test_cpu_barrier_resume_matches_continuous_training_exactly(tmp_path):
    full, split = tmp_path / "full", tmp_path / "split"
    run(arguments(full, 8))
    run(arguments(split, 4))
    run(arguments(split, 8, resume=True))
    complete = torch.load(full / "latest.checkpoint", weights_only=False)
    resumed = torch.load(split / "latest.checkpoint", weights_only=False)
    for key in (
        "net", "optimizer", "episode", "completed_episode_ids", "next_episode_id",
        "learner_transitions", "environment_plies", "updates", "ratio", "best",
        "best_episode", "history_pool", "worker_rngs", "agent_rng", "rng",
        "opponent_counts",
    ):
        same_nested(complete[key], resumed[key])
    assert resumed["episode"] == 8
    assert resumed["completed_episode_ids"] == list(range(1, 9))
    assert resumed["next_episode_id"] == 9
    assert len(resumed["history_pool"]) == 2
    assert len(resumed["worker_rngs"]) == 2
    assert sum(resumed["opponent_counts"].values()) == 8

    training = read_rows(split / "training.csv", {"seconds", "training_seconds"})
    assert len(training) == 8
    assert sorted(int(row["episode"]) for row in training) == list(range(1, 9))
    assert training == read_rows(full / "training.csv", {"seconds", "training_seconds"})
    updates = read_rows(split / "updates.csv", {"training_seconds"})
    assert [int(row["update"]) for row in updates] == list(range(1, resumed["updates"] + 1))
    assert updates == read_rows(full / "updates.csv", {"training_seconds"})
    assert read_rows(full / "validation.csv") == read_rows(split / "validation.csv")
    assert read_rows(full / "final_games.csv") == read_rows(split / "final_games.csv")
