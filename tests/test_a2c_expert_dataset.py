"""Dataset identity, trajectory validation, split isolation and durable resume."""
import copy
import hashlib
import json

import pytest
import torch

from rl.agents.random_agent import RandomAgent
from rl.training.a2c_expert_dataset import (complete_game, generate, make_protocol,
    merge_games, parser, schedule, verify_game, worker)
from rl.training.a2c_research import load_demonstrations
from rl.training.common import exclusive_run


META = {"kind": "seeded_random_test_teacher", "test_only": True,
        "purpose": "pipeline invariants; never production teacher-quality evidence"}


def args_for(path):
    return parser().parse_args(["--output", str(path), "--workers", "1",
        "--train-games", "2", "--validation-games", "2"])


def factory(seed, rules):
    return RandomAgent(seed=seed + 999)


def test_defaults_preserve_requested_teacher_and_budget(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path)])
    assert (args.train_games, args.validation_games, args.workers) == (512, 64, 4)
    assert (args.train_seed, args.validation_seed, args.simulations) == (5_000_000, 5_100_000, 400)
    assert (args.teacher_probability, args.greedy_probability, args.random_probability) == (.8, .1, .1)
    assert args.random_opening == 6 and args.rules == "freestyle"


def test_committed_game_rejects_bad_identity_labels_and_behavior(tmp_path):
    args = args_for(tmp_path)
    protocol = make_protocol(args, META)
    entry = schedule(args)[0]
    game = complete_game(args, entry, protocol, factory)
    verify_game(game, entry, protocol)
    for mutation in ("seed", "label", "behavior"):
        bad = copy.deepcopy(game)
        if mutation == "seed":
            bad["seed"] += 1
        elif mutation == "label":
            row = list(bad["rows"][0])
            row[2] = 99
            bad["rows"][0] = tuple(row)
        else:
            bad["trace"][0]["action"] = 99
        with pytest.raises(ValueError):
            verify_game(bad, entry, protocol)


def test_four_fresh_games_merge_without_board_d4_leakage(tmp_path):
    args = args_for(tmp_path / "data")
    manifest = generate(args, factory, META)
    data = load_demonstrations(tmp_path / "data/demonstrations.pt", "freestyle")
    assert manifest["committed_games"] == 4
    assert len({g["game_id"] for g in manifest["games"]}) == 4
    assert manifest["protocol"]["external_trajectories_allowed"] is False
    assert not set(data["train"]["split_ids"]) & set(data["validation"]["split_ids"])
    assert len(set(data["validation"]["split_ids"])) == len(data["validation"]["split_ids"])
    assert set(data["train"]["game_seeds"].tolist()) <= {5_000_000, 5_000_001}
    assert set(data["validation"]["game_seeds"].tolist()) <= {5_100_000, 5_100_001}
    assert json.loads((tmp_path / "data/status.json").read_text())["state"] == "complete"
    with pytest.raises(ValueError, match="resume"):
        generate(args, factory, META)
    next((tmp_path / "data/shards").glob("worker_*/games/*.pt")).unlink()
    with pytest.raises(ValueError, match="missing=1"):
        merge_games(args, tmp_path / "data", manifest["protocol"])


def test_interrupted_game_is_regenerated_while_commits_remain_unchanged(tmp_path):
    args = args_for(tmp_path / "resumed")
    calls = []
    def interrupted(seed, rules):
        calls.append(seed)
        if len(calls) == 3:
            raise RuntimeError("simulated interruption before game commit")
        return factory(seed, rules)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate(args, interrupted, META)
    files = list((tmp_path / "resumed/shards").glob("worker_*/games/*.pt"))
    assert len(files) == 2
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.resume = True
    generate(args, factory, META)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    uninterrupted = args_for(tmp_path / "fresh")
    generate(uninterrupted, factory, META)
    resumed = load_demonstrations(tmp_path / "resumed/demonstrations.pt", "freestyle")
    fresh = load_demonstrations(tmp_path / "fresh/demonstrations.pt", "freestyle")
    for split in ("train", "validation"):
        for key in ("base", "masks", "actions", "game_seeds"):
            assert torch.equal(resumed[split][key], fresh[split][key])
        for key in ("position_ids", "split_ids"):
            assert resumed[split][key] == fresh[split][key]
    args.simulations += 1
    with pytest.raises(ValueError, match="changing"):
        generate(args, factory, META)


def test_reserved_or_overlapping_seeds_are_rejected(tmp_path):
    args = args_for(tmp_path)
    args.validation_seed = args.train_seed
    with pytest.raises(ValueError, match="overlap"):
        generate(args, factory, META)
    args.validation_seed = 5_100_000
    args.train_seed = 8_100_000
    with pytest.raises(ValueError, match="reserved"):
        generate(args, factory, META)


def test_second_worker_cannot_write_to_a_locked_shard(tmp_path):
    args = args_for(tmp_path)
    protocol = make_protocol(args, META)
    with exclusive_run(tmp_path / "shards/worker_00/run.lock"):
        with pytest.raises(RuntimeError, match="second writer"):
            worker(args, protocol, 0, factory)
