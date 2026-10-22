"""Check the v2 extension preserves source/frozen-test separation."""

import json

import pytest
import torch

from rl.agents.a2c_guarded import GuardedA2CAgent
from rl.agents.a2c_research import ModelSpec, ResearchA2CAgent
from rl.evaluation import a2c_research_eval as original
from rl.evaluation import a2c_research_eval_v2 as v2


def settings(tmp_path, **changes):
    args = v2.parser().parse_args(["--output", str(tmp_path / "output")])
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def test_v2_factory_loads_a_research_checkpoint_and_hashes_both_extensions(tmp_path):
    policy = ResearchA2CAgent(ModelSpec(architecture="residual"))
    path = tmp_path / "policy.pt"
    policy.save(path)
    args = settings(tmp_path, checkpoint=str(path))
    guarded = v2.make_agent(args)
    assert isinstance(guarded, GuardedA2CAgent)
    assert guarded.policy.spec == policy.spec
    for key, tensor in policy.net.state_dict().items():
        assert torch.equal(tensor, guarded.policy.net.state_dict()[key])
    hashes = v2.source_hashes(args)
    for name in ("rl/evaluation/a2c_research_eval.py", "rl/evaluation/a2c_research_eval_v2.py", "rl/agents/a2c_guarded.py"):
        assert hashes[name] == original.digest(original.ROOT / name)


def test_scoped_factory_substitution_is_restored_after_failure(tmp_path, monkeypatch):
    factory, source_function = original.make_agent, original.source_hashes
    def failing_run(args):
        assert original.make_agent is v2.make_agent
        assert original.source_hashes is v2.source_hashes
        raise RuntimeError("interrupted evaluation")
    monkeypatch.setattr(original, "run", failing_run)
    with pytest.raises(RuntimeError, match="interrupted"):
        v2.run(settings(tmp_path))
    assert original.make_agent is factory
    assert original.source_hashes is source_function


def test_development_cannot_use_other_or_reserved_seed_ranges(tmp_path):
    for seeds in ((8_100_000, 8_200_000), (7_100_000, 7_200_000)):
        with pytest.raises(ValueError, match="3100000/3200000"):
            v2.run(settings(tmp_path, seed_black=seeds[0], seed_white=seeds[1]))


def test_guarded_test_identity_includes_extra_sources_and_wrapper(tmp_path):
    manifest = tmp_path / "selection.json"
    args = settings(tmp_path, split="test", games=100, seed_black=8_100_000,
                    seed_white=8_200_000, frozen_selection=str(manifest))
    hashes = v2.source_hashes(args)
    identity = {key: getattr(args, key) for key in (
        "network_kind", "mode", "device", "rules", "opponent", "games", "seed_black", "seed_white",
        "depth", "nodes", "search_candidates", "tactical_candidates", "value_weight", "sims")}
    identity.update(checkpoint_sha256="weight", source_sha256=hashes)
    manifest.write_text(json.dumps({"status": "frozen", "evaluations": [identity]}), encoding="utf-8")
    assert original.validate_split(args, "weight", hashes)["identity"] == identity
    for modified in ({**hashes, "rl/agents/a2c_guarded.py": "changed"},
                     {k: v for k, v in hashes.items() if k != "rl/agents/a2c_guarded.py"}):
        with pytest.raises(ValueError, match="Complete evaluated policy"):
            original.validate_split(args, "weight", modified)
    args.mode = "policy"
    with pytest.raises(ValueError, match="Complete evaluated policy"):
        original.validate_split(args, "weight", hashes)


def test_real_policy_probe_records_v2_sources_without_changing_original(tmp_path):
    previous_threads = torch.get_num_threads()
    source_before = original.digest(original.__file__)
    checkpoint = tmp_path / "policy.pt"
    ResearchA2CAgent(ModelSpec(architecture="residual", residual_width=4, residual_blocks=1)).save(checkpoint)
    args = settings(tmp_path, checkpoint=str(checkpoint), mode="policy", games=1, split="probe",
                    seed_black=1_000_000, seed_white=1_100_000, opponent="random")
    try:
        result = v2.run(args)
    finally:
        torch.set_num_threads(previous_threads)
    assert result["total_games"] == 2
    assert result["sources_and_weights_unchanged"]
    assert "rl/evaluation/a2c_research_eval_v2.py" in result["protocol"]["source_sha256"]
    assert original.digest(original.__file__) == source_before
    assert original.make_agent is v2._ORIGINAL_MAKE_AGENT
