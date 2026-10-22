"""Small auditor integrity checks; no final-test data or model inference."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import audit_final_a2c_research as audit

ROOT = Path("C:/Users/29024/Documents/Codex/2026-10-08/x/outputs/Gobang-local-work")
VISUAL = Path(__file__).parent
FIXTURES = VISUAL / "audit_tool_selftest_fixtures"
FIXTURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from rl.environment.gomoku import GomokuEnv
from rl.agents.greedy_agent import GreedyAgent
from rl.agents.random_agent import RandomAgent
from rl.training.common import choose


def refuses(function, contains):
    try:
        function()
    except audit.AuditError as error:
        assert contains in str(error), (contains, str(error))
    else:
        raise AssertionError("Expected refusal: " + contains)


checks = []
def passed(label):
    checks.append(label)
    print("PASS", label)


modules = GomokuEnv, GreedyAgent, RandomAgent, choose
random_result = audit.random_replay_self_test(modules)
assert random_result["passed"] and random_result["legal_action_rng_tamper_rejected"]
passed("seeded Random trajectories and legal RNG tamper rejection")

lock = FIXTURES / "readonly.lock"
lock.write_bytes(b"immutable lock bytes")
before = lock.read_bytes()
with audit.existing_unheld_lock(lock):
    refuses(lambda: audit.existing_unheld_lock(lock).__enter__(), "Active writer lock")
assert lock.read_bytes() == before
passed("read-only OS-held lock refuses duplicate and preserves bytes")

# These are synthetic protocol dictionaries, not reads of any reserved results.
dev = {key: None for key in audit.IDENTITY_FIELDS}
dev.update(network_kind="research", mode="policy", device="cpu", rules="freestyle",
           opponent="greedy", games=40, seed_black=3100000, seed_white=3200000,
           split="validation", checkpoint_sha256="synthetic", source_sha256={}, frozen_selection=None)
assert audit.frozen_identity(dev, {}, VISUAL, ROOT, True)["verified"] is False
refuses(lambda: audit.frozen_identity(dev, {}, VISUAL, ROOT, False), "Default audit accepts final")
reserved = dict(dev, seed_black=8100000, seed_white=8200000)
refuses(lambda: audit.frozen_identity(reserved, {}, VISUAL, ROOT, True), "Development overlaps reserved")
small_test = dict(reserved, split="test")
refuses(lambda: audit.frozen_identity(small_test, {}, VISUAL, ROOT, False), "at least 100")
passed("development/default/final-budget and reserved-seed refusal")

# A synthetic selection dictionary exercises matching only; it is not an
# experiment selection and contains no game results or test trajectories.
synthetic_test = dict(reserved, split="test", games=100)
identity = {key: synthetic_test[key] for key in audit.IDENTITY_FIELDS}
identity.update(checkpoint_sha256=synthetic_test["checkpoint_sha256"], source_sha256={})
selection_path = FIXTURES / "synthetic_identity_fixture_not_an_experiment.json"
selection = {"status": "frozen", "selected_using": "synthetic unit fixture only",
    "models": {"synthetic": {"role": "candidate", "recipe": "synthetic", "training_seed": 11,
        "training_run": "synthetic", "training_protocol_sha256": "synthetic"}},
    "evaluations": [identity]}
selection_path.write_text(json.dumps(selection))
synthetic_test["frozen_selection"] = {"path": str(selection_path), "sha256": audit.digest(selection_path),
    "identity": identity, "selected_using": selection["selected_using"]}
assert audit.frozen_identity(synthetic_test, {}, VISUAL, ROOT, False)["verified"]
selection["evaluations"][0] = dict(identity, training_seed=11)
selection_path.write_text(json.dumps(selection))
synthetic_test["frozen_selection"]["sha256"] = audit.digest(selection_path)
refuses(lambda: audit.frozen_identity(synthetic_test, {}, VISUAL, ROOT, False), "Complete policy identity is absent")
passed("synthetic frozen identity exact keys and SHA-indexed provenance")

unfinished = FIXTURES / "unfinished_development"
unfinished.mkdir(exist_ok=True)
(unfinished / "eval.lock").write_bytes(b"0")
(unfinished / "protocol.json").write_text(json.dumps(dev))
(unfinished / "summary.json").write_text("{}")
(unfinished / "status.json").write_text('{"state":"evaluating"}')
for filename in ("games.csv", "moves.csv"):
    (unfinished / filename).write_text("")
refuses(lambda: audit.audit_one({"directory": str(unfinished)}, VISUAL, ROOT, True, modules), "unfinished")
passed("unfinished unlocked evaluation is refused before replay")

metrics = {"win_rate": .6, "score": .7, "mean_move_ms": 10., "mean_game_plies": 20.,
           "games": 100, "wins": 60, "draws": 20, "losses": 20, "wilson_95": audit.wilson(60, 100)}
template = {"role": "candidate", "split": "test", "identity": dict(dev),
    "training": {"recipe": "synthetic_only", "recipe_signature": "synthetic", "training_seed": 11,
        "checkpoint_state_sha256": "state11", "model_spec": {"synthetic": True}, "dataset_sha256": "shared"},
    "results": [{"color": color, **metrics} for color in (1, -1)]}
items = []
for seed, score in ((11, .5), (22, .6), (33, .7)):
    item = copy.deepcopy(template); item["training"].update(training_seed=seed, checkpoint_state_sha256=f"state{seed}")
    for row in item["results"]: row["score"] = score
    items.append(item)
summary = audit.aggregate(items)
assert len(summary) == 1 and summary[0]["n_training_seeds"] == 3
assert abs(summary[0]["rows"][0]["metrics"]["score"]["mean"] - .6) < 1e-12
assert abs(summary[0]["rows"][0]["metrics"]["score"]["sample_standard_deviation"] - .1) < 1e-12
assert summary[0]["paired_game_seed_reuse_across_training_seeds"]
assert summary[0]["rows"][0]["per_seed"][0]["wilson_95"] == audit.wilson(60, 100)
passed("three actual-seed sample SD, per-color Wilson and paired seed annotation")
refuses(lambda: audit.aggregate(items + [copy.deepcopy(items[0])]), "same training seed")
cloned = copy.deepcopy(items)
cloned[1]["training"]["checkpoint_state_sha256"] = cloned[0]["training"]["checkpoint_state_sha256"]
refuses(lambda: audit.aggregate(cloned), "Identical model states")
passed("duplicate seed and same-weight false replication rejected")
alternate = copy.deepcopy(items[0]); alternate["identity"]["mode"] = "guarded"
assert len(audit.aggregate([items[0], alternate])) == 2
assert all(group["rows"][0]["metrics"]["score"]["sample_standard_deviation"] is None
           for group in audit.aggregate([items[0], alternate]))
passed("same checkpoint across different wrappers allowed and n1 SD null")

for relative, expected_seed in (("runs/production/a2c4_seed22/best.pt", 22),
                                ("runs/production/az_width128_fixed/best.pt", 11)):
    checkpoint = ROOT / relative
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with ExitStack() as stack:
        provenance = audit.training_provenance(checkpoint, payload,
            {"role": "contextual_reference", "recipe": "legacy_smoke", "training_seed": expected_seed},
            VISUAL, ROOT, False, stack)
    assert provenance["training_seed"] == expected_seed and provenance["legacy_config_provenance"]
    assert provenance["parameter_count"] > 0
passed("legacy standard A2C and AlphaZero config provenance")
reference = audit.deterministic_reference({"mode": "minimax", "checkpoint_sha256": None},
    {"role": "contextual_reference", "recipe": "minimax_reference"})
assert reference["training_seed"] is None and reference["parameter_count"] == 0
passed("Minimax contextual reference has no invented training seed")

zero = ROOT / "runs/a2c_research_20261020/stage4/residual_tactical10_seed22/train/candidates/episode_000000.pt"
payload = torch.load(zero, map_location="cpu", weights_only=True)
with ExitStack() as stack:
    provenance = audit.training_provenance(zero, payload, {}, VISUAL, ROOT, False, stack)
assert provenance["selected_checkpoint_stage"] == "demonstration_pretraining_only"
assert provenance["selected_candidate_has_A2C_updates"] is False
assert provenance["demonstration_or_warm_start_origin"]["selected_state_equals_initial"]
passed("episode-zero selection retains CE lineage without A2C gain claim")

output = VISUAL / "audit_final_tool_selfchecks.json"
report = {"passed": True, "synthetic_only_for_statistics_and_random": True,
          "final_test_results_read_or_generated": False, "checks": checks,
          "random_replay": random_result, "tool_sha256": audit.digest(audit.__file__)}
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("SELF CHECKS SAVED", output, "checks", len(checks))
