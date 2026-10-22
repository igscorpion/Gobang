import argparse
import json

import pytest

from rl.evaluation.a2c_research_eval import validate_split


def settings(**changes):
    values = dict(network_kind="standard",mode="tactical",device="cpu",rules="freestyle",
        opponent="greedy",games=100,seed_black=8100000,seed_white=8200000,depth=2,nodes=2000,
        search_candidates=8,tactical_candidates=12,value_weight=0.0,sims=400,
        split="test",frozen_selection=None)
    return argparse.Namespace(**(values | changes))


def test_validation_cannot_accidentally_consume_final_seeds():
    with pytest.raises(ValueError,match="reserved"):
        validate_split(settings(split="validation"),"weight",{})
    assert validate_split(settings(split="validation",seed_black=3100000,
                                   seed_white=3200000),"weight",{}) is None


def test_final_requires_full_policy_identity(tmp_path):
    args = settings(frozen_selection=str(tmp_path/"selection.json"))
    identity = {k:v for k,v in vars(args).items() if k not in ("split","frozen_selection")}
    identity.update(checkpoint_sha256="weight",source_sha256={"agent.py":"source"})
    (tmp_path/"selection.json").write_text(json.dumps({"status":"frozen",
        "selected_using":"development only","evaluations":[identity]}),encoding="utf-8")
    assert validate_split(args,"weight",{"agent.py":"source"})["identity"] == identity
    for changed in (settings(frozen_selection=args.frozen_selection,mode="search"),
                    settings(frozen_selection=args.frozen_selection,nodes=4000),
                    settings(frozen_selection=args.frozen_selection,games=200)):
        with pytest.raises(ValueError,match="Complete evaluated policy"):
            validate_split(changed,"weight",{"agent.py":"source"})
    with pytest.raises(ValueError,match="Complete evaluated policy"):
        validate_split(args,"different weight",{"agent.py":"source"})
