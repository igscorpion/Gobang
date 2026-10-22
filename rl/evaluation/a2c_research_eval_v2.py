"""Versioned extension of the frozen research evaluator for guarded A2C.

The original evaluator source is never changed. Its serial run loop, CSVs,
statistics and exact frozen-test identity checks are reused through a scoped
factory substitution in this *single evaluation process*. Parallel jobs must
use independent processes, not threads calling run(). Both extension sources
are included in the immutable protocol/source guard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from rl.evaluation import a2c_research_eval as original
from rl.training.common import atomic_json, exclusive_run

_ORIGINAL_MAKE_AGENT = original.make_agent
_ORIGINAL_SOURCE_HASHES = original.source_hashes
MODES = ("policy", "guarded", "tactical", "hand", "search", "minimax", "alphazero")


def make_agent(args):
    if args.mode != "guarded":
        return _ORIGINAL_MAKE_AGENT(args)
    from rl.agents.a2c_guarded import GuardedA2CAgent
    policy = original.load_policy(args.checkpoint, args.network_kind, args.device, args.rules)
    return GuardedA2CAgent(policy, rules=args.rules)


def source_hashes(args):
    hashes = _ORIGINAL_SOURCE_HASHES(args)
    for name in ("rl/agents/a2c_guarded.py", "rl/evaluation/a2c_research_eval_v2.py"):
        hashes[name] = original.digest(original.ROOT / name)
    return hashes


def run(args):
    if args.mode not in MODES:
        raise ValueError("Unsupported v2 policy mode")
    if args.split == "validation" and (args.seed_black, args.seed_white) != (3_100_000, 3_200_000):
        raise ValueError("This development study uses seeds 3100000/3200000 only")
    # original.run resolves these globals at call time. Restoration on failure
    # prevents a subsequent original evaluation from silently using v2 policy.
    with patch.object(original, "make_agent", make_agent), \
            patch.object(original, "source_hashes", source_hashes):
        return original.run(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--network-kind", choices=("standard", "research"), default="research")
    p.add_argument("--mode", choices=MODES, default="guarded")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--rules", choices=("freestyle", "renju"), default="freestyle")
    p.add_argument("--opponent", choices=("greedy", "random"), default="greedy")
    p.add_argument("--split", choices=("validation", "test", "probe"), default="validation")
    p.add_argument("--frozen-selection")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--seed-black", type=int, default=3_100_000)
    p.add_argument("--seed-white", type=int, default=3_200_000)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--nodes", type=int, default=2000)
    p.add_argument("--search-candidates", type=int, default=8)
    p.add_argument("--tactical-candidates", type=int, default=12)
    p.add_argument("--value-weight", type=float, default=0.0)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--output", required=True)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.mode != "minimax" and not args.checkpoint:
        p.error("This mode requires --checkpoint")
    with exclusive_run(Path(args.output) / "eval.lock"):
        try:
            run(args)
        except BaseException as error:
            atomic_json(Path(args.output) / "status.json", {"state": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
