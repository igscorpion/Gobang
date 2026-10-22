# Explicitly frozen A2C final evaluation queue

`rl.evaluation.a2c_research_final_suite` executes a matrix that root has already
selected and frozen from development results. It does not discover candidates,
choose checkpoints or wrappers, create a frozen selection, or tune from tests.
This document and a `status="planned"` draft cannot start the queue.

No real final evaluation was run while implementing this module. Its tests use
synthetic model metadata, non-8m CSV fixtures, and child processes that only
print text. Reserved-seed checks exercise identity dictionaries only.

## Required frozen file

The top-level object requires `status="frozen"`,
`selected_using="development only"`, `expected_runner_sha256`, `models`,
`evaluations`, and `jobs`. `expected_runner_sha256` must match the exact bytes of
`rl/evaluation/a2c_research_final_suite.py` used to execute it.

Each `evaluations` item has exactly these original evaluator identity fields:

```text
network_kind, mode, device, rules, opponent, games, seed_black, seed_white,
depth, nodes, search_candidates, tactical_candidates, value_weight, sims,
checkpoint_sha256, source_sha256
```

Every job uses CPU inference and 100 games per color. Freestyle uses
8,100,000/8,200,000; renju uses 8,300,000/8,400,000. The complete source dictionary
must exactly match the frozen v2 evaluator for that mode. Different rules or
opponents need separate identities.

`jobs` is a one-to-one mapping onto the identity list. Every object has exactly
`name`, `checkpoint`, `identity_index`, and `context`; names and identities cannot
repeat. Checkpoint and training paths resolve against the project root. Context
requires `role`, `recipe`, `training_seed`, `training_run`,
`training_protocol_sha256`, and `checkpoint_sha256`. Model contexts must agree
with the SHA-indexed `models` entry; `model_spec`, if present, is the complete
actual specification. Selected episode/epoch assertions are verified.

```json
{
  "name": "residual_seed11_guarded_freestyle_greedy",
  "checkpoint": "runs/.../train/candidates/episode_015000.pt",
  "identity_index": 0,
  "context": {
    "role": "candidate",
    "recipe": "residual_tactical10_ce20_a2c_ce01",
    "training_seed": 11,
    "training_run": "runs/.../train",
    "training_protocol_sha256": "<actual protocol SHA256>",
    "checkpoint_sha256": "<actual weight-file SHA256>"
  }
}
```

Minimax is a `contextual_reference` with an explicit recipe. Its checkpoint,
checkpoint SHA, training seed, training run and training protocol SHA are null;
it has no entry in `models`. Legacy A2C/AlphaZero references may use a completed
run's `config.json` as documented provenance. Research candidates require a
completed, unlocked training run, actual seed/spec, unchanged training sources,
immutable full companion checkpoint, and unchanged recorded dataset/initial
weights. Episode-zero candidates must equal their recorded initial weights.

## Validate and execute

Only run the execution command after root has created the actual frozen file.
The plan command validates all sources and model lineage without creating any
experiment output or child process. `-B` also disables Python bytecode caches.

```powershell
.\.venv\Scripts\python.exe -B -m rl.evaluation.a2c_research_final_suite --frozen-selection <frozen-file.json> --output <new-final-output> --workers 2 --plan-only
.\.venv\Scripts\python.exe -u -m rl.evaluation.a2c_research_final_suite --frozen-selection <frozen-file.json> --output <new-final-output> --workers 2
```

The queue creates `suite_protocol.json`, `suite_status.json`, `completed_jobs.json`
and `jobs/<name>/attempt_NNN/`. Each attempt preserves its command, full identity,
context, launcher PID/birth, runtime observations, log, and evaluator artifacts.
Children use hidden processes on Windows. They may still belong to a Codex OS
job; no claim is made that closing the app or shutting down the machine leaves
them running.

## Recovery and verification

Re-run the identical queue command only after checking its real supervisor,
child PID/birth identities, and OS-held locks. No queue-level `--resume` flag is
needed. A live child is reattached; a complete result is accepted only after
identity, manifest, hashes, per-color counts, W/D/L, score, Wilson intervals,
game/move CSV integrity, and latency/reason/node aggregates match. Every known
attempt's writer is checked before any older result can be reused.

Failed partial attempts remain intact. A retry uses the identical frozen policy
in a new directory and reruns all 200 games; results are never merged. The
default limit is three attempts per job. Completion accepts one complete
operational attempt, never the attempt with a better score. A source, weight,
training provenance, companion checkpoint or manifest change stops the queue.

The new module requests Windows delete-sharing for short JSON reads and uses
bounded retries for its own state-file writes on WinError 5. Some Windows
execution environments can still reject replacement while a reader is open.
The original common module and v2 evaluator are unchanged; their failures are
handled through preserved operational attempts rather than silent source edits.

After every frozen job completes, `audit_inputs.json` uses
`format="a2c_independent_audit_inputs_v1"`. Its entries flatten each job's context
and add the actual completed `directory` and `frozen_selection`; those paths and
`training_run` are absolute, because the independent tool interprets relative
input paths against the audit manifest's own directory. Minimax retains null
training provenance.

This queue's validation does not replace the independent legality/terminal and
Greedy/Random RNG replay described in [a2c-final-audit.md](a2c-final-audit.md).
Final reporting and registry changes remain separate work after that audit.
