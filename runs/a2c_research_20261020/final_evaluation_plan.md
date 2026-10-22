# Final evaluation plan — development prerequisites pending

Status: planned, NOT a frozen selection; no final-test games have been opened. Written 2026-10-21 while three-seed training and development selection are still running. This document cannot authorize a test invocation by itself.

## Development selection

- Finish all four search cells and the independent replay audit; preserve negative results and measured costs. Search in this study retains full-tactics root filtering, so it is not a general test of every possible search wrapper.
- Finish residual/tactical10 seeds 11/22/33, each CE20 plus 30k A2C+CE0.1, and the flat/tactical10 seed22 extension. Preserve original 10k directories and equal-budget results.
- Complete the immutable 0/5k/.../30k development candidates with policy and guarded wrappers, each 40 games per color, Greedy seeds 3100000/3200000. Exact reused cases remain reuse, not independent training.
- Select each seed/mode checkpoint by mean black/white score, then mean win rate, then earlier checkpoint. Determine one common residual wrapper from the three real training seeds using the selector's declared rule. Do not mix different wrappers under one recipe.
- Freeze one showcase checkpoint for the web registry using development results only. Record the training seed, episode, full ModelSpec, state/weight SHA, common expert data, optimizer continuation and complete training provenance. A selected episode-zero checkpoint is imitation pretraining, not evidence of A2C improvement.

## Intended test matrix

All test jobs use frozen v2 identities and CPU inference. Each job contains 100 black and 100 white games. Freestyle seeds are 8100000/8200000; renju seeds are 8300000/8400000. No policy or model selection may use those outcomes.

1. Primary residual recipe: three separately trained seeds, each against Greedy and Random under freestyle and renju (2400 games).
2. Wrapper ablation: the same three selected checkpoints with the alternate policy/guarded wrapper, same opponents/rules/seeds (2400 games). This holds weights fixed; it is distinct from selecting a separate best checkpoint for each wrapper.
3. Original enhanced A2C reference: original seed22/19000 weight and original tactical semantics, Greedy, both rules (400 games). The old 400 games used for development are not substituted here.
4. Flat/tactical10 30k secondary reference: one seed22 checkpoint and wrapper chosen solely on development, Greedy, both rules (400 games). This is a single-seed secondary reference, not three-seed architecture evidence.
5. Existing AlphaZero400 and current configured Minimax reference: Greedy, both rules, 400 games per method, if their rule/weight/source preflight succeeds. Record actual registry/default Minimax depth and candidate budget in the freeze rather than inventing a stronger setting. Their training initialization and inference budgets differ from A2C; they are contextual references.

The intended maximum is 6400 games. The final manifest must specify the actual matrix and give any development-stage scope change before the first test. Reference results never change the primary recipe, weights, wrapper, or showcase choice.

## Freeze and execution gate

- Require training/selection completion, successful complete development audits, unchanged source/weights, explicit rules and exact compute configuration.
- Preserve a full `models` provenance section and the selected recipe, plus the evaluator-required `evaluations` identities: network_kind, mode, device, rules, opponent, games, seed_black, seed_white, depth, nodes, search_candidates, tactical_candidates, value_weight, sims, checkpoint_sha256 and source_sha256.
- Manifest status must be frozen with selected_using=development only. Each test command uses --split test --frozen-selection and a new independent output directory. Do not pass this planned document as the freeze.
- Preserve failed attempts. A deterministic operational retry may use a new attempt directory with the identical frozen identity; audit its move/outcome prefix and use exactly one complete attempt. Never merge duplicate games or treat retries as independent evidence.
- Active sources are immutable. Any necessary implementation fix requires an explicit version and snapshot at a safe boundary, followed by a new freeze before testing. Do not silently change the source of an ongoing experiment.

## Independent verification and reporting

Replay every completed game, verify legal moves and terminal outcomes, and independently reconstruct Greedy/Random RNG actions from the recorded per-game seed. Check W/D/L, score=(wins+0.5*draws)/games, win-rate Wilson95, source/weight/dataset SHAs and complete identities. Report per-model/per-seed/per-color results and cost. Across the three real training seeds, use the mean and sample standard deviation; shared opponent game seeds are paired and must not be treated as 300 fully independent games for an artificially narrow pooled uncertainty interval.

Report the original model's 1,405,042 parameters and residual10's 305,538 parameters with convolutional/FC allocation and measured inference work. Attribute the new system to tactical features, spatial residual architecture, common AlphaZero demonstrations, A2C fine-tuning and the selected rule wrapper together. Do not claim that network size alone caused the improvement. Renju is rule-aware transfer from freestyle-trained weights, not dedicated renju training; the 9x9 rule experiment is not a full tournament opening protocol.

Only after all frozen evaluations and independent audits complete: update the final report, experiments/a2c_research.json, docs/a2c-research.md, README.md and rl/README.md; register the new showcase under a separate name while preserving a2c4_tactical. Do not stop the web service, other authorized queues, or upload to GitHub. Delete automation a2c only when all required work is complete.
