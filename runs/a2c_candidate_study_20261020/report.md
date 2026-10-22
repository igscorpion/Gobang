# A2C candidate-ranking development study

Frozen seed22/19000 weights; no training. These repeatedly usable development seeds are not a final held-out test.

Direct/certified tactical branches retain the original A2C action. Only ordinary multi-candidate ranking changes.
Hand score is exactly Greedy._features weighted by GREEDY_WEIGHTS; both rankings break ties by smallest action.
Each branch uses the same TacticalEngine, but after a different choice subsequent boards and candidate sets can diverge.

| Rule | Color | Ranking | W/D/L | Win rate | Wilson95 | Score |
|---|---|---|---|---|---|---|
| freestyle | 1 | a2c | 29/5/6 | 72.5% | 57.2%–83.9% | 0.787 |
| freestyle | 1 | hand | 19/14/7 | 47.5% | 32.9%–62.5% | 0.650 |
| freestyle | -1 | a2c | 13/15/12 | 32.5% | 20.1%–48.0% | 0.512 |
| freestyle | -1 | hand | 11/21/8 | 27.5% | 16.1%–42.8% | 0.537 |
| renju | 1 | a2c | 28/10/2 | 70.0% | 54.6%–81.9% | 0.825 |
| renju | 1 | hand | 15/18/7 | 37.5% | 24.2%–53.0% | 0.600 |
| renju | -1 | a2c | 17/16/7 | 42.5% | 28.5%–57.8% | 0.625 |
| renju | -1 | hand | 12/23/5 | 30.0% | 18.1%–45.4% | 0.588 |

Paired score differences (hand minus A2C):

```json
[
  {
    "rules": "freestyle",
    "color": 1,
    "pairs": 40,
    "score_difference_hand_minus_a2c": -0.1375,
    "paired_sample_sd": 0.4802176322866273,
    "hand_better": 5,
    "equal": 19,
    "hand_worse": 16
  },
  {
    "rules": "freestyle",
    "color": -1,
    "pairs": 40,
    "score_difference_hand_minus_a2c": 0.025,
    "paired_sample_sd": 0.5986094998689324,
    "hand_better": 16,
    "equal": 11,
    "hand_worse": 13
  },
  {
    "rules": "renju",
    "color": 1,
    "pairs": 40,
    "score_difference_hand_minus_a2c": -0.225,
    "paired_sample_sd": 0.42290206176626033,
    "hand_better": 4,
    "equal": 19,
    "hand_worse": 17
  },
  {
    "rules": "renju",
    "color": -1,
    "pairs": 40,
    "score_difference_hand_minus_a2c": -0.0375,
    "paired_sample_sd": 0.4855277329743168,
    "hand_better": 10,
    "equal": 15,
    "hand_worse": 15
  }
]
```

Loss replays are observational. Ranking disagreements and Minimax reference omissions do not prove causes of a defeat.
Minimax uses depth2, candidate_limit8, neighborhood2 and alpha-beta; it is a heuristic reference, not a solver.
Renju uses original freestyle-trained weights and rule-aware candidate legality. No result here establishes multi-seed trained superiority.

## Loss replay freestyle/1

```json
{
  "mode": "replay",
  "rules": "freestyle",
  "color": 1,
  "losses": 16,
  "counts": {
    "agent_positions": 265,
    "ranking_positions": 212,
    "disagreements": 152,
    "budget_exhausted": 33,
    "actual_outside_candidates": 0,
    "reason:quiet_candidates": 43,
    "reason:tactical_candidates": 99,
    "reason:defend_forcing_threat": 72,
    "reference_positions": 48,
    "reference_action_omitted": 0,
    "reason:block_immediate_win": 35,
    "reason:unavoidable_loss": 16
  },
  "limitation": "Observed loss trajectories only. Minimax depth2 is a pruned heuristic reference, not ground truth or a counterfactual win label."
}
```

## Loss replay freestyle/-1

```json
{
  "mode": "replay",
  "rules": "freestyle",
  "color": -1,
  "losses": 26,
  "counts": {
    "agent_positions": 367,
    "ranking_positions": 270,
    "disagreements": 202,
    "budget_exhausted": 72,
    "actual_outside_candidates": 0,
    "reason:quiet_candidates": 38,
    "reason:tactical_candidates": 151,
    "reason:defend_forcing_threat": 89,
    "reference_positions": 78,
    "reference_action_omitted": 6,
    "reason:block_immediate_win": 63,
    "reason:unavoidable_loss": 26
  },
  "limitation": "Observed loss trajectories only. Minimax depth2 is a pruned heuristic reference, not ground truth or a counterfactual win label."
}
```

## Loss replay renju/1

```json
{
  "mode": "replay",
  "rules": "renju",
  "color": 1,
  "losses": 19,
  "counts": {
    "agent_positions": 317,
    "ranking_positions": 261,
    "disagreements": 179,
    "budget_exhausted": 33,
    "actual_outside_candidates": 0,
    "reason:quiet_candidates": 54,
    "reason:tactical_candidates": 127,
    "reason:defend_forcing_threat": 81,
    "reference_positions": 57,
    "reference_action_omitted": 1,
    "reason:block_immediate_win": 36,
    "reason:unavoidable_loss": 19
  },
  "limitation": "Observed loss trajectories only. Minimax depth2 is a pruned heuristic reference, not ground truth or a counterfactual win label."
}
```

## Loss replay renju/-1

```json
{
  "mode": "replay",
  "rules": "renju",
  "color": -1,
  "losses": 11,
  "counts": {
    "agent_positions": 174,
    "ranking_positions": 132,
    "disagreements": 107,
    "budget_exhausted": 34,
    "actual_outside_candidates": 0,
    "reason:quiet_candidates": 17,
    "reason:tactical_candidates": 73,
    "reason:defend_forcing_threat": 45,
    "reason:block_immediate_win": 28,
    "reference_positions": 33,
    "reference_action_omitted": 4,
    "reason:unavoidable_loss": 11
  },
  "limitation": "Observed loss trajectories only. Minimax depth2 is a pruned heuristic reference, not ground truth or a counterfactual win label."
}
```

## Reproduction and audit

```powershell
.\.venv\Scripts\python.exe -u -m rl.evaluation.a2c_candidate_study --games 40 --workers 4 --output runs/a2c_candidate_study_fresh
```

Use a fresh output directory. The original baseline protocol must still match the current core source. Frozen source copies are archived in frozen_source/ with the protocol SHA256 values.

See [failure_analysis.md](failure_analysis.md) for inspectable boards and [audit.json](audit.json) for the completed 320-game artifact replay: 14,947 legal moves, 7,458 seeded Greedy actions exactly reproduced, and 7,489 candidate-ranking selections checked.

The fixed hand score uses contiguous-run features, not a complete rule-aware tactical evaluator; legality comes from the shared tactical candidate set. Its smallest-action tie break is one fixed design, not every possible hand ranking.
