# Shallow search on a trained residual10 policy: development study

Same frozen residual10 episode 5000 checkpoint, 40 games per color versus Greedy, freestyle, 3.1m/3.2m.
Original full tactical root set, ordinary non-root candidate cap 8, node budget 2000. No final tests.

| Search | Black W/D/L | White W/D/L | Mean score | Mean win rate | Mean move ms black/white |
|---|---|---|---|---|---|
| depth2_value0 | 8/30/2 | 5/33/2 | 0.5562 | 0.1625 | 202.89/197.66 |
| depth2_value05 | 12/25/3 | 5/24/11 | 0.5188 | 0.2125 | 476.66/425.31 |
| depth3_value0 | 38/1/1 | 13/22/5 | 0.7812 | 0.6375 | 872.25/637.08 |
| depth3_value05 | 8/27/5 | 10/27/3 | 0.5625 | 0.2250 | 1358.83/1136.22 |


Independent audit passed for all 320 games: 18,938 legal moves and 9,458 Greedy actions exactly reconstructed. W/D/L, score, Wilson95, per-move timing, checkpoint and all 16 recorded source SHAs agree. See [audit](audit.json); the earlier partial audit is preserved separately.

The checkpoint SHA is `4df704eee9d0f3b28bd1debcccb1a16ebb5f40bb4bedd67b29e877ec27098f68`. On the same development protocol, this weight with pure policy has black39/0/1, white27/10/3 and mean score .8875; guarded has black39/0/1, white28/9/3 and mean score .89375. The best of these four search settings, depth3/value0, reaches .78125. This evidence does not support adopting these search settings for the main recipe. It does not rule out a different guarded-root search or every possible value calibration.

All four search settings retain the original full-tactics root candidates. The learned value=.5 settings were not beneficial here. Reported timings come from different reached positions and concurrent runs; they describe these evaluations and are not a controlled same-position speed benchmark. The main research continues with three-seed policy/guarded development confirmation. These games are development, not final-test generalization evidence; reserved 8m tests remain unopened.
