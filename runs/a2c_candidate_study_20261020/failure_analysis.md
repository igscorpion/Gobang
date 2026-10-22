# Failure trajectory diagnostics

72 original losses were replayed. Both the original A2C action and each independently seeded Greedy action reproduced exactly.

```json
{
  "counts": {
    "agent_positions": 1123,
    "ranking_positions": 875,
    "ranking_disagreements": 640,
    "actual_outside_candidates": 0,
    "budget_exhausted": 172,
    "reference_positions": 216,
    "reference_omissions": 11,
    "reference_omissions_singleton": 10
  },
  "by_reason": {
    "block_immediate_win": {
      "positions": 162,
      "mean_network_value": 0.1498436910778652,
      "ranking_disagreements": 0
    },
    "defend_forcing_threat": {
      "positions": 287,
      "mean_network_value": 0.16788661820028716,
      "ranking_disagreements": 132
    },
    "quiet_candidates": {
      "positions": 152,
      "mean_network_value": 0.16550981690862068,
      "ranking_disagreements": 121
    },
    "tactical_candidates": {
      "positions": 450,
      "mean_network_value": 0.1467291410929627,
      "ranking_disagreements": 387
    },
    "unavoidable_loss": {
      "positions": 72,
      "mean_network_value": 0.17446877458132803,
      "ranking_disagreements": 0
    }
  },
  "losses": 72,
  "limitations": [
    "All positions come from previously observed losses; no causal attribution is possible.",
    "Minimax depth2 is a pruned heuristic reference and its omitted choices are not certified rescues.",
    "Mean values on failure positions do not measure global calibration.",
    "The old 400-game sample is now development information for this research; future improved models need fresh final-test seeds."
  ]
}
```

The network and hand ranking disagree frequently, but only paired development games can test whether replacement helps. Ten reference omissions occur at singleton candidate sets, so changing ranking cannot affect those choices.

## Inspectable examples

Action indices are zero-based row*9+column. X=black, O=white. This is the board BEFORE the recorded action; a different action is not claimed to save the game.

### freestyle, color=-1, game=10, seed=2100010, ply=24

Observed=2; neural top=2; hand top=2; candidates=[2]; reason=tactical_candidates.
Minimax reference=58; reference in candidates=False; reference value=-980.0.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . . . .
2  . . . . . . . X .
3  . . . X . . O . .
4  . O X O O O O X .
5  . O X O O O X . .
6  . X . O . . . . .
7  . X X X . . . . .
8  . X . . . . . . .
9  . X . . . . . . .
```

### freestyle, color=-1, game=16, seed=2100016, ply=26

Observed=54; neural top=54; hand top=54; candidates=[54]; reason=tactical_candidates.
Minimax reference=61; reference in candidates=False; reference value=-1010.0.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . . . .
2  . . . . . . . . .
3  . . . . O . . . .
4  . . . O X O O O .
5  . . . . X O X . .
6  . . . O X X X X O
7  . . . . X X X . .
8  . . . X O . X X .
9  . . O . . . O . O
```

### freestyle, color=-1, game=41, seed=2100041, ply=18

Observed=54; neural top=54; hand top=54; candidates=[54]; reason=tactical_candidates.
Minimax reference=60; reference in candidates=False; reference value=-1010.0.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . . . .
2  . . . . . . . . .
3  . . . . . . . . .
4  . . . . O . . . .
5  . O . . . O . . .
6  . . X O O O X . .
7  . . . X X X . . .
8  . . . O X X X X O
9  . . . . . . . . .
```

### freestyle, color=-1, game=79, seed=2100079, ply=16

Observed=64; neural top=64; hand top=64; candidates=[64]; reason=tactical_candidates.
Minimax reference=48; reference in candidates=False; reference value=-1000.0.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . . . .
2  . . . . . . . . .
3  . . . . . . . . .
4  . . . . . . . . .
5  . . X O O O O X .
6  . . . . . O . X .
7  . . . O X X X X O
8  . . . . . X . . .
9  . . . . . . . . .
```

### renju, color=1, game=99, seed=2200099, ply=15

Observed=47; neural top=47; hand top=16; candidates=[8, 16, 47, 52]; reason=defend_forcing_threat.
Minimax reference=not run; reference in candidates=not run; reference value=not run.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . . . .
2  . . . . . O . . .
3  . O . . . X X . .
4  . . . O X X X O .
5  . . . . X X . . .
6  . . . O . O O . .
7  . . . . . . . . .
8  . . . . . . . . .
9  . . . . . . . . .
```

### renju, color=1, game=85, seed=2200085, ply=27

Observed=58; neural top=58; hand top=29; candidates=[14, 15, 25, 28, 29, 34, 47, 52, 57, 58, 61, 66, 71]; reason=tactical_candidates.
Minimax reference=29; reference in candidates=True; reference value=-60.0.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . . . .
2  . . . . O . . O O
3  . . O O X O X . .
4  . . . X X X O . .
5  . . O X X X O X .
6  . . . X O X X . .
7  . . O . . O X . .
8  . . . . . . . O .
9  . . . . . . . . .
```

### renju, color=1, game=84, seed=2200084, ply=39

Observed=60; neural top=60; hand top=56; candidates=[56, 60]; reason=defend_forcing_threat.
Minimax reference=56; reference in candidates=True; reference value=90.0.

```
   1 2 3 4 5 6 7 8 9
1  . O . . O . X X .
2  . . X . X O . O O
3  . . . X O O O O X
4  . . . O X X O O .
5  . . O X X X X O X
6  . X O X X X O X .
7  . . . O O O . . .
8  . . . . X . . . .
9  . . . . . . . . .
```

### renju, color=1, game=82, seed=2200082, ply=21

Observed=60; neural top=60; hand top=21; candidates=[7, 13, 17, 21, 25, 26, 35, 52, 57, 58, 59, 60]; reason=tactical_candidates.
Minimax reference=21; reference in candidates=True; reference value=-1010.0.

```
   1 2 3 4 5 6 7 8 9
1  . . . . . . O . O
2  . . . . . O X X .
3  . . . . . X X . .
4  . . . O X X X O .
5  . . . O X X X O .
6  . . . O . O O . .
7  . . . . . . . . .
8  . . . . . . . . .
9  . . . . . . . . .
```
