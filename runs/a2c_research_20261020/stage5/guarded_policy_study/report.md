# Exact-only guard versus full legal policy: development study

Six frozen checkpoints, 40 games per color versus Greedy, freestyle only. Seeds 3.1m/3.2m.
These development games may select a recipe and must not be presented as final tests.

| Checkpoint | Mode | Black W/D/L | White W/D/L | Mean score | Mean win rate |
|---|---|---|---|---|---|
| demo_regularized_005k | guarded | 4/0/36 | 1/0/39 | 0.0625 | 0.0625 |
| demo_regularized_005k | policy | 0/0/40 | 0/0/40 | 0.0000 | 0.0000 |
| flat10_010k | guarded | 34/5/1 | 29/10/1 | 0.8812 | 0.7875 |
| flat10_010k | policy | 34/5/1 | 25/11/4 | 0.8375 | 0.7375 |
| flat10_pretrain | guarded | 33/6/1 | 16/7/17 | 0.6937 | 0.6125 |
| flat10_pretrain | policy | 30/8/2 | 12/9/19 | 0.6312 | 0.5250 |
| residual10_005k | guarded | 39/0/1 | 28/9/3 | 0.8938 | 0.8375 |
| residual10_005k | policy | 39/0/1 | 27/10/3 | 0.8875 | 0.8250 |
| residual10_010k | guarded | 37/1/2 | 27/8/5 | 0.8562 | 0.8000 |
| residual10_010k | policy | 32/2/6 | 24/10/6 | 0.7750 | 0.7000 |
| residual4_pretrain | guarded | 28/1/11 | 17/1/22 | 0.5750 | 0.5625 |
| residual4_pretrain | policy | 23/1/16 | 14/0/26 | 0.4688 | 0.4625 |
