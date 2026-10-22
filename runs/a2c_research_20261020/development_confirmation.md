# Three-seed confirmation: development selection complete

All four training runs reached 30,000 episodes; the queue and selector exited naturally. The original phase4 10k directories remain preserved. No final-test games have been opened.

56 cases contain 4,480 evaluated game records, including six exact reused 80-game cases. Reuse is explicitly recorded and does not constitute extra games or independent training seeds. Every row below uses Greedy freestyle development seeds 3100000/3200000, 40 games per color.

| Model / wrapper | Selected episode | Black W/D/L | White W/D/L | Mean score | Mean win rate |
|---|---:|---|---|---:|---:|
| residual_tactical10_seed11 / policy | 30000 | 39/0/1 | 33/4/3 | 0.92500 | 0.90000 |
| residual_tactical10_seed11 / guarded | 30000 | 40/0/0 | 35/4/1 | 0.96250 | 0.93750 |
| residual_tactical10_seed22 / policy | 15000 | 39/0/1 | 28/8/4 | 0.88750 | 0.83750 |
| residual_tactical10_seed22 / guarded | 15000 | 39/0/1 | 31/6/3 | 0.91250 | 0.87500 |
| residual_tactical10_seed33 / policy | 20000 | 35/4/1 | 34/4/2 | 0.91250 | 0.86250 |
| residual_tactical10_seed33 / guarded | 25000 | 36/2/2 | 35/5/0 | 0.93125 | 0.88750 |
| flat_tactical10_seed22 / policy | 25000 | 35/2/3 | 34/3/3 | 0.89375 | 0.86250 |
| flat_tactical10_seed22 / guarded | 30000 | 37/2/1 | 37/1/2 | 0.94375 | 0.92500 |

The common residual wrapper selected using the three independently trained seeds is **guarded**. Its mean development score is .9354166667 (sample SD .0252590743), and mean win rate is .9000 (sample SD .0330718914). Policy selected separately per seed has score .9083333333 and win rate .8666666667. These are development-selected estimates, not held-out test results.

The policy and guarded best checkpoints can differ (seed33 policy20k versus guarded25k). Their separate-best comparison does not isolate the wrapper effect. The planned final ablation uses the same primary selected checkpoint for both wrappers.

The planned showcase is residual10 seed11 episode30k + guarded, selected by development score/win rate before final tests. Shared expert data, original 10k-to30k continuation for seed22, and single-seed flat22 reference are disclosed; flat22 is not a fourth residual training replicate.

Independent complete development replay audit has passed all 56 cases: 140,829 legal moves and 69,662 exact Greedy reconstructions. Six reused cases (480 game records) are explicitly marked; the other 4,000 game records are newly evaluated complete cases. Weight/source/spec/dataset/companion and selection checks all passed. See [complete audit](confirmation/selection/audit.json). The final-runner review must still finish before a final manifest is frozen. See [planned matrix](final_evaluation_plan.md), [selection JSON](confirmation/selection/selection.json), and [auditor documentation](../../docs/a2c-final-audit.md).
