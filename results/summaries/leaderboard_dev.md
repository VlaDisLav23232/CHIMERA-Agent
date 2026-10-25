# Leaderboard: dev split, deterministic

Reference for paired comparison: `baseline_t1`. Difference: paired bootstrap 95% CI of the Task 1 score; p: exact McNemar on decision correctness.

| Run | System | Temp | Task 1 score (95% CI) | Mean case | F1(yes) | Sens | Spec | Tools/case | Invalid | vs reference | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agent_v1_fixed | agent_v1 | 1.0 | 0.669 (0.575–0.750) | 0.528 | 0.809 | 0.973 | 0.261 | 1.93 | 2 | +0.006 (-0.031…+0.039) | 1.000 |
| baseline_t1 | baseline | 1.0 | 0.662 (0.571–0.746) | 0.520 | 0.804 | 1.000 | 0.217 | 2.15 | 0 | — | — |
| agent_v2_dev | agent_v2 | 1.0 | 0.660 (0.564–0.745) | 0.524 | 0.795 | 0.946 | 0.304 | 2.60 | 0 | -0.003 (-0.052…+0.044) | 1.000 |
| agent_v1_dev | agent_v1 | 1.0 | 0.652 (0.557–0.738) | 0.512 | 0.791 | 0.973 | 0.174 | 1.93 | 2 | -0.011 (-0.047…+0.017) | 0.500 |

## Tool usage (share of cases)

| Run | family_history | laboratory_results | previous_notes | psa_trend | radiology_report | Recall of reference |
|---|---|---|---|---|---|---|
| agent_v1_dev | 0.03 | 0.25 | 0.10 | 0.58 | 0.97 | 0.561 |
| agent_v1_fixed | 0.07 | 0.13 | 0.15 | 0.62 | 0.97 | 0.558 |
| agent_v2_dev | 0.10 | 0.08 | 0.92 | 0.52 | 0.98 | 0.734 |
| baseline_t1 | 0.13 | 0.25 | 0.17 | 0.60 | 1.00 | 0.603 |
