# Leaderboard: dev split, deterministic

Reference for paired comparison: `baseline_t1`. Difference: paired bootstrap 95% CI of the Task 1 score; p: exact McNemar on decision correctness.

| Run | System | Temp | Task 1 score (95% CI) | Mean case | F1(yes) | Sens | Spec | Tools/case | Invalid | vs reference | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_t1 | baseline | 1.0 | 0.662 (0.571–0.746) | 0.520 | 0.804 | 1.000 | 0.217 | 2.15 | 0 | — | — |

## Tool usage (share of cases)

| Run | family_history | laboratory_results | previous_notes | psa_trend | radiology_report | Recall of reference |
|---|---|---|---|---|---|---|
| baseline_t1 | 0.13 | 0.25 | 0.17 | 0.60 | 1.00 | 0.603 |
