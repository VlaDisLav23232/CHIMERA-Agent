# Run `agent_v1_fixed`

System `agent_v1`, model `gemma4:e2b-it-q8_0`, temperature 1.0, seed None, split `dev` (60 cases), judge `none`, evaluator `92365b9`, 2026-10-25.

| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |
|---|---|---|---|---|---|---|---|---|
| 0.669 (0.575–0.750) | 0.528 | 0.809 | 0.700 | 0.973 | 0.261 | 0.692 | 1.000 | 2 |

Confusion: TP 36, FP 17, TN 6, FN 1.

Case score components (correct decisions only): confidence 0.762, variable_weight 0.739, important_decisive_factor 0.581, tool 0.884, section_grounding 0.834, rationale —.

| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Negative | 12 | 0.833 | 0.900 | 1.000 | 0.333 |
| None | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| Positive | 32 | 0.500 | 0.651 | 0.933 | 0.118 |

| Confidence | n | Accuracy |
|---|---|---|
| clear | 57 | 0.719 |
| borderline | 1 | 1.000 |
| uncertain | 0 | — |

Confidence equal to the reference: 0.617.

Tools per case 1.93 (reference 3.10), recall of the reference's tools 0.558. Share of cases: radiology_report 0.97, psa_trend 0.62, previous_notes 0.15, laboratory_results 0.13, family_history 0.07.
