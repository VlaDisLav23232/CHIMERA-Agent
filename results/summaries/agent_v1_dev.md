# Run `agent_v1_dev`

System `agent_v1`, model `gemma4:e2b-it-q8_0`, temperature 1.0, seed None, split `dev` (60 cases), judge `none`, evaluator `92365b9`, 2026-10-24.

| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |
|---|---|---|---|---|---|---|---|---|
| 0.652 (0.557–0.738) | 0.512 | 0.791 | 0.667 | 0.973 | 0.174 | 0.667 | 1.000 | 2 |

Confusion: TP 36, FP 19, TN 4, FN 1.

Case score components (correct decisions only): confidence 0.775, variable_weight 0.751, important_decisive_factor 0.610, tool 0.898, section_grounding 0.831, rationale —.

| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Negative | 12 | 0.750 | 0.900 | 1.000 | 0.000 |
| None | 16 | 0.875 | 0.923 | 0.923 | 0.667 |
| Positive | 32 | 0.531 | 0.667 | 1.000 | 0.118 |

| Confidence | n | Accuracy |
|---|---|---|
| clear | 57 | 0.702 |
| borderline | 1 | 0.000 |
| uncertain | 0 | — |

Confidence equal to the reference: 0.617.

Tools per case 1.93 (reference 3.10), recall of the reference's tools 0.561. Share of cases: radiology_report 0.97, psa_trend 0.58, laboratory_results 0.25, previous_notes 0.10, family_history 0.03.
