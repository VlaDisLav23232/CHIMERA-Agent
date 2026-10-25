# Run `agent_v3_dev`

System `agent_v3`, model `gemma4:e2b-it-q8_0`, temperature 1.0, seed None, split `dev` (60 cases), judge `none`, evaluator `92365b9`, 2026-10-25.

| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |
|---|---|---|---|---|---|---|---|---|
| 0.707 (0.614–0.790) | 0.595 | 0.818 | 0.733 | 0.973 | 0.348 | 0.706 | 0.889 | 0 |

Confusion: TP 36, FP 15, TN 8, FN 1.

Case score components (correct decisions only): confidence 0.773, variable_weight 0.780, important_decisive_factor 0.666, tool 0.879, section_grounding 1.000, rationale —.

| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Negative | 12 | 0.917 | 0.947 | 1.000 | 0.667 |
| None | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| Positive | 32 | 0.531 | 0.651 | 0.933 | 0.176 |

| Confidence | n | Accuracy |
|---|---|---|
| clear | 60 | 0.733 |
| borderline | 0 | — |
| uncertain | 0 | — |

Confidence equal to the reference: 0.650.

Tools per case 3.00 (reference 3.10), recall of the reference's tools 0.884. Share of cases: radiology_report 1.00, psa_trend 1.00, previous_notes 1.00.
