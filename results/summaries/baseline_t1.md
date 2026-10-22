# Run `baseline_t1`

System `baseline`, model `gemma4:e2b-it-q8_0`, temperature 1.0, seed None, split `dev` (60 cases), judge `none`, evaluator `92365b9`, 2026-10-22.

| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |
|---|---|---|---|---|---|---|---|---|
| 0.662 (0.571–0.746) | 0.520 | 0.804 | 0.700 | 1.000 | 0.217 | 0.673 | 1.000 | 0 |

Confusion: TP 37, FP 18, TN 5, FN 0.

Case score components (correct decisions only): confidence 0.774, variable_weight 0.640, important_decisive_factor 0.571, tool 0.854, section_grounding 0.943, rationale —.

| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Negative | 12 | 0.750 | 0.857 | 1.000 | 0.000 |
| None | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| Positive | 32 | 0.531 | 0.667 | 1.000 | 0.118 |

| Confidence | n | Accuracy |
|---|---|---|
| clear | 60 | 0.700 |
| borderline | 0 | — |
| uncertain | 0 | — |

Confidence equal to the reference: 0.650.

Tools per case 2.15 (reference 3.10), recall of the reference's tools 0.603. Share of cases: radiology_report 1.00, psa_trend 0.60, laboratory_results 0.25, previous_notes 0.17, family_history 0.13.
