# Run `agent_v2_dev`

System `agent_v2`, model `gemma4:e2b-it-q8_0`, temperature 1.0, seed None, split `dev` (60 cases), judge `none`, evaluator `92365b9`, 2026-10-25.

| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |
|---|---|---|---|---|---|---|---|---|
| 0.660 (0.564–0.745) | 0.524 | 0.795 | 0.700 | 0.946 | 0.304 | 0.686 | 0.778 | 0 |

Confusion: TP 35, FP 16, TN 7, FN 2.

Case score components (correct decisions only): confidence 0.762, variable_weight 0.743, important_decisive_factor 0.592, tool 0.858, section_grounding 0.805, rationale —.

| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Negative | 12 | 0.833 | 0.900 | 1.000 | 0.333 |
| None | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| Positive | 32 | 0.500 | 0.619 | 0.867 | 0.176 |

| Confidence | n | Accuracy |
|---|---|---|
| clear | 57 | 0.702 |
| borderline | 3 | 0.667 |
| uncertain | 0 | — |

Confidence equal to the reference: 0.633.

Tools per case 2.60 (reference 3.10), recall of the reference's tools 0.734. Share of cases: radiology_report 0.98, previous_notes 0.92, psa_trend 0.52, family_history 0.10, laboratory_results 0.08.
