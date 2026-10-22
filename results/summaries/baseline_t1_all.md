# Run `baseline_t1_all`

System `baseline`, model `gemma4:e2b-it-q8_0`, temperature 1.0, seed None, split `all` (91 cases), judge `none`, evaluator `92365b9`, 2026-10-22.

| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |
|---|---|---|---|---|---|---|---|---|
| 0.657 (0.582–0.722) | 0.513 | 0.800 | 0.692 | 1.000 | 0.200 | 0.667 | 1.000 | 0 |

Confusion: TP 56, FP 28, TN 7, FN 0.

Case score components: confidence 0.762, variable_weight 0.635, important_decisive_factor 0.562, tool 0.871, section_grounding 0.950, rationale —.

| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Negative | 18 | 0.722 | 0.839 | 1.000 | 0.000 |
| None | 24 | 1.000 | 1.000 | 1.000 | 1.000 |
| Positive | 49 | 0.531 | 0.667 | 1.000 | 0.115 |

| Confidence | n | Accuracy |
|---|---|---|
| clear | 91 | 0.692 |
| borderline | 0 | — |
| uncertain | 0 | — |

Confidence equal to the reference: 0.637.

Tools per case 2.10 (reference 3.13), recall of the reference's tools 0.599. Share of cases: radiology_report 1.00, psa_trend 0.64, laboratory_results 0.24, previous_notes 0.13, family_history 0.09.
