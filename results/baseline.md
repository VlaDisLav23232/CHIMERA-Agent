# Baseline results (Task 1)

## Baseline (unchanged, Ollama gemma4:e2b-it-q8_0 on Kaggle T4): 18 Oct 2026

Run setup (from `run_info.json`):
- Baseline commit `1afa788efc21c0e0144706d75170b0a4cd267367` ([DIAGNijmegen/chimera-agent-baseline](https://github.com/DIAGNijmegen/chimera-agent-baseline))
- Ollama 0.40.6, model `gemma4:e2b-it-q8_0` (ID `8cb19ff6df84`, 6.0 GB), context length 16384
- Generation: baseline defaults (temperature 1.0, top_p 0.95, max_new_tokens 4096)
- Kaggle, 2× Tesla T4, scope `"labeled"` (91 cases), batches of 10

| Metric | Baseline | Always "yes" |
|---|---|---|
| Coverage | 91/91 | 91/91 |
| Accuracy | 0.692 | 0.615 |
| Precision (yes) | 0.667 | 0.615 |
| Recall (yes) | 1.000 | 1.000 |
| F1 (yes) | 0.800 | 0.762 |
| Time per case | ~79 s (incl. ~25 s startup per batch); ~2 h total | — |

Confusion (ref → pred): yes→yes 56, no→yes 28, no→no 7, yes→no 0.

By prior biopsy result (`bx`):

| bx | n | Ref "yes" | Pred "yes" | Accuracy | F1 (yes) |
|---|---|---|---|---|---|
| None (no prior biopsy) | 24 | 20 | 20 | 1.000 | 1.000 |
| Negative | 18 | 13 | 18 | 0.722 | 0.839 |
| Positive | 49 | 23 | 46 | 0.531 | 0.667 |

Agent behaviour:
- Confidence: `clear` on all 91 cases (never hedges).
- Tools revealed per case: 2.1 on average. `radiology_report` 91, `psa_trend` 58, `laboratory_results` 22, `previous_notes` 12, `family_history` 8.

Notes:
- No failed cases: every batch finished, and no single-case retries were needed.
- The baseline almost never says "no": 28 of 35 "no" cases are false positives. It does only slightly better than always answering "yes".
- The errors are concentrated in patients with a **positive prior biopsy** (accuracy 0.53). That's the main place to improve.
- Metrics computed locally from the downloaded outputs with cell C of [RUN_ALL_CASES.md](../RUN_ALL_CASES.md), since the zip was made before cell C ran.
- Not yet the official Task 1 score (PLAN task 1.6).
