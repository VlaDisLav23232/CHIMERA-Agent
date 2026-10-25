# agent_v3 results (Task 1, dev split)

agent_v3 adds three parts to the agent_v1 graph. The MRI report, PSA trend and previous notes are always retrieved
through the MCP tools before the agent runs, and the agent may then only search the guidelines. A PI-RADS rule decides:
PI-RADS 4-5 biopsy, 1-2 none, 3 biopsy if PSA density > 0.20, and the model decides only when PI-RADS is missing.
The 0.20 threshold is the highest PSA-density band of the EAU 2026 risk table and was chosen on dev; the EAU
recommendation itself omits a biopsy at PI-RADS 3 only below 0.10. The
confidence and variable weights come from a table fitted on the dev references (`configs/calibration_task1.json`,
`python -m evaluation.calibrate`); a weight is kept only for a variable whose source document was opened. The model
writes 2-4 claims that cite evidence IDs.

Setup as for agent_v1 (see [agent_v1.md](agent_v1.md)): Ollama `gemma4:e2b-it-q8_0`, temperature 1.0, Kaggle 2× T4.
Commit `ca6a05a`. Deterministic official scoring (evaluator `92365b9`, no LLM judge).

## Runs

| Run | System | Cases succeeded | Median time per case |
|---|---|---|---|
| `agent_v3_dev` | agent_v3 | 60/60 | 65 s |
| `agent_v3_model_decides_dev` | ablation: no decision rule, the model decides | 60/60 | 62 s |

## Decision quality

| Dev, 60 cases | Baseline | agent_v1_fixed | agent_v2 | agent_v3 | v3, model decides |
|---|---|---|---|---|---|
| Task 1 score (95% CI) | 0.662 (0.571–0.746) | 0.669 (0.575–0.750) | 0.660 (0.564–0.745) | 0.707 (0.614–0.790) | 0.705 (0.610–0.787) |
| Paired difference vs baseline (95% CI) | — | +0.006 (−0.031…+0.039) | −0.003 (−0.052…+0.044) | +0.044 (−0.002…+0.091) | +0.043 (−0.013…+0.101) |
| McNemar p (correctness) | — | 1.000 | 1.000 | 0.625 | 0.688 |
| Mean case score | 0.520 | 0.528 | 0.524 | 0.595 | 0.597 |
| F1(yes) | 0.804 | 0.809 | 0.795 | 0.818 | 0.814 |
| TP / FP / TN / FN | 37 / 18 / 5 / 0 | 36 / 17 / 6 / 1 | 35 / 16 / 7 / 2 | 36 / 15 / 8 / 1 | 35 / 14 / 9 / 2 |
| Invalid outputs | 0 | 2 | 0 | 0 | 0 |

- agent_v3 − agent_v2: +0.047 (+0.019…+0.086).
- agent_v3 − model decides: +0.002 (−0.044…+0.047); the two differ in 4 decisions. The decision rule adds nothing
  measurable; the gain over the baseline comes from the reasoning fields.
- agent_v3's decisions differ from the baseline's only at PI-RADS 3 (4 cases): 3 became correct, 1 wrong.
- By prior biopsy: accuracy None 1.000 (16 cases), Negative 0.917 (12), Positive 0.531 (32), as for every system.

## Reasoning fields (correct decisions only)

| Component | Baseline | agent_v3 |
|---|---|---|
| Confidence | 0.774 | 0.773 |
| Variable weights vs reference | 0.640 | 0.780 |
| Important/decisive factor F1 | 0.571 | 0.666 |
| Tool precision | 0.854 | 0.879 |
| Section grounding | 0.943 | 1.000 |

## Evidence checks (from the traces)

| | agent_v3 |
|---|---|
| Claims | 200 |
| Claims whose cited evidence IDs all exist | 100% |
| Claims whose numbers all appear in the cited evidence | 87.5% |
| Mean decision attempts | 1.02 |
| LLM calls per case (median) | 2 |

## Caveats

- The calibration table was fitted on these 60 dev cases and the PSA-density threshold was chosen after looking at
  them, so the dev score is optimistic. The held-out test split (31 cases) is the comparison that counts.
- The calibrated weights are typical values per decision, not per-case reasoning; the claims are per case.
- With 60 cases and temperature 1.0, run-to-run variation is 4-6 decisions; the paired CI against the baseline still
  includes zero.
- Patients with a positive prior biopsy remain at chance for every method tried (prompts, regression, trees, a text
  model, guideline features such as PSA doubling time: AUC 0.44-0.62 on the 32 dev cases); the deciding facts, the
  earlier grade and a comparison with the earlier MRI, are missing from most records.
