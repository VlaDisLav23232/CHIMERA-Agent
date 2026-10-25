# agent_v1 results (Task 1, dev split)

agent_v1 keeps the baseline's ReAct loop, MCP tools, prompts and model, and replaces the final form-filling step.
The decision step sees the case record and every retrieved document and guideline passage under an ID (`P`, `E1…`,
`G1…`) and writes 2-4 claims that cite them. Citation, number and weight checks are recorded per case in a trace.

Setup: Ollama `gemma4:e2b-it-q8_0`, temperature 1.0, top_p 0.95, context 16384, max 25 graph steps, supplied
guideline index, Kaggle 2× T4, one worker process per case (≤ 600 s). Dev split: 60 cases. Deterministic official
scoring (evaluator `92365b9`, no LLM judge).

## Runs

| Run | Commit | Cases succeeded | Time |
|---|---|---|---|
| `agent_v1_dev` (first run) | `c32912f` | 58/60 | 100 min |
| `agent_v1_fixed` (re-run with the apostrophe repair and the stricter number check) | `40ce449` | 58/60 | 106 min |

Both runs lost 2 cases to the same failure: the decision answer was not valid JSON because of an invalid escape
inside a string, three attempts in a row. In the first run it was `\'`, which the re-run repairs. In the re-run the
repair never triggered (0 repairs in 58 traces), so the remaining failures are a different invalid escape that the
log does not show. Invalid outputs count as wrong decisions in all metrics below.

## Decision quality versus the baseline

| Dev, 60 cases | Baseline | agent_v1_dev | agent_v1_fixed |
|---|---|---|---|
| Task 1 score (95% CI) | 0.662 (0.571–0.746) | 0.652 (0.557–0.738) | 0.669 (0.575–0.750) |
| Paired difference vs baseline (95% CI) | — | −0.011 (−0.047…+0.017) | +0.006 (−0.031…+0.039) |
| McNemar p (correctness) | — | 0.500 | 1.000 |
| Mean case score | 0.520 | 0.512 | 0.528 |
| F1(yes) | 0.804 | 0.791 | 0.809 |
| Accuracy | 0.700 | 0.667 | 0.700 |
| Sensitivity / specificity | 1.000 / 0.217 | 0.973 / 0.174 | 0.973 / 0.261 |
| TP / FP / TN / FN | 37 / 18 / 5 / 0 | 36 / 19 / 4 / 1 | 36 / 17 / 6 / 1 |
| Invalid outputs | 0 | 2 | 2 |

The two agent_v1 runs differ in 6 of 60 decisions. At temperature 1.0 this run-to-run variation is as large as the
differences to the baseline, so agent_v1's decision quality is not distinguishable from the baseline's.

By prior biopsy (agent_v1_fixed): accuracy None 1.000 (16 cases), Negative 0.833 (12), Positive 0.500 (32). The
errors are false positives in patients with a prior positive biopsy, as for the baseline.

## Reasoning fields (correct decisions only)

| Component | Baseline | agent_v1_fixed |
|---|---|---|
| Confidence | 0.774 | 0.762 |
| Variable weights vs reference | 0.640 | 0.739 |
| Important/decisive factor F1 | 0.571 | 0.581 |
| Tool precision | 0.854 | 0.884 |
| Section grounding | 0.943 | 0.834 |

agent_v1 weights more variables, which brings its weights closer to the reference. Section grounding drops because
the scorer counts a weighted prior biopsy (`bx`) as grounded only if a pathology report was revealed, which Task 1
has no tool for, and a weighted DRE only if the lab results were revealed.

## Evidence checks (from the traces)

| | agent_v1_fixed |
|---|---|
| Cases with a trace | 58 |
| Claims | 200 |
| Claims whose cited evidence IDs all exist | 100% |
| Claims whose numbers all appear in the cited evidence | 86.5% |
| Cases with weights on unavailable variables | 0 |
| Mean decision attempts | 1.05 |

The number check ignores the fixed scale legends of the case template. The first run reported 90% because those
legend digits still counted as evidence. The check is literal: a number written in words is not seen, and a number
can match the cited text by coincidence.

The baseline's explanation cites nothing, so it cannot be checked this way. Whether agent_v1's explanations contain
fewer unsupported claims than the baseline's is measured by the manual audit (10 dev cases, both systems, two
annotators), which is still pending.

## Acceptance

- At most 1 failed case in the full run: not met (2 failed in each run).
- Decision quality not worse than the baseline: the score part is met (CI lower bound −0.031 > −0.05); the
  invalid-output part is not (2 vs 0).
- Faithfulness audit: pending.

## Notes

- Gemma tried to call a pathology report tool in 14 of 58 cases. The baseline system prompt mentions one, but Task 1
  does not provide it; the call returns an error and is not counted as a reveal.
- The scaffold's guideline search through MCP never reached the embedding service, because the tool server did not
  inherit the socket path. agent_v1 passes it explicitly; the baseline numbers above come from the organizers'
  unchanged run, which used the default path and is not affected.
- Earlier exploratory analysis used all 91 labeled cases; the test split has not been run.
