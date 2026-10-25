# agent_v2 results (Task 1, dev split)

agent_v2 runs the agent_v1 graph, tools, decision step, checks and trace unchanged; only the prompts differ. The
agent's system prompt lists only the Task 1 tools, drops the hint that the previous notes duplicate the PSA trend,
and explains that a positive prior biopsy means cancer is already diagnosed, so a new biopsy is recommended when it
would change management. The decision step gets the same framing and definitions of clear, borderline and uncertain
confidence.

Setup as for agent_v1 (see [agent_v1.md](agent_v1.md)). Run `agent_v2_dev`, commit `1b6b706`, 60/60 cases
succeeded, 103 min. The comparison run `agent_v1_fixed` (commit `40ce449`) ran in a separate Kaggle session. Its
agent_v1 behaviour is identical to agent_v1 at `1b6b706`: the decision prompt is pinned by a hash test and the agent
prompt is the unchanged baseline prompt.

## Decision quality

| Dev, 60 cases | Baseline | agent_v1_dev | agent_v1_fixed | agent_v2 |
|---|---|---|---|---|
| Task 1 score (95% CI) | 0.662 (0.571–0.746) | 0.652 (0.557–0.738) | 0.669 (0.575–0.750) | 0.660 (0.564–0.745) |
| Mean case score | 0.520 | 0.512 | 0.528 | 0.524 |
| F1(yes) | 0.804 | 0.791 | 0.809 | 0.795 |
| Accuracy | 0.700 | 0.667 | 0.700 | 0.700 |
| Sensitivity / specificity | 1.000 / 0.217 | 0.973 / 0.174 | 0.973 / 0.261 | 0.946 / 0.304 |
| TP / FP / TN / FN | 37 / 18 / 5 / 0 | 36 / 19 / 4 / 1 | 36 / 17 / 6 / 1 | 35 / 16 / 7 / 2 |
| Invalid outputs | 0 | 2 | 2 | 0 |

Paired differences (95% bootstrap CI, exact McNemar p on correctness):

- agent_v2 − agent_v1_fixed: −0.009 (−0.048…+0.028), p 1.000. The two runs differ in 4 of 60 decisions; correctness
  changes in 2 cases, one each way.
- agent_v2 − baseline: −0.003 (−0.052…+0.044), p 1.000.

Patients with a prior positive biopsy (32 cases; reference 17 no, 15 yes):

| | TN | FP | TP | FN | Invalid |
|---|---|---|---|---|---|
| agent_v1_fixed | 2 | 14 | 14 | 0 | 2 |
| agent_v2 | 3 | 14 | 13 | 2 | 0 |

Among the cases both runs answered, agent_v2 gained 1 true negative and lost 1 true positive. The false positives in
this group did not go down.

## Confidence

| | Clear | Borderline | Uncertain | Exact agreement | Weighted kappa | Confidence component (correct cases) |
|---|---|---|---|---|---|---|
| Reference | 39 | 11 | 10 | — | — | — |
| agent_v1_fixed | 57 | 1 | 0 | 0.617 | −0.021 | 0.762 |
| agent_v2 | 57 | 3 | 0 | 0.633 | −0.022 | 0.762 |

Kappa is quadratic-weighted, computed with the official evaluator's aggregate. The confidence guidance barely changed
the output: the model still answers `clear` in 57 of 60 cases.

## Tool use

| | agent_v1_fixed | agent_v2 |
|---|---|---|
| Cases reading the previous notes | 15% | 92% |
| Cases attempting the non-existent pathology tool | 14 of 58 | 0 of 60 |
| Cases searching the guidelines | 46 of 58 | 33 of 60 |
| Tools per case | 1.93 | 2.60 |
| Recall of the reference's tools | 0.558 | 0.734 |
| Tool precision component (correct cases) | 0.884 | 0.858 |
| LLM calls per case | 4.29 | 5.02 |

The prompt changed what the agent does: it now reads the notes almost always and no longer looks for a pathology
tool. It did not change what it decides.

## Evidence checks (from the traces)

| | agent_v1_fixed | agent_v2 |
|---|---|---|
| Claims | 200 | 214 |
| Claims whose cited evidence IDs all exist | 100% | 99.5% |
| Claims whose numbers all appear in the cited evidence | 86.5% | 90.2% |

## Verdict

- Improved over agent_v1_fixed (higher Task 1 score, at least 3 more true negatives, F1 loss ≤ 0.03, sensitivity
  ≥ 0.90): **no**. The score is 0.009 lower and true negatives rose by 1.
- Significant: no.
- Non-inferior to the baseline (CI lower bound > −0.05, no more invalid outputs): **not met, narrowly**: the lower
  bound is −0.052; invalid outputs 0 vs 0.
- Confidence (kappa above agent_v1_fixed, component within 0.03): **no**. Kappa is unchanged.

This is a negative result. agent_v1 stays the main system.

## Interpretation and caveats

- The agent read the previous notes in almost every case and still recommended a biopsy for most prior-positive
  patients whose reference answer is "no". Together with the earlier check that these patients look alike on PI-RADS,
  PSA density, csPCa and age, this suggests the reference's "no" in this group is not recoverable from the notes or
  the framing with this model.
- The run-to-run variation at temperature 1.0 is about 4-6 decisions per 60 cases (agent_v1_dev vs agent_v1_fixed: 6),
  so a prompt effect of 1-2 decisions cannot be separated from sampling noise with one run per system.
- The prompts were written after looking at dev errors, so dev numbers are optimistic. The test split has not been
  run.
- The clinical framing and the confidence guidance were changed together in one run; their effects are not
  separable.
- agent_v2 had no invalid outputs, against 2 for each agent_v1 run; with these counts that can be chance.
