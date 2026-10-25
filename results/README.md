# Results

Aggregate results only. Raw runs (`results/<run>/`) and per-case scores (`results/runs/<run>/cases.json`) contain
patient information and stay local.

## Reports

| File | Content |
|---|---|
| [baseline.md](baseline.md) | The organizers' baseline on all 91 labeled cases (first run, 18 Oct 2026). |
| [agent_v1.md](agent_v1.md) | agent_v1 on the dev split: both runs, decision quality vs the baseline, reasoning fields, evidence checks. |
| [agent_v2.md](agent_v2.md) | agent_v2 on the dev split vs agent_v1: decisions, prior-positive cases, confidence, tool use, verdict. |

## Summaries

`summaries/` holds one pair of files per scored run. They are written by `python -m evaluation.score`.
- `<run>.md` is readable: settings, metrics with confidence intervals, a breakdown by prior biopsy, confidence and tool use.
- `<run>.json` holds the same data for scripts.

| Run | System | Split | Notes |
|---|---|---|---|
| `baseline_t1` | baseline | dev (60) | the comparison reference |
| `baseline_t1_all` | baseline | all 91 | context only; includes test cases |
| `agent_v1_dev` | agent_v1 | dev | first run, superseded |
| `agent_v1_fixed` | agent_v1 | dev | re-run with the parsing repair; the agent_v1 result |
| `agent_v2_dev` | agent_v2 | dev | the agent_v2 result |

`summaries/leaderboard_dev.md` ranks every dev run, with a paired comparison against the baseline: a bootstrap CI of
the Task 1 score difference, and an exact McNemar test on decision correctness. It is rebuilt with
`python -m evaluation.leaderboard`. Metric definitions are in [evaluation/README.md](../evaluation/README.md).

All runs use `gemma4:e2b-it-q8_0` at temperature 1.0, scored deterministically with the official evaluator (no LLM
judge). At that temperature, two runs of the same system differ in about 6 of 60 decisions, so small differences
between runs are noise.
