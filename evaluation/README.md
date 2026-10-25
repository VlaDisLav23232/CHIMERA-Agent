# Evaluation

Scores Task 1 runs with the organizers' official evaluator and adds clinical metrics, confidence intervals and a leaderboard.

## Setup

```
bash scripts/fetch_evaluator.sh                                    # organizers' evaluator, pinned commit, into external/
uv pip install --python .venv/bin/python -r requirements-judge.txt # only for --judge
```

## Commands

```
python -m evaluation.split                                         # fixed dev/test split -> splits/task1.json (already committed)
python -m evaluation.score <run_dir> --name <run> --system <system> --temperature <t>
python -m evaluation.leaderboard
python -m evaluation.calibrate                                     # agent_v3 table -> configs/calibration_task1.json
```

`calibrate` reads the dev references only: per decision, the most frequent confidence and the upper median of each
variable weight. agent_v3 reads the JSON at inference; the test split never enters it.

`<run_dir>` holds `output/task1/<case>/` with the two official output files, and optionally `run_info.json` (model tag, code commit).

`score` options: `--split dev|test|all` (default `dev`; `test` and `all` need `--confirm-test`), `--seed N`, `--judge` (needs Ollama with `gemma4:e4b`; another model via `JUDGE_MODEL` is marked unofficial). `leaderboard` options: `--split`, `--judge`, `--reference <run>` (default: the only `baseline` run).

## Claim audit

A manual check of explanation claims on a fixed set of 10 dev cases (`splits/audit_task1.json`), for the baseline and an agent run:

```
python -m evaluation.audit sheet results/baseline_run results/<agent_run>   # blinded sheets in results/audit/ (local)
python -m evaluation.audit summary                                           # unsupported-claim share per system, agreement, kappa
python -m evaluation.audit checks results/<agent_run>                        # aggregate of the automatic trace checks
```

Each `free_text` is split into sentences with the same rule for both systems. The rows are shuffled and the system is hidden.
Each annotator labels every row `supported` or `unsupported` against the case's input files, without opening `key.csv`.
Disagreements go into `results/audit/agreed.csv` (`row_id,label`).

## Outputs

- `results/summaries/<run>.md|json`: run settings and aggregate metrics (committed).
- `results/summaries/leaderboard_<split>.md`: all runs of a split ranked by Task 1 score (committed).
- `results/runs/<run>/cases.json`: per-case results (local only, reveals labels per patient).

## Metrics

- **Task 1 score**: official, (mean case score + F1(yes)) / 2. A wrong or missing decision gives the case 0; a correct one is scored on confidence, variable weights, decisive factors, tool efficiency, grounding and, with `--judge`, the explanation.
- **95% CI**: percentile bootstrap over cases, 2000 resamples.
- **Sensitivity / specificity / PPV / NPV**: "yes" = biopsy recommended. A missing or invalid output counts as an error.
- **vs reference**: paired bootstrap CI of the Task 1 score difference on the same cases; **p**: exact McNemar test on decision correctness.
- **Tools**: share of cases opening each document, and how many of the reference's documents were opened.

Repeated-run statistics require model, code commit and `config_id` in the run metadata.
Use the same `config_id` only when prompts, tools and generation settings match
(except the seed). Runs without this metadata remain individually ranked.
Groups also match evaluator version and judge mode.

A judge failure aborts scoring before any results are written; no deterministic
scores are substituted inside judged runs. Malformed output objects count as failures.
