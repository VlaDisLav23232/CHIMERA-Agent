# CHIMERA-Agent

UCU "Artificial Intelligence" semester project, 2026: Oleksii Lasiichuk, Vladyslav Shymanovskyi.

An LLM agent for Task 1 of the [CHIMERA-Agent challenge](https://chimera-agent.grand-challenge.org/): read a
prostate cancer patient's case, decide whether a **biopsy is recommended** (yes / no) and explain why. The goal is
an agent whose explanation can be checked: every claim tied to the patient data or a guideline passage it came from,
with decision quality at least as good as the organizers' baseline. This is a research prototype, not a clinical tool.

## Systems

| System | What it is |
|---|---|
| `baseline` | The organizers' agent, unchanged: a LangGraph ReAct loop over MCP tools (MRI report, PSA trend, labs, previous notes, family history, guideline search), then a form-filling LLM call. |
| `agent_v1` | Ours. Same loop, tools, prompts and model. The final step is replaced: it sees the case record and every retrieved document and guideline passage under an ID, writes 2-4 claims that cite those IDs, and records citation and number checks in a per-case trace. |
| `agent_v2` | agent_v1 with our own prompts: only the Task 1 tools are described, and a positive prior biopsy is framed as an existing diagnosis. It also adds confidence guidance. |
| `agent_v3` | What the strongest challenge entries do: the MRI report, PSA trend and previous notes are always retrieved, an EAU policy decides (PI-RADS 4-5 biopsy, 1-2 none, 3 by PSA density > 0.20; the model decides only when PI-RADS is missing), confidence and weights come from a table fitted on the dev split, and the model writes the cited claims. |

All systems use Gemma 4 E2B-it (`gemma4:e2b-it-q8_0`) through Ollama, EmbeddingGemma for guideline search and the
supplied EAU guideline index.

**Reused from the organizers:**
- the model choice, the MCP tool server, the guideline index and search;
- the ReAct loop and the baseline prompts;
- the output schema and the official evaluator.

**Ours:**
- the evidence ledger, the citing decision step, the checks and the trace;
- the agent_v2 and agent_v3 prompts, the agent_v3 decision policy and calibration;
- the batch runner with input isolation;
- the dev/test split, the scoring harness with confidence intervals, paired tests and the leaderboard;
- the claim-audit tooling;
- the Kaggle notebooks.

## Results (dev split, 60 cases)

| System | Task 1 score (95% CI) | F1(yes) | Specificity | Invalid outputs |
|---|---|---|---|---|
| baseline | 0.662 (0.571–0.746) | 0.804 | 0.217 | 0 |
| agent_v1 | 0.669 (0.575–0.750) | 0.809 | 0.261 | 2 |
| agent_v2 | 0.660 (0.564–0.745) | 0.795 | 0.304 | 0 |

- **No system is distinguishable from the baseline.** Paired differences are within ±0.01, with 95% CIs spanning zero. Two runs of the same system already differ in about 6 of 60 decisions.
- **agent_v1's explanations can be checked.** All its cited evidence IDs exist, and 86.5% of its claims contain only numbers found in the cited evidence. The baseline's explanation cites nothing. Whether this means fewer unsupported claims is being measured by a manual audit.
- **agent_v2 is a negative result.** The agent now reads the previous notes in 92% of cases, against 15% before. But it still recommends a biopsy for most patients with a prior positive biopsy where the reference says no: it treats a high PI-RADS score as "progression".

Details: [results/agent_v1.md](results/agent_v1.md), [results/agent_v2.md](results/agent_v2.md),
[results/baseline.md](results/baseline.md), and the [leaderboard](results/summaries/leaderboard_dev.md). The test
split (31 cases) has not been run yet.

## Repository layout

```
chimera_agent/   our agent: runner, worker, graph, decision step, evidence ledger, prompts (README inside)
evaluation/      scoring with the official evaluator, metrics, statistics, leaderboard, claim audit (README inside)
tests/           unit tests (synthetic; no patient data)
notebooks/       00 baseline run on Kaggle, 01 exploratory data analysis, 02 and 03 agent_v1 and agent_v2 runs on Kaggle
configs/         agent settings (model, sampling, limits)
splits/          committed dev/test split and the fixed audit cases
scripts/         fetch the pinned baseline and evaluator, set up the agent and models
results/         aggregate reports and per-run summaries (README inside)
```

Downloaded at setup and not committed:
- `external/`: the organizers' baseline and evaluator at pinned commits.
- `data/`: the challenge data, which you must obtain yourself.
- `results/<run>/`: raw runs, which contain patient information.

## Quickstart

Python 3.12 with [uv](https://docs.astral.sh/uv/). Agent runs need Linux, macOS or WSL, and in practice a GPU.

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
bash scripts/fetch_evaluator.sh
bash scripts/setup_agent.sh
.venv/bin/python -m unittest discover -s tests -t .
```

- **Run the agent:** see [chimera_agent/README.md](chimera_agent/README.md). That includes the models, Ollama and `python -m chimera_agent.run --system agent_v1 --all-dev`. Full runs were done on Kaggle with [notebooks/02_agent_v1_run.ipynb](notebooks/02_agent_v1_run.ipynb) and [notebooks/03_agent_v2_run.ipynb](notebooks/03_agent_v2_run.ipynb).
- **Score a run:** see [evaluation/README.md](evaluation/README.md), e.g. `python -m evaluation.score results/<run> --name <run> --system agent_v1 --temperature 1.0`.

Place the authorized challenge data at `data/task1/`.

## Data and evaluation rules

- **The agent only ever receives a case's three input files.** The reference decision and reasoning files are read only by the evaluator, and inference code never imports it.
- **The dev split is used for development and the test split only for final comparisons.** The split is fixed and committed.
- **Exploratory analysis was done on all 91 labeled cases before the split**, so the test split is not a completely unseen holdout.
- **Patient data, raw runs and per-case scores stay local.** Only aggregate summaries are committed.

The project proposal is [proposal_shymanovskyi_lasiichuk.pdf](proposal_shymanovskyi_lasiichuk.pdf).
