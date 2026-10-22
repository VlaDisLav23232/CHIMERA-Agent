# CHIMERA-Agent
AI Semester Project

Link to the competition: https://chimera-agent.grand-challenge.org/

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`

Evaluation: `bash scripts/fetch_evaluator.sh`, then `python -m evaluation.score <run_dir> --name <run> --system <system> --temperature <t>` (see [evaluation/README.md](evaluation/README.md)).

Local agent scaffold: see [chimera_agent/README.md](chimera_agent/README.md).
