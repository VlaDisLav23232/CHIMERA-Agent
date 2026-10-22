#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/fetch_baseline.sh
if [ "${1:-}" = "--models" ]; then
  ollama pull gemma4:e2b-it-q8_0
  .venv/bin/python -c 'from dotenv import load_dotenv; import runpy; load_dotenv(".env"); runpy.run_path("external/chimera-agent-baseline/scripts/download_embedding_model.py", run_name="__main__")' --output-dir "$PWD/external/chimera-agent-baseline/model"
else
  [ "$#" -eq 0 ] || { echo 'Usage: setup_agent.sh [--models]' >&2; exit 1; }
  uv pip install --python .venv/bin/python -r requirements-agent.txt -c requirements.txt
  uv pip check --python .venv/bin/python
fi
