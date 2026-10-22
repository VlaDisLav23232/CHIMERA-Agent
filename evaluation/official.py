import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DIR = ROOT / "external" / "CHIMERA-agent" / "evaluation"
DATA_DIR = ROOT / "data" / "task1"

if not (EVALUATOR_DIR / "evaluate.py").exists():
    sys.exit("Official evaluator not found: run scripts/fetch_evaluator.sh")

os.environ["USE_RATIONALE_JUDGE"] = "0"
os.environ["SECTION_MAPPING_FILE"] = str(EVALUATOR_DIR / "ground_truth" / "section_variable_mapping.json")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
sys.path.insert(0, str(EVALUATOR_DIR))
import evaluate as _ev  # noqa: E402

COMMIT = subprocess.run(
    ["git", "-C", str(EVALUATOR_DIR), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, check=True,
).stdout.strip()
JUDGE_MODEL = _ev.JUDGE_MODEL
OFFICIAL_JUDGE_MODEL = "gemma4:e4b"


def load_references() -> dict[str, dict]:
    # The loader prints a warning for every unlabeled case; those are expected.
    with contextlib.redirect_stdout(io.StringIO()):
        records = _ev.load_ground_truth_records(DATA_DIR, "task1")
    return {r["case_id"]: r for r in records}


def make_judge():
    try:
        _ev.requests.get(_ev.OLLAMA_BASE_URL, timeout=3)
    except _ev.requests.ConnectionError:
        sys.exit(f"LLM judge unavailable: Ollama is not running at {_ev.OLLAMA_BASE_URL}")
    _ev.USE_RATIONALE_JUDGE = True
    judge = _ev.build_rationale_judge()
    if judge is None:
        sys.exit(f"LLM judge unavailable: start Ollama with {JUDGE_MODEL} and install requirements-judge.txt")
    return judge


def score_case(reference: dict, prediction: dict | None, judge=None) -> dict:
    row = _ev.evaluate_case(reference, prediction, None, judge)
    if judge is not None and row["gate"] == "passed" and row["rationale_score"] is None:
        raise RuntimeError("LLM judge returned no score; run aborted to avoid mixing scoring modes")
    return row


def aggregate(rows: list[dict]) -> dict:
    return _ev.compute_aggregate_metrics(rows)
