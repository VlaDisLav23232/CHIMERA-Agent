import json
import re
from pathlib import Path

from evaluation.official import DATA_DIR, ROOT

SPLIT_FILE = ROOT / "splits" / "task1.json"
CASES_DIR = ROOT / "results" / "runs"
SUMMARIES_DIR = ROOT / "results" / "summaries"
DECISION_FILE = "prostate-biopsy-decision.json"
REASONING_FILE = "prostate-biopsy-decision-reasoning.json"


def split_cases(split: str) -> list[str]:
    data = json.loads(SPLIT_FILE.read_text())
    return sorted(data["dev"] + data["test"]) if split == "all" else data[split]


def prior_biopsy(case_id: str) -> str:
    return json.loads((DATA_DIR / case_id / "structured-prompt.json").read_text())["bx"]


def read_prediction(run_dir: Path, case_id: str, reference: dict) -> dict | None:
    case_dir = run_dir / "output" / "task1" / case_id
    try:
        prediction = json.loads((case_dir / REASONING_FILE).read_text())
        decision = json.loads((case_dir / DECISION_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(prediction, dict):
        return None
    if not isinstance(decision, str):
        return None
    if not isinstance(prediction.get("confidence"), str):
        return None
    weights = prediction.get("variable_weights")
    if not isinstance(weights, dict) or not all(isinstance(v, str) for v in weights.values()):
        return None
    reveals = prediction.get("reveal_sequence")
    if not isinstance(reveals, list) or not all(isinstance(v, str) for v in reveals):
        return None
    if not isinstance(prediction.get("free_text"), str):
        return None
    prediction["biopsy_decision"] = decision
    prediction["case_id"] = case_id
    prediction["clinical_data"] = reference["clinical_data"]
    return prediction


def read_run_info(run_dir: Path) -> dict:
    path = run_dir / "run_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def check_name(name: str) -> None:
    # Names become path components under results/; anything else could escape it.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError(f"Run name {name!r} must contain only letters, numbers, underscores or hyphens")


def write_cases(name: str, cases: list[dict]) -> None:
    check_name(name)
    CASES_DIR.joinpath(name).mkdir(parents=True, exist_ok=True)
    (CASES_DIR / name / "cases.json").write_text(json.dumps(cases, indent=1))


def read_cases(name: str) -> list[dict]:
    check_name(name)
    return json.loads((CASES_DIR / name / "cases.json").read_text())


def write_summary(name: str, summary: dict, markdown: str) -> None:
    check_name(name)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARIES_DIR / f"{name}.json").write_text(json.dumps(summary, indent=1))
    (SUMMARIES_DIR / f"{name}.md").write_text(markdown)


def read_summaries() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(SUMMARIES_DIR.glob("*.json"))]
