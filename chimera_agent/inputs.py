import json
import shutil
from pathlib import Path

PROMPT = "structured-prompt.json"
CLINICAL = "prostate-biopsy-decision-clinical-data.json"
EMBEDDING = "prostate-modality-level-neural-representations.json"


def select_cases(split_file: Path, requested: list[str], all_dev: bool, all_test: bool) -> list[str]:
    split = json.loads(split_file.read_text())
    if sum((bool(requested), all_dev, all_test)) > 1:
        raise ValueError("Use only one of --case, --all-dev and --all-test")
    if all_test:
        return sorted(split["test"])
    dev = split["dev"]
    cases = sorted(dev) if all_dev else requested or [sorted(dev)[0]]
    if len(set(cases)) != len(cases):
        raise ValueError("A case was requested more than once")
    for case_id in cases:
        if case_id not in dev or Path(case_id).name != case_id:
            raise ValueError(f"Select cases from the committed dev split: {case_id}")
    return cases


def stage_case(source: Path, destination: Path, case_id: str) -> Path:
    case = source / case_id
    if case.is_symlink():
        raise ValueError("Symlinked case directories are not allowed")
    target = destination / "task1" / "agent_input" / case_id
    for name in (PROMPT, CLINICAL, EMBEDDING):
        path = case / name
        if path.is_symlink():
            raise ValueError(f"Symlinked input is not allowed: {name}")
        if name == EMBEDDING and not path.exists():
            continue
        if name != EMBEDDING:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"{name} must contain an object")
            if payload.get("case_id", case_id) != case_id:
                raise ValueError(f"{name} has a mismatched case_id")
            if name == PROMPT and payload.get("task") != 1:
                raise ValueError("Only Task 1 inputs are allowed")
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / name)
    return target
