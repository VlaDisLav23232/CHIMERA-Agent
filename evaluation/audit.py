import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from evaluation.runs import REASONING_FILE, ROOT, SPLIT_FILE, prior_biopsy

AUDIT_FILE = ROOT / "splits" / "audit_task1.json"
AUDIT_DIR = ROOT / "results" / "audit"
SEED = 42
N_AUDIT = 10
LABELS = ("supported", "unsupported")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def select_cases(dev: list[str], bx: dict[str, str]) -> list[str]:
    # Drawn by prior biopsy only, before any agent output is read, so the choice cannot follow results.
    strata = defaultdict(list)
    for case_id in sorted(dev):
        strata[bx[case_id]].append(case_id)
    rng = random.Random(SEED)
    chosen = []
    for key in sorted(strata):
        cases = strata[key]
        rng.shuffle(cases)
        chosen += cases[: round(len(cases) * N_AUDIT / len(dev))]
    if len(chosen) != N_AUDIT:
        raise ValueError(f"Stratified rounding gave {len(chosen)} cases, expected {N_AUDIT}")
    return sorted(chosen)


def sentences(text: str) -> list[str]:
    # One splitting rule for both systems, so claim counts are comparable.
    return [s for s in SENTENCE_END.split(text.strip()) if s]


def build_sheet(texts: dict[tuple[str, str], str]) -> tuple[list[dict], list[dict]]:
    """texts maps (system, case_id) to free_text; returns blinded rows and the key."""
    entries = [(system, case_id, i, s) for (system, case_id), text in sorted(texts.items())
               for i, s in enumerate(sentences(text))]
    random.Random(SEED).shuffle(entries)
    rows = [{"row_id": n, "case_id": c, "claim": s, "label": "", "note": ""} for n, (_, c, _, s) in enumerate(entries)]
    key = [{"row_id": n, "system": system, "case_id": c, "sentence": i} for n, (system, c, i, _) in enumerate(entries)]
    return rows, key


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    expected = sum((a.count(label) / len(a)) * (b.count(label) / len(b)) for label in LABELS)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def summarize(key: list[dict], sheets: dict[str, list[dict]], agreed: dict[str, str]) -> dict:
    labels = {name: {r["row_id"]: r["label"].strip().lower() for r in rows} for name, rows in sheets.items()}
    for name, by_row in labels.items():
        bad = sorted((r for r, v in by_row.items() if v not in LABELS), key=int)
        if bad or set(by_row) != {k["row_id"] for k in key}:
            raise ValueError(f"Sheet {name}: rows missing or not labelled {'/'.join(LABELS)}: {bad}")
    first, second = sorted(labels)
    disputed = sorted((r for r in labels[first] if labels[first][r] != labels[second][r]), key=int)
    unresolved = [r for r in disputed if agreed.get(r) not in LABELS]
    if unresolved:
        raise ValueError(f"Resolve disagreements in agreed.csv for rows: {unresolved}")
    final = {r: agreed.get(r, labels[first][r]) for r in labels[first]}
    systems = {}
    for system in sorted({k["system"] for k in key}):
        rows = [k["row_id"] for k in key if k["system"] == system]
        unsupported = sum(final[r] == "unsupported" for r in rows)
        systems[system] = {
            "cases": len({k["case_id"] for k in key if k["system"] == system}), "claims": len(rows),
            "unsupported": unsupported, "unsupported_share": unsupported / len(rows),
            "by_annotator": {n: sum(labels[n][r] == "unsupported" for r in rows) / len(rows) for n in (first, second)},
        }
    order = sorted(labels[first], key=int)
    return {"systems": systems, "rows": len(order), "disagreements": len(disputed),
            "raw_agreement": 1 - len(disputed) / len(order),
            "cohen_kappa": cohen_kappa([labels[first][r] for r in order], [labels[second][r] for r in order])}


def check_summary(traces: list[dict]) -> dict:
    claims = [(t, i) for t in traces for i in range(len(t["decision"]["claims"]))]
    missing = {(t["case_id"], c["claim"]) for t in traces for c in t["checks"]["missing_citations"]}
    numbers = {(t["case_id"], c["claim"]) for t in traces for c in t["checks"]["unsupported_numbers"]}
    return {
        "cases": len(traces), "claims": len(claims),
        "citations_present_share": 1 - len(missing) / len(claims),
        "numbers_supported_share": 1 - len(numbers) / len(claims),
        "cases_with_ungrounded_weights": sum(bool(t["checks"]["ungrounded_weights"]) for t in traces),
        "mean_decision_attempts": sum(t["decision_attempts"] for t in traces) / len(traces),
    }


def cmd_select(_: argparse.Namespace) -> None:
    if AUDIT_FILE.exists():
        raise SystemExit(f"{AUDIT_FILE} exists; the audit set is fixed once")
    dev = json.loads(SPLIT_FILE.read_text())["dev"]
    cases = select_cases(dev, {c: prior_biopsy(c) for c in dev})
    AUDIT_FILE.write_text(json.dumps({"seed": SEED, "cases": cases}, indent=1) + "\n")
    print(f"{len(cases)} audit cases -> {AUDIT_FILE}")


def cmd_sheet(a: argparse.Namespace) -> None:
    cases = json.loads(AUDIT_FILE.read_text())["cases"]
    texts, missing = {}, []
    for system, run_dir in (("baseline", a.baseline_run), ("agent_v1", a.agent_run)):
        for case_id in cases:
            path = run_dir / "output" / "task1" / case_id / REASONING_FILE
            if path.exists():
                texts[(system, case_id)] = json.loads(path.read_text())["free_text"]
            else:
                missing.append(f"{system}:{case_id}")
    if missing:
        raise SystemExit(f"Audit cases without output: {missing}")
    rows, key = build_sheet(texts)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in a.annotators:
        write_csv(AUDIT_DIR / f"sheet_{name}.csv", rows)
    write_csv(AUDIT_DIR / "key.csv", key)
    print(f"{len(rows)} claims from {len(cases)} cases -> {AUDIT_DIR}. Label each row supported or unsupported "
          "against the case's input files; do not open key.csv before both sheets are done.")


def cmd_summary(a: argparse.Namespace) -> None:
    sheets = {name: read_csv(AUDIT_DIR / f"sheet_{name}.csv") for name in a.annotators}
    agreed_path = AUDIT_DIR / "agreed.csv"
    agreed = {r["row_id"]: r["label"].strip().lower() for r in read_csv(agreed_path)} if agreed_path.exists() else {}
    print(json.dumps(summarize(read_csv(AUDIT_DIR / "key.csv"), sheets, agreed), indent=1))


def cmd_checks(a: argparse.Namespace) -> None:
    traces = [json.loads(p.read_text()) for p in sorted((a.run_dir / "trace").glob("*.json"))]
    if not traces:
        raise SystemExit(f"No traces under {a.run_dir / 'trace'}")
    print(json.dumps(check_summary(traces), indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description="Manual claim audit of Task 1 explanations.")
    sub = p.add_subparsers(required=True)
    sub.add_parser("select", help="fix the audit cases from the dev split").set_defaults(func=cmd_select)
    sheet = sub.add_parser("sheet", help="blinded labelling sheets for the audit cases")
    sheet.add_argument("baseline_run", type=Path)
    sheet.add_argument("agent_run", type=Path)
    summary = sub.add_parser("summary", help="unsupported-claim shares and annotator agreement")
    for parser in (sheet, summary):
        parser.add_argument("--annotators", nargs=2, default=["vladyslav", "oleksii"])
    sheet.set_defaults(func=cmd_sheet)
    summary.set_defaults(func=cmd_summary)
    checks = sub.add_parser("checks", help="aggregate the deterministic trace checks of a run")
    checks.add_argument("run_dir", type=Path)
    checks.set_defaults(func=cmd_checks)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
