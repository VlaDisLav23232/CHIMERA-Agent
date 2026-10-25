import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation import official, runs

VARIABLES = ("bx", "fh", "age", "dre", "psa", "vol", "psad", "cspca", "pirads", "comorbidity")
WEIGHTS = ("not_used", "noted", "important", "decisive")
CONFIDENCES = ("clear", "borderline", "uncertain")
OUTPUT = official.ROOT / "configs" / "calibration_task1.json"


def upper_median(levels: list[str]) -> str:
    # The official weight score is a mean absolute ordinal error, which a median minimises; of two middle levels the
    # higher is taken, so the choice does not depend on case order.
    ranks = sorted(WEIGHTS.index(level) for level in levels)
    return WEIGHTS[ranks[len(ranks) // 2]]


def mode(values: list[str], order: tuple[str, ...]) -> str:
    counts = Counter(values)
    return max(order, key=lambda v: (counts[v], -order.index(v)))


def fit(references: list[dict]) -> dict:
    table = {"fitted_on": "dev", "cases": {}, "confidence": {}, "variable_weights": {}}
    for decision in ("yes", "no"):
        group = [r for r in references if r["biopsy_decision"] == decision]
        if not group:
            raise ValueError(f"no dev reference with decision {decision!r}")
        table["cases"][decision] = len(group)
        table["confidence"][decision] = mode([r["confidence"] for r in group], CONFIDENCES)
        # A variable the reference left out of its weights counts as not_used, as in the official evaluator.
        table["variable_weights"][decision] = {
            v: upper_median([r["variable_weights"].get(v, "not_used") for r in group]) for v in VARIABLES}
    return table


def main() -> None:
    p = argparse.ArgumentParser(description="Fit agent_v3's confidence and weight table on the dev references.")
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    references = official.load_references()
    table = fit([references[c] for c in runs.split_cases("dev")])
    a.output.write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
