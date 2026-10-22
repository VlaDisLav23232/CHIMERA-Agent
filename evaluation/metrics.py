from collections import Counter, defaultdict
from statistics import mean

CONFIDENCE_LEVELS = ("clear", "borderline", "uncertain")


def f1_yes(cases: list[dict]) -> float:
    tp = sum(c["pred"] == "yes" and c["ref"] == "yes" for c in cases)
    fp = sum(c["pred"] == "yes" and c["ref"] == "no" for c in cases)
    fn = sum(c["pred"] != "yes" and c["ref"] == "yes" for c in cases)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def task1_score(cases: list[dict]) -> float:
    return (mean(c["case_score"] for c in cases) + f1_yes(cases)) / 2


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def clinical(cases: list[dict]) -> dict:
    # An invalid output is an error in either direction: a missed biopsy or an unnecessary one.
    tp = sum(c["pred"] == "yes" and c["ref"] == "yes" for c in cases)
    tn = sum(c["pred"] == "no" and c["ref"] == "no" for c in cases)
    fp = sum(c["pred"] != "no" and c["ref"] == "no" for c in cases)
    fn = sum(c["pred"] != "yes" and c["ref"] == "yes" for c in cases)
    return {
        "n": len(cases), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "invalid": sum(c["pred"] not in ("yes", "no") for c in cases),
        "accuracy": _ratio(tp + tn, len(cases)),
        "f1_yes": f1_yes(cases),
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, sum(c["pred"] == "yes" for c in cases)),
        "npv": _ratio(tn, sum(c["pred"] == "no" for c in cases)),
    }


def by_prior_biopsy(cases: list[dict]) -> dict:
    groups = defaultdict(list)
    for c in cases:
        groups[c["bx"]].append(c)
    return {bx: clinical(group) for bx, group in sorted(groups.items())}


def confidence(cases: list[dict]) -> dict:
    levels = {}
    for level in CONFIDENCE_LEVELS:
        group = [c for c in cases if c["confidence"] == level]
        levels[level] = {"n": len(group), "accuracy": _ratio(sum(c["correct"] for c in group), len(group))}
    return {
        "by_level": levels,
        "agreement_with_reference": _ratio(sum(c["confidence"] == c["ref_confidence"] for c in cases), len(cases)),
    }


def tool_usage(cases: list[dict]) -> dict:
    counts = Counter(tool for c in cases for tool in set(c["tools"]))
    with_ref = [c for c in cases if c["ref_tools"]]
    return {
        "tools_per_case": mean(len(set(c["tools"])) for c in cases),
        "reference_tools_per_case": mean(len(set(c["ref_tools"])) for c in cases),
        "share_of_cases": {tool: n / len(cases) for tool, n in counts.most_common()},
        "recall_of_reference": mean(len(set(c["tools"]) & set(c["ref_tools"])) / len(set(c["ref_tools"])) for c in with_ref),
    }
