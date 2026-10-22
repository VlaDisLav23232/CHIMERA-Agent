import argparse
from datetime import date
from pathlib import Path
from statistics import mean

from evaluation import metrics, official, runs, stats

COMPONENTS = ("confidence_score", "variable_weight_score", "important_decisive_factor_score",
              "tool_score", "section_grounding_score", "rationale_score")


def to_case(case_id: str, reference: dict, prediction: dict | None, row: dict) -> dict:
    prediction = prediction or {}
    return {
        "case_id": case_id,
        "bx": runs.prior_biopsy(case_id),
        "ref": reference["biopsy_decision"],
        "pred": row["pred_biopsy_decision"],
        "gate": row["gate"],
        "correct": row["decision_correct"],
        "case_score": row["case_score"],
        **{k: row[k] for k in COMPONENTS},
        "confidence": prediction.get("confidence"),
        "ref_confidence": reference.get("confidence"),
        "tools": prediction.get("reveal_sequence") or [],
        "ref_tools": reference.get("reveal_sequence") or [],
    }


def score_run(run_dir: Path, split: str, judge) -> tuple[list[dict], dict]:
    references = official.load_references()
    cases, rows = [], []
    for case_id in runs.split_cases(split):
        prediction = runs.read_prediction(run_dir, case_id, references[case_id])
        row = official.score_case(references[case_id], prediction, judge)
        rows.append(row)
        cases.append(to_case(case_id, references[case_id], prediction, row))
    return cases, official.aggregate(rows)


def summarize(cases: list[dict], aggregate: dict) -> dict:
    return {
        "task1_score": aggregate["ranking_score"],
        "task1_score_ci": stats.bootstrap_ci(cases, metrics.task1_score),
        "mean_case_score": aggregate["mean_case_score"],
        "f1_yes_ci": stats.bootstrap_ci(cases, metrics.f1_yes),
        "components": {k: mean(c[k] for c in cases if c[k] is not None) if any(c[k] is not None for c in cases) else None
                       for k in COMPONENTS},
        "clinical": metrics.clinical(cases),
        "by_prior_biopsy": metrics.by_prior_biopsy(cases),
        "confidence": metrics.confidence(cases),
        "tools": metrics.tool_usage(cases),
    }


def fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def to_markdown(s: dict) -> str:
    m, c = s["metrics"], s["metrics"]["clinical"]
    lo, hi = m["task1_score_ci"]
    lines = [
        f"# Run `{s['name']}`", "",
        f"System `{s['system']}`, model `{s['model']}`, temperature {s['temperature']}, seed {s['seed']}, "
        f"split `{s['split']}` ({c['n']} cases), judge `{s['judge']}`, evaluator `{s['evaluator_commit']}`, {s['date']}.", "",
        "| Task 1 score (95% CI) | Mean case score | F1(yes) | Accuracy | Sensitivity | Specificity | PPV | NPV | Invalid |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {fmt(m['task1_score'])} ({fmt(lo)}–{fmt(hi)}) | {fmt(m['mean_case_score'])} | {fmt(c['f1_yes'])} | {fmt(c['accuracy'])} "
        f"| {fmt(c['sensitivity'])} | {fmt(c['specificity'])} | {fmt(c['ppv'])} | {fmt(c['npv'])} | {c['invalid']} |", "",
        f"Confusion: TP {c['tp']}, FP {c['fp']}, TN {c['tn']}, FN {c['fn']}.", "",
        "Case score components (correct decisions only): " +", ".join(f"{k.removesuffix('_score')} {fmt(v)}" for k, v in m["components"].items()) + ".", "",
        "| Prior biopsy | n | Accuracy | F1(yes) | Sensitivity | Specificity |", "|---|---|---|---|---|---|",
        *[f"| {bx} | {g['n']} | {fmt(g['accuracy'])} | {fmt(g['f1_yes'])} | {fmt(g['sensitivity'])} | {fmt(g['specificity'])} |"
          for bx, g in m["by_prior_biopsy"].items()], "",
        "| Confidence | n | Accuracy |", "|---|---|---|",
        *[f"| {level} | {v['n']} | {fmt(v['accuracy'])} |" for level, v in m["confidence"]["by_level"].items()], "",
        f"Confidence equal to the reference: {fmt(m['confidence']['agreement_with_reference'])}.", "",
        f"Tools per case {m['tools']['tools_per_case']:.2f} (reference {m['tools']['reference_tools_per_case']:.2f}), "
        f"recall of the reference's tools {fmt(m['tools']['recall_of_reference'])}. Share of cases: "
        + ", ".join(f"{t} {v:.2f}" for t, v in m["tools"]["share_of_cases"].items()) + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Score a Task 1 run folder with the official evaluator.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--confirm-test", action="store_true")
    a = p.parse_args()

    if a.split in ("test", "all") and not a.confirm_test:
        p.error(f"--split {a.split} includes test cases: add --confirm-test")
    if not (a.run_dir / "output" / "task1").is_dir():
        p.error(f"{a.run_dir}/output/task1 not found")

    judge = official.make_judge() if a.judge else None
    judge_label = "none"
    if a.judge:
        judge_label = official.JUDGE_MODEL + ("" if official.JUDGE_MODEL == official.OFFICIAL_JUDGE_MODEL else " (unofficial)")

    cases, aggregate = score_run(a.run_dir, a.split, judge)
    info = runs.read_run_info(a.run_dir)
    summary = {
        "name": a.name, "system": a.system, "model": info.get("ollama_model", "unknown"),
        "code_commit": info.get("baseline_commit") or info.get("code_commit"),
        "config_id": info.get("config_id"),
        "temperature": a.temperature, "seed": a.seed, "split": a.split, "judge": judge_label,
        "evaluator_commit": official.COMMIT, "date": date.today().isoformat(), "run_dir": str(a.run_dir),
        "metrics": summarize(cases, aggregate),
    }
    runs.write_cases(a.name, cases)
    markdown = to_markdown(summary)
    runs.write_summary(a.name, summary, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
