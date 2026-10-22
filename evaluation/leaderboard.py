import argparse
from collections import defaultdict
from statistics import mean, stdev

from evaluation import metrics, runs, stats
from evaluation.score import fmt


def pick_reference(summaries: list[dict], name: str | None) -> dict | None:
    if name:
        matches = [s for s in summaries if s["name"] == name]
        if not matches:
            raise SystemExit(f"reference run {name!r} not found for this split and judge mode")
        return matches[0]
    baselines = [s for s in summaries if s["system"] == "baseline"]
    return baselines[0] if len(baselines) == 1 else None


def run_rows(summaries: list[dict], reference: dict | None) -> list[str]:
    ref_cases = runs.read_cases(reference["name"]) if reference else None
    rows = []
    for s in sorted(summaries, key=lambda s: -s["metrics"]["task1_score"]):
        m, c = s["metrics"], s["metrics"]["clinical"]
        lo, hi = m["task1_score_ci"]
        vs = p = "—"
        if reference and s["name"] != reference["name"]:
            cases = runs.read_cases(s["name"])
            d = stats.paired_bootstrap(cases, ref_cases, metrics.task1_score)
            vs = f"{d['diff']:+.3f} ({d['ci'][0]:+.3f}…{d['ci'][1]:+.3f})"
            p = f"{stats.mcnemar_exact(cases, ref_cases):.3f}"
        rows.append(
            f"| {s['name']} | {s['system']} | {s['temperature']} | {fmt(m['task1_score'])} ({fmt(lo)}–{fmt(hi)}) "
            f"| {fmt(m['mean_case_score'])} | {fmt(c['f1_yes'])} | {fmt(c['sensitivity'])} | {fmt(c['specificity'])} "
            f"| {m['tools']['tools_per_case']:.2f} | {c['invalid']} | {vs} | {p} |"
        )
    return rows


def setting_rows(summaries: list[dict]) -> list[str]:
    groups = defaultdict(list)
    for s in summaries:
        identity = (s.get("model"), s.get("code_commit"), s.get("config_id"))
        if any(v in (None, "", "unknown") for v in identity):
            continue
        groups[(s["system"], s["temperature"], *identity, s["evaluator_commit"], s["judge"])].append(s["metrics"]["task1_score"])
    return [
        f"| {system} | {t} | {model} | {commit} | {config} | {len(v)} | {mean(v):.3f} | {stdev(v):.3f} |"
        for (system, t, model, commit, config, evaluator, judge), v in sorted(groups.items()) if len(v) > 1
    ]


def tool_rows(summaries: list[dict]) -> tuple[list[str], list[str]]:
    tools = sorted({t for s in summaries for t in s["metrics"]["tools"]["share_of_cases"]})
    header = ["| Run | " + " | ".join(tools) + " | Recall of reference |", "|---" * (len(tools) + 2) + "|"]
    rows = [
        f"| {s['name']} | " + " | ".join(f"{s['metrics']['tools']['share_of_cases'].get(t, 0):.2f}" for t in tools)
        + f" | {fmt(s['metrics']['tools']['recall_of_reference'])} |"
        for s in summaries
    ]
    return header, rows


def main() -> None:
    p = argparse.ArgumentParser(description="Rank all scored runs of a split.")
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--judge", action="store_true", help="rank runs scored with the LLM judge")
    p.add_argument("--reference", help="run to compare against (default: the only baseline run)")
    a = p.parse_args()

    summaries = [s for s in runs.read_summaries()
                 if s["split"] == a.split and (s["judge"] != "none") == a.judge and "(unofficial)" not in s["judge"]]
    if not summaries:
        raise SystemExit(f"no scored runs for split {a.split}")
    reference = pick_reference(summaries, a.reference)
    mode = "with LLM judge" if a.judge else "deterministic"
    tool_header, tools = tool_rows(summaries)
    settings = setting_rows(summaries)
    lines = [
        f"# Leaderboard: {a.split} split, {mode}", "",
        f"Reference for paired comparison: `{reference['name'] if reference else 'none'}`. "
        "Difference: paired bootstrap 95% CI of the Task 1 score; p: exact McNemar on decision correctness.", "",
        "| Run | System | Temp | Task 1 score (95% CI) | Mean case | F1(yes) | Sens | Spec | Tools/case | Invalid | vs reference | p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *run_rows(summaries, reference), "",
        "## Tool usage (share of cases)", "", *tool_header, *tools,
    ]
    if settings:
        lines += ["", "## Repeated runs", "", "| System | Temp | Model | Commit | Config | Runs | Mean Task 1 | SD |", "|---|---|---|---|---|---|---|---|", *settings]
    markdown = "\n".join(lines) + "\n"
    suffix = "_judge" if a.judge else ""
    (runs.SUMMARIES_DIR / f"leaderboard_{a.split}{suffix}.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
