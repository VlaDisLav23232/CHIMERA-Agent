import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from chimera_agent.config import BASELINE_COMMIT, ROOT, Settings, config_id, load_settings
from chimera_agent.inputs import select_cases, stage_case

# agent_v3_* are ablations of agent_v3, each without one of its parts (worker.variant_for).
ABLATIONS = ("agent_v3_model_decides", "agent_v3_model_rates", "agent_v3_agent_fetches")
SYSTEMS = ("baseline_scaffold", "agent_v1", "agent_v2", "agent_v3", *ABLATIONS)
# Systems that run our graph and write a trace per case.
TRACED = ("agent_v1", "agent_v2", "agent_v3", *ABLATIONS)
# Readiness fields that must agree across the cases of one run.
READINESS_IDENTITY = ("model_digest", "context_length", "embedding_revision")


class ReadinessMismatch(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "chimera_agent").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def terminate_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_worker(command: list[str], log: Path, seconds: float) -> str:
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
            return "succeeded" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            return "timed_out"
        finally:
            # Includes orphaned MCP/embedding children, even after the worker exits.
            terminate_group(process)


def publish(case_dir: Path, run_dir: Path, case_id: str, system: str) -> None:
    from chimera_agent_baseline.output.schema import Task1Output

    from chimera_agent.evidence import Trace
    case = case_dir / "work/output/task1" / case_id
    decision = json.loads((case / "prostate-biopsy-decision.json").read_text())
    reasoning = json.loads((case / "prostate-biopsy-decision-reasoning.json").read_text())
    if decision not in ("yes", "no"):
        raise ValueError("Invalid output decision")
    Task1Output.model_validate({
        "case_id": case_id, "task": 1, "biopsy_decision": decision == "yes",
        "confidence": reasoning["confidence"], "variable_weights": reasoning["variable_weights"],
        "reasoning": reasoning["free_text"], "reveal_sequence": reasoning["reveal_sequence"],
    })
    if system in TRACED:
        trace = Trace.model_validate_json((case_dir / "work/trace.json").read_text())
        if trace.case_id != case_id:
            raise ValueError("Trace belongs to another case")
        if trace.system != system:
            raise ValueError("Trace belongs to another system")
    # Everything is validated above; the official files move first so a trace is never published without them.
    (run_dir / "output/task1").mkdir(parents=True, exist_ok=True)
    case.rename(run_dir / "output/task1" / case_id)
    if system in TRACED:
        (run_dir / "trace").mkdir(exist_ok=True)
        (case_dir / "work/trace.json").rename(run_dir / "trace" / f"{case_id}.json")


def run_one(settings: Settings, run_dir: Path, system: str, case_id: str, reference: dict | None) -> dict:
    started = time.monotonic()
    case_dir = run_dir / "cases" / case_id
    record: dict = {}
    try:
        case_dir.mkdir(parents=True)
        stage_case(settings.data_dir, case_dir / "input", case_id)
        (case_dir / "settings.json").write_text(json.dumps(settings.payload(), indent=2))
        record["status"] = run_worker(
            [sys.executable, "-m", "chimera_agent.worker", str(case_dir), system, str(run_dir / "resources")],
            case_dir / "worker.log", settings.wall_seconds)
        if record["status"] == "succeeded":
            record["readiness"] = json.loads((case_dir / "readiness.json").read_text())
            if reference is not None and any(record["readiness"][k] != reference[k] for k in READINESS_IDENTITY):
                raise ReadinessMismatch("Model or embedding identity changed during the run")
            publish(case_dir, run_dir, case_id, system)
    except Exception as error:
        # One broken case must not stop the batch; its status and error type stay in the manifest.
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def overall_status(records: list[dict]) -> str:
    done = [r["status"] == "succeeded" for r in records]
    return "succeeded" if all(done) else "partial" if any(done) else "failed"


def run(settings: Settings, name: str, system: str, cases: list[str]) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Run name must contain only letters, numbers, underscores or hyphens")
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}")
    run_dir = ROOT / "results" / name
    run_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    identity = {"code_commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
                "source_digest": source_digest(), "baseline_commit": BASELINE_COMMIT, "system": system}
    manifest = {
        "status": "running", "ollama_model": settings.model, **identity,
        "settings": settings.payload(), "config_id": None, "cases": {c: {"status": "pending"} for c in cases},
    }
    def save():
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (run_dir / "run_info.json").write_text(json.dumps(manifest, indent=2))
    save()
    try:
        # Chroma writes runtime SQLite metadata even for retrieval-only queries; the pinned index stays untouched.
        shutil.copytree(settings.baseline_dir / "resources", run_dir / "resources")
        reference = None
        for case_id in cases:
            record = run_one(settings, run_dir, system, case_id, reference)
            manifest["cases"][case_id] = record
            if record["status"] == "succeeded" and reference is None:
                reference = record["readiness"]
                manifest["config_id"] = config_id(settings, {**identity, **reference})
            save()
        manifest["status"] = overall_status(list(manifest["cases"].values()))
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        save()
    if manifest["status"] != "succeeded":
        raise RuntimeError(f"Run {manifest['status']}; see {run_dir / 'run_info.json'} and cases/*/worker.log")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cases with the unchanged baseline or one of our agents")
    parser.add_argument("--name", required=True)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--case", action="append", default=[], help="dev case ID; repeat for several")
    parser.add_argument("--all-dev", action="store_true", help="run every case of the committed dev split")
    parser.add_argument("--all-test", action="store_true", help="run every case of the test split (final runs only)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent.json")
    args = parser.parse_args()
    try:
        cases = select_cases(ROOT / "splits/task1.json", args.case, args.all_dev, args.all_test)
        directory = run(load_settings(ROOT / args.config), args.name, args.system, cases)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Succeeded: {directory}")


if __name__ == "__main__":
    main()
