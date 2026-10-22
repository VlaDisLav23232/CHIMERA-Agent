import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from chimera_agent.config import BASELINE_COMMIT, ROOT, Settings, config_id, load_settings
from chimera_agent.inputs import select_case, stage_case


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


def publish(run_dir: Path, case_id: str) -> None:
    from chimera_agent_baseline.output.schema import Task1Output
    case = run_dir / "work/output/task1" / case_id
    decision = json.loads((case / "prostate-biopsy-decision.json").read_text())
    reasoning = json.loads((case / "prostate-biopsy-decision-reasoning.json").read_text())
    if decision not in ("yes", "no"):
        raise ValueError("Invalid output decision")
    Task1Output.model_validate({
        "case_id": case_id, "task": 1, "biopsy_decision": decision == "yes",
        "confidence": reasoning["confidence"], "variable_weights": reasoning["variable_weights"],
        "reasoning": reasoning["free_text"], "reveal_sequence": reasoning["reveal_sequence"],
    })
    (run_dir / "work/output").rename(run_dir / "output")


def run(settings: Settings, name: str, requested: str | None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Run name must contain only letters, numbers, underscores or hyphens")
    case_id = select_case(ROOT / "splits/task1.json", requested)
    run_dir = ROOT / "results" / name
    run_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    identity = {"code_commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")),
                "source_digest": source_digest(), "baseline_commit": BASELINE_COMMIT}
    manifest = {
        "system": "baseline_scaffold", "status": "running", "case_id": case_id,
        "ollama_model": settings.model, **identity,
        "settings": settings.payload(), "config_id": None,
    }
    def save():
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (run_dir / "run_info.json").write_text(json.dumps(manifest, indent=2))
    save()
    try:
        stage_case(settings.data_dir, run_dir / "input", case_id)
        (run_dir / "settings.json").write_text(json.dumps(settings.payload(), indent=2))
        manifest["status"] = run_worker(
            [sys.executable, "-m", "chimera_agent.worker", str(run_dir)],
            run_dir / "worker.log", max(0.01, settings.wall_seconds - (time.monotonic() - started)))
        if manifest["status"] == "succeeded":
            manifest["readiness"] = json.loads((run_dir / "readiness.json").read_text())
            manifest["config_id"] = config_id(settings, {**identity, **manifest["readiness"]})
            publish(run_dir, case_id)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        save()
    if manifest["status"] != "succeeded":
        raise RuntimeError(f"Run {manifest['status']}; see {run_dir / 'worker.log'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dev case with the unchanged baseline")
    parser.add_argument("--name", required=True)
    parser.add_argument("--case")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent.json")
    args = parser.parse_args()
    try:
        directory = run(load_settings(ROOT / args.config), args.name, args.case)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Succeeded: {directory}")


if __name__ == "__main__":
    main()
