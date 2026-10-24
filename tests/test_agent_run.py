import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chimera_agent.config import ROOT, load_settings
from chimera_agent.run import publish, run, run_worker

READINESS = {"model_digest": "d", "context_length": 16384, "embedding_revision": "r", "retrieval_hits": 5}


def write_outputs(case_dir: Path, case_id: str, reasoning: dict | None = None, trace: bool = False) -> None:
    from chimera_agent_baseline.output.schema import VARIABLES_BY_TASK
    case = case_dir / "work/output/task1" / case_id
    case.mkdir(parents=True)
    (case / "prostate-biopsy-decision.json").write_text('"yes"')
    if reasoning is None:
        reasoning = {
            "confidence": "uncertain",
            "variable_weights": {key: "not_used" for key in VARIABLES_BY_TASK[1]},
            "reveal_sequence": [],
            "free_text": "Synthetic explanation for testing the output publication boundary.",
        }
    (case / "prostate-biopsy-decision-reasoning.json").write_text(json.dumps(reasoning))
    if trace:
        from chimera_agent.evidence import Claim, DecisionRecord, Timings, Trace, build_ledger, run_checks
        from tests.test_agent_evidence import PROMPT, synthetic_messages
        items, calls = build_ledger(PROMPT, synthetic_messages())
        decision = DecisionRecord(biopsy_decision=True, confidence="clear", variable_weights={},
                                  claims=[Claim(text="PI-RADS 4.", evidence=["P"])])
        (case_dir / "work/trace.json").write_text(Trace(
            case_id=case_id, evidence=items, tool_calls=calls, agent_message="", decision=decision,
            checks=run_checks(decision, items, calls), decision_attempts=1, llm_calls=2,
            timings=Timings(agent_s=1, decision_s=1, total_s=2)).model_dump_json(by_alias=True))


class RunTest(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT / "configs/agent.json")

    def test_worker_exit_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "worker.log"
            self.assertEqual(run_worker([sys.executable, "-c", "pass"], log, 5), "succeeded")
            self.assertEqual(run_worker([sys.executable, "-c", "raise SystemExit(2)"], log, 5), "failed")
            self.assertEqual(run_worker([sys.executable, "-c", "import time; time.sleep(30)"], log, .1), "timed_out")

    def batch(self, root: Path, name: str, cases: list[str], worker, system="agent_v1"):
        (root / "results").mkdir(exist_ok=True)
        with patch("chimera_agent.run.ROOT", root), patch("chimera_agent.run.git", return_value="abc"), \
             patch("chimera_agent.run.shutil.copytree"), patch("chimera_agent.run.stage_case"), \
             patch("chimera_agent.run.run_worker", side_effect=worker):
            with self.assertRaises(RuntimeError):
                run(self.settings, name, system, cases)
        directory = root / "results" / name
        return directory, json.loads((directory / "run_info.json").read_text())

    def test_batch_continues_after_failures_and_publishes_only_good_cases(self):
        def worker(command, log, seconds):
            case_dir = Path(command[3])
            case_dir.mkdir(parents=True, exist_ok=True)
            if case_dir.name in ("bad", "late"):
                return {"bad": "failed", "late": "timed_out"}[case_dir.name]
            (case_dir / "readiness.json").write_text(json.dumps(READINESS))
            write_outputs(case_dir, case_dir.name, trace=True)
            return "succeeded"
        with tempfile.TemporaryDirectory() as temp:
            directory, info = self.batch(Path(temp), "mixed", ["good", "bad", "late"], worker)
            self.assertEqual(info["status"], "partial")
            self.assertEqual({c: r["status"] for c, r in info["cases"].items()},
                             {"good": "succeeded", "bad": "failed", "late": "timed_out"})
            self.assertEqual((info["system"], info["code_commit"]), ("agent_v1", "abc"))
            self.assertIsNotNone(info["baseline_commit"])
            self.assertIsNotNone(info["config_id"])
            self.assertEqual(sorted(p.name for p in (directory / "output/task1").iterdir()), ["good"])
            self.assertEqual(sorted(p.name for p in (directory / "trace").iterdir()), ["good.json"])

    def test_invalid_output_or_identity_change_fails_only_that_case(self):
        def worker(command, log, seconds):
            case_dir = Path(command[3])
            case_dir.mkdir(parents=True, exist_ok=True)
            readiness = READINESS | ({"model_digest": "other"} if case_dir.name == "swapped" else {})
            (case_dir / "readiness.json").write_text(json.dumps(readiness))
            write_outputs(case_dir, case_dir.name, reasoning={} if case_dir.name == "broken" else None, trace=True)
            return "succeeded"
        with tempfile.TemporaryDirectory() as temp:
            directory, info = self.batch(Path(temp), "checks", ["good", "broken", "swapped"], worker)
            self.assertEqual({c: (r["status"], r.get("error_type")) for c, r in info["cases"].items()},
                             {"good": ("succeeded", None), "broken": ("failed", "KeyError"),
                              "swapped": ("failed", "ReadinessMismatch")})
            self.assertEqual(sorted(p.name for p in (directory / "output/task1").iterdir()), ["good"])

    def test_all_failed_is_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, info = self.batch(Path(temp), "none", ["a"], lambda *args: "failed", system="baseline_scaffold")
            self.assertEqual((info["status"], info["system"]), ("failed", "baseline_scaffold"))

    def test_publish_validates_before_exposing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_outputs(directory / "case", "dev", reasoning={})
            with self.assertRaises((KeyError, ValueError)):
                publish(directory / "case", directory, "dev", "baseline_scaffold")
            self.assertFalse((directory / "output").exists())

    def test_publish_rejects_trace_of_another_case(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_outputs(directory / "case", "dev", trace=True)
            trace = json.loads((directory / "case/work/trace.json").read_text())
            (directory / "case/work/trace.json").write_text(json.dumps(trace | {"case_id": "other"}))
            with self.assertRaisesRegex(ValueError, "another case"):
                publish(directory / "case", directory, "dev", "agent_v1")
            self.assertFalse((directory / "output").exists())

    def test_invalid_run_name_and_system_rejected(self):
        with self.assertRaises(ValueError):
            run(self.settings, "../escape", "agent_v1", ["a"])
        with self.assertRaises(ValueError):
            run(self.settings, "fine", "other", ["a"])
