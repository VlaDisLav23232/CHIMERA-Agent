import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chimera_agent.config import ROOT, load_settings
from chimera_agent.run import publish, run, run_worker


class RunTest(unittest.TestCase):
    def test_worker_exit_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "worker.log"
            self.assertEqual(run_worker([sys.executable, "-c", "pass"], log, 5), "succeeded")
            self.assertEqual(run_worker([sys.executable, "-c", "raise SystemExit(2)"], log, 5), "failed")
            self.assertEqual(run_worker([sys.executable, "-c", "import time; time.sleep(30)"], log, .1), "timed_out")

    def test_failure_records_status_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "results").mkdir()
            settings = load_settings(ROOT / "configs/agent.json")
            for status in ("failed", "timed_out"):
                with patch("chimera_agent.run.ROOT", root), patch("chimera_agent.run.git", return_value="abc"), \
                     patch("chimera_agent.run.select_case", return_value="dev"), \
                     patch("chimera_agent.run.stage_case"), patch("chimera_agent.run.run_worker", return_value=status):
                    with self.assertRaises(RuntimeError):
                        run(settings, status, None)
                directory = root / "results" / status
                self.assertEqual(json.loads((directory / "run_info.json").read_text())["status"], status)
                self.assertFalse((directory / "output").exists())

    def test_publish_validates_before_exposing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case = directory / "work/output/task1/dev"
            case.mkdir(parents=True)
            (case / "prostate-biopsy-decision.json").write_text('"yes"')
            (case / "prostate-biopsy-decision-reasoning.json").write_text('{}')
            with self.assertRaises((KeyError, ValueError)):
                publish(directory, "dev")
            self.assertFalse((directory / "output").exists())

    def test_success_publishes_valid_output(self):
        from chimera_agent_baseline.output.schema import VARIABLES_BY_TASK
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case = directory / "work/output/task1/dev"
            case.mkdir(parents=True)
            (case / "prostate-biopsy-decision.json").write_text('"yes"')
            (case / "prostate-biopsy-decision-reasoning.json").write_text(json.dumps({
                "confidence": "uncertain",
                "variable_weights": {key: "not_used" for key in VARIABLES_BY_TASK[1]},
                "reveal_sequence": [],
                "free_text": "Synthetic explanation for testing the output publication boundary."
            }))
            publish(directory, "dev")
            self.assertTrue((directory / "output/task1/dev/prostate-biopsy-decision.json").exists())
            self.assertFalse((directory / "work/output").exists())

    def test_invalid_run_name_rejected(self):
        settings = load_settings(ROOT / "configs/agent.json")
        with self.assertRaises(ValueError):
            run(settings, "../escape", None)
