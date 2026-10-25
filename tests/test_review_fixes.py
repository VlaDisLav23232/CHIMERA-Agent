import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import leaderboard, official, runs, score


class ReviewFixesTest(unittest.TestCase):
    def test_malformed_outputs_count_as_failures(self):
        valid = {"confidence": "clear", "variable_weights": {},
                 "reveal_sequence": [], "free_text": "Synthetic explanation"}
        malformed = [[], None, "text", 7]
        malformed += [{**valid, key: value} for key, value in (
            ("confidence", []), ("variable_weights", []),
            ("variable_weights", {"psa": []}), ("reveal_sequence", {}),
            ("reveal_sequence", [{}]), ("free_text", []))]
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "output/task1/synthetic"
            case_dir.mkdir(parents=True)
            (case_dir / runs.DECISION_FILE).write_text('"yes"')
            for payload in malformed:
                with self.subTest(payload=payload):
                    (case_dir / runs.REASONING_FILE).write_text(json.dumps(payload))
                    self.assertIsNone(runs.read_prediction(Path(tmp), "synthetic", {"clinical_data": {}}))

    def test_failed_judge_aborts(self):
        with patch.object(official._ev, "evaluate_case", return_value={
            "gate": "passed", "rationale_score": None
        }):
            with self.assertRaisesRegex(RuntimeError, "mixing scoring modes"):
                official.score_case({}, {}, judge=object())

    def test_judge_not_required_for_failed_decision(self):
        row = {"gate": "biopsy_decision_failed", "rationale_score": None}
        with patch.object(official._ev, "evaluate_case", return_value=row):
            self.assertEqual(official.score_case({}, {}, judge=object()), row)

    def test_repeat_groups_match_model_code_and_configuration(self):
        base = {"system": "agent", "temperature": 1.0, "model": "model-a",
                "code_commit": "abc", "config_id": "tools-a",
                "evaluator_commit": "123", "judge": "none",
                "metrics": {"task1_score": 0.6}}
        self.assertEqual(len(leaderboard.setting_rows([base, dict(base)])), 1)
        for key in ("model", "code_commit", "config_id", "evaluator_commit", "judge"):
            with self.subTest(key=key):
                self.assertEqual(leaderboard.setting_rows([base, {**base, key: "different"}]), [])
        self.assertEqual(leaderboard.setting_rows([{**base, "config_id": None}] * 2), [])

    def test_run_names_cannot_leave_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runs, "CASES_DIR", root / "runs"), patch.object(runs, "SUMMARIES_DIR", root / "summaries"):
                for name in ("../escape", "a/b", "a\b", "", ".hidden", "/abs"):
                    with self.subTest(name=name):
                        with self.assertRaises(ValueError):
                            runs.write_cases(name, [])
                        with self.assertRaises(ValueError):
                            runs.write_summary(name, {}, "")
                        with self.assertRaises(ValueError):
                            runs.read_cases(name)
                runs.write_cases("ok_name-1", [])
            self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob("*")),
                             ["runs", "runs/ok_name-1", "runs/ok_name-1/cases.json"])

    def test_summary_keeps_code_and_baseline_commits_apart(self):
        info = {"ollama_model": "m", "code_commit": "ours", "baseline_commit": "theirs", "config_id": "c"}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "output/task1").mkdir(parents=True)
            argv = ["score", tmp, "--name", "synthetic", "--system", "agent", "--temperature", "1.0"]
            with patch("sys.argv", argv), patch.object(score, "score_run", return_value=([], {})),                     patch.object(score, "summarize", return_value={}), patch.object(score, "to_markdown", return_value=""),                     patch.object(runs, "read_run_info", return_value=info), patch.object(runs, "write_cases"),                     patch.object(runs, "write_summary") as write_summary, patch("builtins.print"):
                score.main()
        summary = write_summary.call_args.args[1]
        self.assertEqual((summary["code_commit"], summary["baseline_commit"]), ("ours", "theirs"))
