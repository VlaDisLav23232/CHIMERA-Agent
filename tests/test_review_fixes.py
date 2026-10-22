import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import leaderboard, official, runs


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
