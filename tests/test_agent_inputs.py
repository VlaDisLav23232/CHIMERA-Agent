import json
import tempfile
import unittest
from pathlib import Path

from chimera_agent.inputs import CLINICAL, PROMPT, select_cases, stage_case


class InputTest(unittest.TestCase):
    def test_only_inputs_are_copied_and_missing_embeddings_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "raw/dev"
            case.mkdir(parents=True)
            (case / PROMPT).write_text(json.dumps({"case_id": "dev", "task": 1}))
            (case / CLINICAL).write_text(json.dumps({"case_id": "dev"}))
            (case / "prostate-biopsy-decision.json").write_text('"LABEL_SENTINEL"')
            (case / "prostate-biopsy-decision-reasoning.json").write_text('"LABEL_SENTINEL"')
            dest = stage_case(root / "raw", root / "input", "dev")
            self.assertEqual({p.name for p in dest.iterdir()}, {PROMPT, CLINICAL})
            self.assertNotIn("LABEL_SENTINEL", "".join(p.read_text() for p in dest.iterdir()))

    def test_symlink_and_wrong_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "raw/dev"
            case.mkdir(parents=True)
            (case / "label").write_text('"LABEL_SENTINEL"')
            (case / PROMPT).symlink_to(case / "label")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                stage_case(root / "raw", root / "input", "dev")
            (case / PROMPT).unlink()
            (case / PROMPT).write_text('{"case_id":"wrong", "task":1}')
            with self.assertRaisesRegex(ValueError, "mismatched"):
                stage_case(root / "raw", root / "input", "dev")

    def test_test_case_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            split = Path(temp) / "split.json"
            split.write_text('{"dev":["b","a"],"test":["c"]}')
            self.assertEqual(select_cases(split, [], False, False), ["a"])
            self.assertEqual(select_cases(split, [], True, False), ["a", "b"])
            self.assertEqual(select_cases(split, ["b", "a"], False, False), ["b", "a"])
            for requested, all_dev in ((["c"], False), (["a", "c"], False), (["a", "a"], False), (["a"], True)):
                with self.subTest(requested=requested, all_dev=all_dev), self.assertRaises(ValueError):
                    select_cases(split, requested, all_dev, False)

    def test_test_split_only_on_explicit_request(self):
        with tempfile.TemporaryDirectory() as temp:
            split = Path(temp) / "split.json"
            split.write_text('{"dev":["b","a"],"test":["d","c"]}')
            self.assertEqual(select_cases(split, [], False, True), ["c", "d"])
            for requested, all_dev in ((["a"], False), ([], True)):
                with self.subTest(requested=requested, all_dev=all_dev), self.assertRaises(ValueError):
                    select_cases(split, requested, all_dev, True)
