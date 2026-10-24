import json
import unittest

from evaluation import audit
from evaluation.runs import SPLIT_FILE


class SelectTest(unittest.TestCase):
    def test_selection_is_deterministic_stratified_and_dev_only(self):
        dev = [f"c{i:02d}" for i in range(60)]
        # Same stratum sizes as the committed dev split: 12 Negative, 16 None, 32 Positive.
        bx = {c: "Negative" if i < 12 else "None" if i < 28 else "Positive" for i, c in enumerate(dev)}
        first = audit.select_cases(dev, bx)
        self.assertEqual(first, audit.select_cases(list(reversed(dev)), bx))
        self.assertTrue(set(first) <= set(dev))
        self.assertEqual([sum(bx[c] == k for c in first) for k in ("Negative", "None", "Positive")], [2, 3, 5])

    def test_rounding_that_misses_the_size_fails(self):
        dev = [f"c{i:02d}" for i in range(60)]
        with self.assertRaises(ValueError):
            audit.select_cases(dev, {c: ("a", "b", "c")[i % 3] for i, c in enumerate(dev)})

    def test_committed_audit_set_is_inside_dev_and_avoids_test(self):
        if not audit.AUDIT_FILE.exists():
            self.skipTest("audit set not selected yet")
        split = json.loads(SPLIT_FILE.read_text())
        cases = json.loads(audit.AUDIT_FILE.read_text())["cases"]
        self.assertEqual(len(cases), audit.N_AUDIT)
        self.assertTrue(set(cases) <= set(split["dev"]))
        self.assertFalse(set(cases) & set(split["test"]))


class SheetTest(unittest.TestCase):
    def test_sheet_hides_system_and_key_maps_back(self):
        texts = {("baseline", "c1"): "PSA is high. Biopsy advised.", ("agent_v1", "c1"): "PI-RADS 4 lesion."}
        rows, key = audit.build_sheet(texts)
        self.assertEqual(len(rows), 3)
        self.assertNotIn("system", rows[0])
        self.assertEqual(sorted((k["system"], r["claim"]) for r, k in zip(rows, key)),
                         [("agent_v1", "PI-RADS 4 lesion."), ("baseline", "Biopsy advised."), ("baseline", "PSA is high.")])
        self.assertEqual(rows, audit.build_sheet(dict(reversed(texts.items())))[0])


class SummaryTest(unittest.TestCase):
    key = [{"row_id": "0", "system": "baseline", "case_id": "c1"}, {"row_id": "1", "system": "baseline", "case_id": "c1"},
           {"row_id": "2", "system": "agent_v1", "case_id": "c1"}, {"row_id": "3", "system": "agent_v1", "case_id": "c1"}]

    def sheet(self, *labels):
        return [{"row_id": str(i), "label": label} for i, label in enumerate(labels)]

    def test_shares_agreement_and_kappa(self):
        sheets = {"a": self.sheet("unsupported", "supported", "supported", "supported"),
                  "b": self.sheet("unsupported", "supported", "unsupported", "supported")}
        result = audit.summarize(self.key, sheets, {"2": "supported"})
        self.assertEqual(result["systems"]["baseline"]["unsupported_share"], 0.5)
        self.assertEqual(result["systems"]["agent_v1"]["unsupported_share"], 0.0)
        self.assertEqual(result["systems"]["agent_v1"]["by_annotator"], {"a": 0.0, "b": 0.5})
        self.assertEqual((result["disagreements"], result["raw_agreement"]), (1, 0.75))
        self.assertAlmostEqual(result["cohen_kappa"], 0.5)

    def test_unresolved_or_missing_labels_fail(self):
        sheets = {"a": self.sheet("unsupported", "supported", "supported", "supported"),
                  "b": self.sheet("unsupported", "supported", "unsupported", "supported")}
        with self.assertRaisesRegex(ValueError, "Resolve"):
            audit.summarize(self.key, sheets, {})
        sheets["b"] = self.sheet("unsupported", "", "supported", "supported")
        with self.assertRaisesRegex(ValueError, "not labelled"):
            audit.summarize(self.key, sheets, {})


class ChecksTest(unittest.TestCase):
    def test_check_shares(self):
        def trace(case_id, n, missing=(), numbers=(), ungrounded=()):
            return {"case_id": case_id, "decision": {"claims": [{}] * n}, "decision_attempts": 1,
                    "checks": {"missing_citations": [{"claim": c} for c in missing],
                               "unsupported_numbers": [{"claim": c} for c in numbers],
                               "ungrounded_weights": list(ungrounded)}}
        result = audit.check_summary([trace("a", 2, missing=[0], numbers=[0, 1]), trace("b", 2, ungrounded=["fh"])])
        self.assertEqual((result["claims"], result["citations_present_share"], result["numbers_supported_share"],
                          result["cases_with_ungrounded_weights"]), (4, 0.75, 0.5, 1))
