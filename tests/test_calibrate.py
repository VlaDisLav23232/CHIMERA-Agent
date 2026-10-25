import unittest
from dataclasses import replace

from evaluation.calibrate import VARIABLES, fit, upper_median


def reference(decision, confidence, **weights):
    return {"biopsy_decision": decision, "confidence": confidence, "variable_weights": weights}


class CalibrateTest(unittest.TestCase):
    def test_upper_median(self):
        self.assertEqual(upper_median(["noted", "decisive", "not_used"]), "noted")
        self.assertEqual(upper_median(["noted", "important"]), "important")

    def test_fit_per_decision_with_missing_weights_as_not_used(self):
        table = fit([reference("yes", "clear", pirads="decisive"), reference("yes", "borderline", pirads="decisive"),
                     reference("yes", "clear", pirads="important", bx="noted"),
                     reference("no", "uncertain", bx="decisive")])
        self.assertEqual(table["cases"], {"yes": 3, "no": 1})
        self.assertEqual(table["confidence"], {"yes": "clear", "no": "uncertain"})
        self.assertEqual(set(table["variable_weights"]["yes"]), set(VARIABLES))
        self.assertEqual((table["variable_weights"]["yes"]["pirads"], table["variable_weights"]["yes"]["bx"]),
                         ("decisive", "not_used"))
        self.assertEqual(table["variable_weights"]["no"]["bx"], "decisive")

    def test_fit_needs_both_decisions(self):
        with self.assertRaises(ValueError):
            fit([reference("yes", "clear")])


class VariantTest(unittest.TestCase):
    def test_agent_v3_variant_uses_the_committed_dev_calibration(self):
        from chimera_agent.worker import FETCHED, variant_for
        v3 = variant_for("agent_v3", {"pirads": "3", "psad": 0.25})
        self.assertEqual((v3.gate.biopsy_decision, v3.mandatory_tools, v3.agent_tools, v3.grounded_only),
                         (True, FETCHED, ("search_guidelines",), True))
        self.assertEqual(v3.calibration.fitted_on, "dev")
        self.assertIsNone(variant_for("agent_v3", {"pirads": "NA", "psad": 0.25}).gate)
        self.assertIsNone(variant_for("agent_v2", {"pirads": "5", "psad": 0.25}).gate)

    def test_each_ablation_removes_one_part(self):
        from chimera_agent.graph import PLAIN
        from chimera_agent.worker import variant_for
        card = {"pirads": "5", "psad": 0.3}
        full = variant_for("agent_v3", card)
        self.assertEqual(variant_for("agent_v3_model_decides", card), replace(full, gate=None))
        self.assertEqual(variant_for("agent_v3_model_rates", card), replace(full, calibration=None))
        self.assertEqual(variant_for("agent_v3_agent_fetches", card),
                         replace(full, mandatory_tools=(), agent_tools=None))
        self.assertEqual(variant_for("agent_v1", card), PLAIN)
