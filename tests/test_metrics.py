import unittest

from evaluation import metrics


def case(ref, pred, bx="None", confidence="clear", ref_confidence="clear", tools=(), ref_tools=(), score=0.5):
    return {"case_id": f"{ref}{pred}{bx}{len(tools)}", "ref": ref, "pred": pred, "bx": bx,
            "correct": ref == pred, "case_score": score if ref == pred else 0.0,
            "confidence": confidence, "ref_confidence": ref_confidence,
            "tools": list(tools), "ref_tools": list(ref_tools)}


class ClinicalTest(unittest.TestCase):
    def test_confusion_and_rates(self):
        cases = [case("yes", "yes"), case("yes", "no"), case("no", "no"), case("no", "yes")]
        m = metrics.clinical(cases)
        self.assertEqual((m["tp"], m["fn"], m["tn"], m["fp"]), (1, 1, 1, 1))
        self.assertEqual(m["sensitivity"], 0.5)
        self.assertEqual(m["specificity"], 0.5)
        self.assertEqual(m["f1_yes"], 0.5)

    def test_invalid_output_is_an_error_both_ways(self):
        m = metrics.clinical([case("yes", None), case("no", None)])
        self.assertEqual((m["fn"], m["fp"], m["invalid"]), (1, 1, 2))
        self.assertEqual(m["accuracy"], 0.0)

    def test_invalid_output_is_not_a_false_positive_for_f1(self):
        self.assertEqual(metrics.f1_yes([case("yes", "yes"), case("no", None)]), 1.0)

    def test_by_prior_biopsy(self):
        groups = metrics.by_prior_biopsy([case("yes", "yes", bx="Positive"), case("no", "yes", bx="None")])
        self.assertEqual(set(groups), {"Positive", "None"})
        self.assertEqual(groups["Positive"]["accuracy"], 1.0)


class ConfidenceAndToolsTest(unittest.TestCase):
    def test_accuracy_per_confidence_level(self):
        c = metrics.confidence([case("yes", "yes", confidence="clear"), case("no", "yes", confidence="uncertain",
                                                                         ref_confidence="uncertain")])
        self.assertEqual(c["by_level"]["clear"], {"n": 1, "accuracy": 1.0})
        self.assertEqual(c["by_level"]["borderline"]["accuracy"], None)
        self.assertEqual(c["agreement_with_reference"], 1.0)

    def test_tool_usage(self):
        t = metrics.tool_usage([case("yes", "yes", tools=["a", "b"], ref_tools=["a", "c"])])
        self.assertEqual(t["tools_per_case"], 2)
        self.assertEqual(t["recall_of_reference"], 0.5)
        self.assertEqual(t["share_of_cases"], {"a": 1.0, "b": 1.0})


if __name__ == "__main__":
    unittest.main()
