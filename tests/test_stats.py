import unittest

from evaluation import metrics, stats


def case(i, correct, ref="yes"):
    pred = ref if correct else ("no" if ref == "yes" else "yes")
    return {"case_id": str(i), "ref": ref, "pred": pred, "correct": correct, "case_score": 0.5 if correct else 0.0}


class StatsTest(unittest.TestCase):
    def test_mcnemar_identical_runs(self):
        cases = [case(i, i % 2 == 0) for i in range(10)]
        self.assertEqual(stats.mcnemar_exact(cases, cases), 1.0)

    def test_mcnemar_one_sided_disagreement(self):
        a = [case(i, True) for i in range(10)]
        b = [case(i, i >= 8) for i in range(10)]
        self.assertAlmostEqual(stats.mcnemar_exact(a, b), 2 / 2**8)

    def test_paired_bootstrap_self_is_zero(self):
        cases = [case(i, i % 3 != 0) for i in range(20)]
        d = stats.paired_bootstrap(cases, cases, metrics.task1_score)
        self.assertEqual(d["diff"], 0.0)
        self.assertEqual(d["ci"], (0.0, 0.0))

    def test_paired_bootstrap_needs_same_cases(self):
        with self.assertRaises(ValueError):
            stats.paired_bootstrap([case(1, True)], [case(2, True)], metrics.task1_score)

    def test_bootstrap_ci_contains_estimate(self):
        cases = [case(i, i % 4 != 0) for i in range(40)]
        low, high = stats.bootstrap_ci(cases, metrics.task1_score)
        self.assertLess(low, metrics.task1_score(cases))
        self.assertGreater(high, metrics.task1_score(cases))


if __name__ == "__main__":
    unittest.main()
