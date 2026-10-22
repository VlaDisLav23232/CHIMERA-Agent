import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_RUN = ROOT / "results" / "baseline_run"
HAS_DATA = (ROOT / "data" / "task1").is_dir() and BASELINE_RUN.is_dir()


@unittest.skipUnless(HAS_DATA, "needs data/task1 and results/baseline_run (local only)")
class BaselineTest(unittest.TestCase):
    def test_reproduces_published_baseline(self):
        from evaluation import metrics
        from evaluation.score import score_run
        cases, aggregate = score_run(BASELINE_RUN, "all", None)
        self.assertEqual(len(cases), 91)
        self.assertAlmostEqual(aggregate["decision_accuracy"], 0.692, places=3)
        self.assertAlmostEqual(aggregate["decision_f1_yes"], 0.800, places=3)
        self.assertAlmostEqual(aggregate["ranking_score"], 0.657, places=3)
        self.assertAlmostEqual(metrics.task1_score(cases), aggregate["ranking_score"], places=9)


if __name__ == "__main__":
    unittest.main()
