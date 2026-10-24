import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "chimera_agent"


class IsolationTest(unittest.TestCase):
    def test_inference_never_imports_evaluation(self):
        # The evaluator alone reads reference answers; inference code must not be able to reach it.
        for path in sorted(PACKAGE.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    with self.subTest(file=path.name, module=name):
                        self.assertFalse(name == "evaluation" or name.startswith("evaluation."))
