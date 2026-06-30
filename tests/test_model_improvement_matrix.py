import json
import os
import subprocess
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "model_improvement_matrix.py")


class ModelImprovementMatrixTestCase(unittest.TestCase):
    def test_markdown_ranks_kra_dual_phase_first_for_horse(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--sport", "horse"],
            check=True,
            capture_output=True,
            text=True,
        )

        lines = [line for line in result.stdout.splitlines() if line.startswith("| 1 |")]
        self.assertTrue(lines)
        self.assertIn("KRA-DUAL-PHASE-ODDS", lines[0])

    def test_json_contains_scores_and_keirin_filter(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--sport", "keirin", "--format", "json"],
            check=True,
            capture_output=True,
            text=True,
        )

        data = json.loads(result.stdout)
        self.assertTrue(data)
        self.assertTrue(all(item["sport"] in ("keirin", "both") for item in data))
        self.assertTrue(all(isinstance(item["score"], int) for item in data))


if __name__ == "__main__":
    unittest.main()
