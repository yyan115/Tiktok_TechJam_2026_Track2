from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Project" / "harness"))
import iterate


class ControllerSurfaceTests(unittest.TestCase):
    def test_conclusive_review_rerolls_are_bounded_per_stage(self):
        conclusive = [
            {
                "type": "preflight_review",
                "scope": "attempt",
                "iteration": 7,
                "review": {"verdict": "REJECT"},
            }
            for _ in range(iterate.MAX_CONCLUSIVE_REVIEWS_PER_STAGE)
        ]
        with self.assertRaisesRegex(iterate.ControllerError, "exhausted"):
            iterate._require_review_budget(
                conclusive, scope="attempt", iteration=7
            )
        transport_failures = [
            {"type": "preflight_review", "scope": "attempt", "iteration": 7,
             "review": None}
            for _ in range(20)
        ]
        iterate._require_review_budget(
            transport_failures, scope="attempt", iteration=7
        )

    def test_removed_ledger_and_override_flags_are_unrecognized(self):
        commands = [
            [sys.executable, str(iterate.__file__), "--ledger", "/tmp/x", "log"],
            [sys.executable, str(iterate.__file__), "final", "--force"],
            [
                sys.executable, str(iterate.__file__), "run",
                "--solution", "x.py", "--card", "x.json",
                "--continue-past-convergence",
            ],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertTrue(
                "unrecognized arguments" in result.stderr
                or "invalid choice" in result.stderr
            )

    def test_source_policy_blocks_dynamic_escape_and_network(self):
        findings = iterate.scan_candidate(
            "import socket\nHYPOTHESIS='x'\ndef run(s): return eval('1')\n"
        )
        self.assertTrue(any("import is outside" in finding for finding in findings))
        self.assertTrue(any("dynamic-execution" in finding for finding in findings))

    def test_source_policy_allows_normal_numeric_candidate(self):
        findings = iterate.scan_candidate(
            "import numpy as np\nHYPOTHESIS='numeric'\ndef run(s):\n"
            "    return {'valid': np.zeros(len(s['valid'])), 'test': np.zeros(len(s['test']))}\n"
        )
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
