from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy


def start(epoch: float = 0.0) -> dict:
    return {
        "type": "run_start",
        "entry_id": "start",
        "run_id": "run",
        "started_epoch": epoch,
        "deadline_epoch": epoch + policy.WALL_CEILING_S,
    }


def attempt(index: int, score: float | None, *, error=None, eligible=True) -> dict:
    return {
        "type": "iteration",
        "entry_id": f"i{index}",
        "run_id": "run",
        "iteration": index,
        "started_epoch": float(index),
        "completed_epoch": float(index),
        "error": error,
        "eligible_for_final": eligible,
        "valid_metrics": None if score is None else {"primary": score},
        "solution": {"sha256": f"{index:064x}"},
        "sealed_test_scores": (
            {"path": f"Project/results/sealed/i{index}.npy", "sha256": f"{index + 100:064x}"}
            if eligible else None
        ),
    }


def basis(claim_id: str = "metric.within_user.c01", target: str = "mechanism") -> list[dict]:
    return [{"claim_id": claim_id, "relationship": "supports", "target": target}]


def portfolio() -> dict:
    families = []
    for index in range(1, 5):
        families.append({
            "family_id": f"family_{index}",
            "mechanism": "mechanism " + "m" * 70,
            "causal_claim": "causal claim " + "c" * 60,
            "smallest_experiment": "smallest experiment " + "e" * 50,
            "falsifier": "falsifier " + "f" * 40,
            "known_risks": "known risk " + "r" * 30,
            "expected_primary_delta": {"min": -0.01, "max": 0.01},
            "bank_topics": ["metric.within_user"],
            "research_basis": basis(f"metric.within_user.c0{index}"),
        })
    return {
        "schema_version": 2,
        "benchmark": policy.BENCHMARK,
        "selection_rubric": "selection rubric " + "s" * 120,
        "opening_order": [f"family_{index}" for index in range(1, 5)],
        "families": families,
    }


def card(family_id: str = "family_1", *, extension=None, card_type="explore") -> dict:
    return {
        "schema_version": 2,
        "benchmark": policy.BENCHMARK,
        "run_id": "run-id-for-policy-test",
        "iteration": 1,
        "family_id": family_id,
        "card_type": card_type,
        "candidate_path": "Project/solutions/s001.py",
        "candidate_sha256": "a" * 64,
        "mechanism": "mechanism " + "m" * 60,
        "hypothesis": "hypothesis " + "h" * 60,
        "change_summary": "change " + "c" * 50,
        "falsifier": "falsifier " + "f" * 40,
        "why_now": "why now " + "w" * 50,
        "expected_primary_delta": {"min": -0.01, "max": 0.01},
        "research_basis": basis(target="hypothesis"),
        "family_extension": extension,
        "prior_outcomes_considered": [],
    }


class TerminalPolicyTests(unittest.TestCase):
    def test_earliest_convergence_cannot_be_erased_by_later_gain(self):
        rows = [start()] + [attempt(i, 0.6) for i in range(1, 5)]
        self.assertEqual(policy.triggered_reasons(rows, 10), ["convergence"])
        self.assertEqual(policy.terminal_snapshot(rows, 10)["terminal_iteration"], 4)
        rows.append(attempt(5, 0.7))
        self.assertEqual(policy.triggered_reasons(rows, 11), ["convergence"])
        self.assertEqual(policy.terminal_snapshot(rows, 11)["terminal_iteration"], 4)

    def test_terminal_after_illegal_later_attempt_is_rejected(self):
        rows = [start()] + [attempt(i, 0.6) for i in range(1, 5)]
        terminal = {"type": "run_terminated", "entry_id": "terminal", **policy.terminal_snapshot(rows, 10)}
        legal = rows + [terminal]
        policy.validate_terminal_snapshot(legal, terminal)

        illegal_prefix = rows + [attempt(5, 0.7)]
        late = {"type": "run_terminated", "entry_id": "late", **policy.terminal_snapshot(illegal_prefix, 11)}
        with self.assertRaisesRegex(policy.PolicyError, "past the first terminal"):
            policy.validate_terminal_snapshot(illegal_prefix + [late], late)

    def test_failed_attempts_are_consecutive_non_improvements(self):
        rows = [attempt(1, 0.6)]
        rows += [attempt(2, 0.99, error="failed", eligible=False)]
        rows += [attempt(3, None, error="failed", eligible=False)]
        rows += [attempt(4, None, error="failed", eligible=False)]
        self.assertTrue(policy.convergence_reached(rows))

    def test_final_eligibility_is_explicit_and_ties_choose_earliest(self):
        absent = attempt(1, 0.7)
        absent.pop("eligible_for_final")
        self.assertFalse(policy.final_eligible(absent))
        candidates = [attempt(1, 0.7), attempt(2, 0.7)]
        self.assertEqual(policy.best_eligible(candidates)["entry_id"], "i1")

    def test_full_official_prefix_is_bound(self):
        rows = [start()] + [attempt(i, 0.6) for i in range(1, 5)]
        terminal = {"type": "run_terminated", "entry_id": "terminal", **policy.terminal_snapshot(rows, 10)}
        tampered = copy.deepcopy(rows + [terminal])
        tampered[2]["card"] = {"hypothesis": "changed"}
        with self.assertRaisesRegex(policy.PolicyError, "official_prefix_sha256"):
            policy.validate_terminal_snapshot(tampered, tampered[-1])

    def test_wall_and_cap(self):
        self.assertEqual(policy.triggered_reasons([start()], policy.WALL_CEILING_S), ["wall_clock_ceiling"])
        scores = [attempt(i, 0.6 + i * 0.003) for i in range(1, 51)]
        boundary, reasons = policy.earliest_attempt_terminal(scores)
        self.assertEqual(boundary, 50)
        self.assertEqual(reasons, ["iteration_cap"])


class ArtifactSchemaTests(unittest.TestCase):
    def test_expected_delta_interval_is_domain_bounded_and_informative(self):
        for interval in (
            {"min": -1.0, "max": 0.01},
            {"min": -0.20, "max": 0.20},
            {"min": 0.31, "max": 0.32},
        ):
            value = portfolio()
            value["families"][0]["expected_primary_delta"] = interval
            with self.assertRaisesRegex(policy.PolicyError, "width|within"):
                policy.validate_portfolio(value)

    def test_boolean_schema_version_is_rejected(self):
        with self.assertRaises(policy.PolicyError):
            policy.validate_portfolio({"schema_version": True})

    def test_unhashable_opening_order_is_policy_error(self):
        portfolio = {
            "schema_version": 2,
            "benchmark": policy.BENCHMARK,
            "families": [],
            "opening_order": [["bad"]],
            "selection_rubric": "x" * 100,
        }
        with self.assertRaises(policy.PolicyError):
            policy.validate_portfolio(portfolio)

    def test_old_freeform_evidence_schema_is_rejected(self):
        value = portfolio()
        value["families"][0]["evidence"] = [
            {"source": "somewhere", "locator": "page 1", "claim": "x" * 30}
        ]
        with self.assertRaisesRegex(policy.PolicyError, "missing or extra"):
            policy.validate_portfolio(value)

    def test_seed_family_forbids_extension(self):
        value = card(extension={})
        with self.assertRaisesRegex(policy.PolicyError, "forbids"):
            policy.validate_attempt_card(
                value, run_id=value["run_id"], expected_iteration=1,
                portfolio=portfolio(), candidate_path=value["candidate_path"],
                candidate_sha256=value["candidate_sha256"],
            )

    def test_new_family_requires_explore_and_exact_registration(self):
        extension = {
            "nearest_family_ids": ["family_1"],
            "bank_topics": ["metric.within_user"],
            "novel_topics": ["model.session_graph"],
            "mechanism_delta": "material mechanism delta " + "d" * 100,
        }
        value = card("session_graph", extension=extension)
        value["iteration"] = 5
        policy.validate_attempt_card(
            value, run_id=value["run_id"], expected_iteration=5,
            portfolio=portfolio(), candidate_path=value["candidate_path"],
            candidate_sha256=value["candidate_sha256"],
        )
        changed = copy.deepcopy(value)
        changed["family_extension"]["mechanism_delta"] += " changed"
        with self.assertRaisesRegex(policy.PolicyError, "does not match"):
            policy.validate_attempt_card(
                changed, run_id=changed["run_id"], expected_iteration=5,
                portfolio=portfolio(), candidate_path=changed["candidate_path"],
                candidate_sha256=changed["candidate_sha256"],
                registered_families={"session_graph": extension},
            )
        refine = copy.deepcopy(value)
        refine["card_type"] = "refine"
        with self.assertRaisesRegex(policy.PolicyError, "must first appear"):
            policy.validate_attempt_card(
                refine, run_id=refine["run_id"], expected_iteration=5,
                portfolio=portfolio(), candidate_path=refine["candidate_path"],
                candidate_sha256=refine["candidate_sha256"],
            )

    def test_research_basis_has_strict_keys_and_primary_target(self):
        value = portfolio()
        value["families"][0]["research_basis"][0]["quote"] = "caller supplied"
        with self.assertRaisesRegex(policy.PolicyError, "missing or extra"):
            policy.validate_portfolio(value)
        value = portfolio()
        value["families"][0]["research_basis"][0]["target"] = "risk"
        with self.assertRaisesRegex(policy.PolicyError, "mechanism"):
            policy.validate_portfolio(value)


if __name__ == "__main__":
    unittest.main()
