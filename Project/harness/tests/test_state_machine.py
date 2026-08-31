from __future__ import annotations

import copy
import io
import math
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Project" / "harness"))
import iterate


policy = iterate.policy

RUN_ID = "0123456789abcdef0123456789abcdef"
BEGUN = 1_000.0
DEADLINE = BEGUN + policy.WALL_CEILING_S
NOTE_PATH = "Project/research/bank/notes/state-machine.md"
SEED_FAMILIES = ["seed_alpha", "seed_beta", "seed_gamma", "seed_delta"]
CANDIDATE_SOURCE = "HYPOTHESIS = 'synthetic'\ndef run(splits): return {}\n"
CANDIDATE_SHA256 = iterate.sha256_bytes(CANDIDATE_SOURCE.encode("utf-8"))
CANDIDATE_AST_SHA256 = iterate._candidate_fingerprint(CANDIDATE_SOURCE)


def digest(number: int) -> str:
    return f"{number:064x}"


def runtime_capability() -> dict:
    descriptor = {
        "format": iterate.sandbox.RUNTIME_ATTESTATION_FORMAT,
        "python": {
            "cache_tag": "cpython-314",
            "executable": "/usr/bin/python3.14",
            "sha256": digest(60_001),
            "size": 1024,
            "version": "3.14 synthetic",
        },
        "stdlib": {
            "root": "/usr/lib64/python3.14",
            "file_count": 600,
            "manifest_sha256": digest(60_002),
        },
        "numpy": {
            "version": "2.4.6",
            "file_count": 900,
            "manifest_sha256": digest(60_003),
        },
        "bubblewrap": {
            "path": "/usr/bin/bwrap",
            "sha256": digest(60_004),
            "size": 1024,
        },
    }
    return {
        "engine": "bubblewrap",
        "mount_namespace": True,
        "new_pid_namespace": True,
        "network_namespace": True,
        "raw_dataset_mounted": False,
        "parent_repo_mounted": False,
        "runtime_manifest_sha256": iterate.sandbox._canonical_sha256(descriptor),
        "runtime": descriptor,
    }


def metrics(primary: float) -> dict:
    return {
        "GAUC": primary + 0.05,
        "nDCG@5": primary - 0.05,
        "primary": primary,
        "rows": 100,
        "users": 20,
    }


def attempt_id(iteration: int) -> str:
    return f"{iteration:032x}"


def journal_row(entry_id: str, row_type: str, recorded_epoch: float, **fields) -> dict:
    return {
        "entry_id": entry_id,
        "type": row_type,
        "timestamp": "2026-08-30T00:00:00+00:00",
        "recorded_epoch": recorded_epoch,
        "harness_version": iterate.HARNESS_VERSION,
        **fields,
    }


def intervention_row(entry_id: str, recorded_epoch: float, run_id: str | None) -> dict:
    return journal_row(
        entry_id,
        "intervention",
        recorded_epoch,
        run_id=run_id,
        description="Synthetic disclosed intervention used as an adjacency gap.",
    )


def signed(rows: list[dict]) -> list[dict]:
    """Attach authority-shaped values without pretending to know the HMAC key."""

    result = copy.deepcopy(rows)
    previous = digest(900_000)
    for sequence, row in enumerate(result, 1):
        current = digest(900_000 + sequence)
        row["journal_authority"] = {
            "format": iterate.authority.FORMAT,
            "journal_id": "12345678123456781234567812345678",
            "sequence": sequence,
            "previous_hmac_sha256": previous,
            "hmac_sha256": current,
        }
        previous = current
    return result


def review_result(scope: str, number: int) -> dict:
    request_id = digest(10_000 + number)
    review_id = f"20260830-000000-{number:012x}"
    calls = [
        {
            "verdict": "APPROVE",
            "findings": [],
            "summary": f"Independent approval {call} for test {number}.",
            "response_id": f"resp_{number}_{call}",
            "actual_model": iterate.preflight_review.MODEL,
            "usage": {"total_tokens": 10},
            "transport_attempts": 1,
            "provider_request_id": f"provider_{number}_{call}",
        }
        for call in (1, 2)
    ]
    verdict, findings, summary = iterate.preflight_review._consensus(calls)
    return {
        "review_id": review_id,
        "request_id": request_id,
        "kind": scope,
        "requested_model": iterate.preflight_review.MODEL,
        "reasoning_effort": iterate.preflight_review.REASONING_EFFORT,
        "verdict": verdict,
        "findings": findings,
        "summary": summary,
        "packet_sha256": digest(20_000 + number),
        "calls": calls,
        "accepted": True,
        "packet_path": f"Project/audits/preflight/packet_{review_id}.json",
        "raw_log_path": f"Project/audits/preflight/review_{review_id}.json",
        "raw_log_sha256": digest(30_000 + number),
        "reviewer_tool_sha256": digest(40_000 + number),
    }


def review_row(
    scope: str,
    number: int,
    recorded_epoch: float,
    *,
    iteration: int | None = None,
    expired: bool = False,
) -> dict:
    review = review_result(scope, number)
    fields = {
        "scope": scope,
        "request_id": review["request_id"],
        "review": review,
        "error": None,
        "accepted": True,
    }
    if scope == "portfolio":
        fields.update(
            artifact_path="Project/research/portfolio.json",
            artifact_sha256=digest(50_000 + number),
        )
    else:
        fields["run_id"] = RUN_ID
    if iteration is not None:
        fields.update(
            iteration=iteration,
            artifact_path=f"Project/research/attempts/a{iteration:03d}.json",
            card_sha256=digest(71_000 + iteration),
            candidate_path=f"Project/solutions/s{iteration:03d}.py",
            candidate_sha256=CANDIDATE_SHA256,
        )
    if scope == "final":
        fields["designated_entry"] = "outcome-1"
    if expired:
        fields["expired_before_use"] = True
    return journal_row(f"review-{scope}-{number}", "preflight_review", recorded_epoch, **fields)


def bank_binding() -> dict:
    note_sha = digest(61_001)
    descriptor = {
        "schema_version": 1,
        "catalog": {
            "path": iterate.research_bank.CATALOG_PATH,
            "sha256": digest(61_002),
        },
        "notes": [{"path": NOTE_PATH, "sha256": note_sha}],
    }
    return {
        "snapshot_sha256": policy.canonical_sha256(descriptor),
        "descriptor": descriptor,
        "claim_count": 1,
        "known_topics": ["metric.within_user"],
    }


def base_unsigned() -> list[dict]:
    accepted = review_row("portfolio", 1, BEGUN - 1)
    frozen = {
        "path": "Project/research/portfolio.json",
        "sha256": accepted["artifact_sha256"],
        "canonical_sha256": digest(62_001),
        "review_id": accepted["review"]["review_id"],
        "request_id": accepted["request_id"],
        "family_ids": list(SEED_FAMILIES),
        "opening_order": list(SEED_FAMILIES),
        "resolved_research_sha256": digest(62_002),
    }
    start = journal_row(
        "run-start",
        "run_start",
        BEGUN,
        policy_id=policy.POLICY_ID,
        run_id=RUN_ID,
        benchmark=policy.BENCHMARK,
        started_epoch=BEGUN,
        deadline_epoch=DEADLINE,
        git_revision="a" * 40,
        trusted_components={
            path: digest(62_100 + index)
            for index, path in enumerate(iterate.TRUSTED_COMPONENTS)
        },
        portfolio=frozen,
        research_bank=bank_binding(),
        input_snapshot={
            "manifest_sha256": digest(62_004),
            "candidate_data_manifest_sha256": digest(62_005),
        },
        sandbox=runtime_capability(),
    )
    return [accepted, start]


def resolved_basis() -> dict:
    excerpt = "A bounded synthetic research-bank excerpt."
    return {
        "bank_snapshot_sha256": bank_binding()["snapshot_sha256"],
        "citations": [
            {
                "claim_id": "metric.within_user.c01",
                "relationship": "supports",
                "target": "hypothesis",
                "note_path": NOTE_PATH,
                "note_sha256": digest(61_001),
                "line_start": 1,
                "line_end": 1,
                "topics": ["metric.within_user"],
                "excerpt": excerpt,
                "excerpt_sha256": iterate.sha256_bytes(excerpt.encode("utf-8")),
            }
        ],
    }


def family_extension() -> dict:
    return {
        "nearest_family_ids": ["seed_alpha"],
        "bank_topics": ["metric.within_user"],
        "novel_topics": ["model.session_graph"],
        "mechanism_delta": "A materially distinct session mechanism " + "d" * 80,
    }


def attempt_card(
    iteration: int,
    family_id: str,
    prior_ids: list[str],
    *,
    extension: dict | None,
) -> dict:
    candidate_sha = CANDIDATE_SHA256
    return {
        "schema_version": 2,
        "benchmark": policy.BENCHMARK,
        "run_id": RUN_ID,
        "iteration": iteration,
        "family_id": family_id,
        "card_type": "explore" if extension is not None else "refine",
        "candidate_path": f"Project/solutions/s{iteration:03d}.py",
        "candidate_sha256": candidate_sha,
        "mechanism": "Mechanism description " + "m" * 60,
        "hypothesis": "Hypothesis description " + "h" * 60,
        "change_summary": "Change summary " + "c" * 50,
        "falsifier": "Falsifier description " + "f" * 40,
        "why_now": "Why this follows now " + "w" * 50,
        "expected_primary_delta": {"min": -0.01, "max": 0.01},
        "research_basis": [
            {
                "claim_id": "metric.within_user.c01",
                "relationship": "supports",
                "target": "hypothesis",
            }
        ],
        "family_extension": copy.deepcopy(extension),
        "prior_outcomes_considered": list(prior_ids),
        "path": f"Project/research/attempts/a{iteration:03d}.json",
        "sha256": digest(71_000 + iteration),
    }


def attempt_rows(
    iteration: int,
    family_id: str,
    prior_ids: list[str],
    *,
    score: float | None = None,
    extension: dict | None = None,
    registration: dict | None = None,
) -> list[dict]:
    started_epoch = BEGUN + iteration * 10.0
    completed_epoch = started_epoch + 1.0
    review = review_row(
        "attempt", 100 + iteration, started_epoch - 1.0, iteration=iteration
    )
    solution = {
        "path": f"Project/solutions/s{iteration:03d}.py",
        "sha256": CANDIDATE_SHA256,
        "normalized_ast_sha256": CANDIDATE_AST_SHA256,
        "source": CANDIDATE_SOURCE,
    }
    card = attempt_card(
        iteration, family_id, prior_ids, extension=extension
    )
    opened = journal_row(
        f"opened-{iteration}",
        "attempt_started",
        started_epoch,
        policy_id=policy.POLICY_ID,
        run_id=RUN_ID,
        attempt_id=attempt_id(iteration),
        iteration=iteration,
        started_epoch=started_epoch,
        solution=solution,
        card=card,
        resolved_research_basis=resolved_basis(),
        family_registration=copy.deepcopy(registration),
        preflight_review=copy.deepcopy(review["review"]),
    )
    if score is None:
        valid_metrics = None
        seal = None
        eligible = False
        error = "SyntheticFailure: controlled failed attempt"
    else:
        valid_metrics = metrics(score)
        seal = {
            "path": f"Project/results/sealed/i{iteration}.npy",
            "sha256": digest(73_000 + iteration),
        }
        eligible = True
        error = None
    outcome = journal_row(
        f"outcome-{iteration}",
        "iteration",
        completed_epoch,
        policy_id=policy.POLICY_ID,
        run_id=RUN_ID,
        attempt_id=attempt_id(iteration),
        iteration=iteration,
        started_epoch=started_epoch,
        completed_epoch=completed_epoch,
        wall_seconds=1.0,
        effective_timeout_seconds=30.0,
        solution=copy.deepcopy(solution),
        card=copy.deepcopy(card),
        resolved_research_basis=resolved_basis(),
        family_registration=copy.deepcopy(registration),
        preflight_review=copy.deepcopy(review["review"]),
        hypothesis=card["hypothesis"],
        sandbox={
            "engine": "bubblewrap",
            "candidate_path_inside": f"/candidate/s{iteration:03d}.py",
            "raw_dataset_mounted": False,
            "parent_repo_mounted": False,
            "new_pid_namespace": True,
            "network_namespace": True,
            "reviewed_candidate_sha256": CANDIDATE_SHA256,
            "sanitized_snapshot_manifest_sha256": digest(74_000 + iteration),
            "numpy_manifest_sha256": digest(75_000 + iteration),
            "worker_sha256": digest(76_000 + iteration),
            "organizer_manifest_sha256": digest(77_000 + iteration),
        },
        valid_metrics=valid_metrics,
        sealed_test_scores=seal,
        eligible_for_final=eligible,
        error=error,
        git_revision="a" * 40,
    )
    return [review, opened, outcome]


def ledger_with_attempts(
    families: list[str], scores: list[float | None]
) -> list[dict]:
    rows = base_unsigned()
    prior_ids: list[str] = []
    registrations: dict[str, dict] = {}
    for iteration, (family_id, score) in enumerate(zip(families, scores), 1):
        extension = None
        registration = None
        if family_id not in SEED_FAMILIES:
            extension = family_extension()
            registration = registrations.get(family_id)
            if registration is None:
                registration = {
                    "family_id": family_id,
                    "extension": copy.deepcopy(extension),
                    "extension_sha256": policy.canonical_sha256(extension),
                    "first_iteration": iteration,
                }
                registrations[family_id] = registration
        rows.extend(
            attempt_rows(
                iteration,
                family_id,
                prior_ids,
                score=score,
                extension=extension,
                registration=registration,
            )
        )
        prior_ids.append(f"outcome-{iteration}")
    return signed(rows)


def wall_terminal(rows: list[dict], *, terminal_id: str = "terminal-wall") -> list[dict]:
    unsigned = copy.deepcopy(rows)
    for row in unsigned:
        row.pop("journal_authority", None)
    terminal = journal_row(
        terminal_id,
        "run_terminated",
        DEADLINE,
        **policy.terminal_snapshot(unsigned, DEADLINE),
    )
    return signed(unsigned + [terminal])


def finalized_ledger() -> list[dict]:
    rows = wall_terminal(
        ledger_with_attempts(["seed_alpha"], [0.61])
    )
    unsigned = copy.deepcopy(rows)
    for row in unsigned:
        row.pop("journal_authority", None)
    terminal = next(row for row in unsigned if row["type"] == "run_terminated")
    final_review = review_row("final", 300, DEADLINE + 1.0)
    final_review["designated_entry"] = terminal["eligible_best_entry_id"]
    pending = journal_row(
        "final-pending",
        "final_pending",
        DEADLINE + 2.0,
        policy_id=policy.POLICY_ID,
        run_id=RUN_ID,
        designated_entry=terminal["eligible_best_entry_id"],
        terminal_entry_id=terminal["entry_id"],
        review=copy.deepcopy(final_review["review"]),
    )
    final = journal_row(
        "final",
        "final",
        DEADLINE + 3.0,
        policy_id=policy.POLICY_ID,
        run_id=RUN_ID,
        designated_entry=terminal["eligible_best_entry_id"],
        designated_solution={
            "path": "Project/solutions/s001.py",
            "sha256": CANDIDATE_SHA256,
        },
        valid_metrics=metrics(0.61),
        test_metrics_from_submitted_csv=metrics(0.60),
        baseline_test_primary=iterate.BASELINE_TEST_PRIMARY,
        delta_over_baseline=0.60 - iterate.BASELINE_TEST_PRIMARY,
        submission_csv="Project/results/final_submission_test.csv",
        submission_csv_sha256=digest(79_001),
        terminal_entry_id=terminal["entry_id"],
        final_review_id=final_review["review"]["review_id"],
    )
    return signed(unsigned + [final_review, pending, final])


class FakeAuthority:
    def __init__(self, starting_sequence: int):
        self.sequence = starting_sequence
        self.calls: list[list[dict]] = []

    def append(self, rows: list[dict]) -> list[dict]:
        appended = []
        for original in rows:
            self.sequence += 1
            row = copy.deepcopy(original)
            row["journal_authority"] = {
                "format": iterate.authority.FORMAT,
                "journal_id": "12345678123456781234567812345678",
                "sequence": self.sequence,
                "previous_hmac_sha256": digest(800_000 + self.sequence - 1),
                "hmac_sha256": digest(800_000 + self.sequence),
            }
            appended.append(row)
        self.calls.append(appended)
        return appended


class PredictionAssessmentTests(unittest.TestCase):
    @staticmethod
    def outcome(iteration: int, score: float | None) -> dict:
        return attempt_rows(
            iteration,
            SEED_FAMILIES[(iteration - 1) % len(SEED_FAMILIES)],
            [f"outcome-{number}" for number in range(1, iteration)],
            score=score,
        )[-1]

    def test_reference_uses_better_of_official_baseline_and_incumbent(self):
        first = self.outcome(1, 0.60)
        self.assertEqual(
            iterate._prediction_reference([]),
            {
                "kind": "official_baseline",
                "entry_id": None,
                "primary": policy.OFFICIAL_VALIDATION_BASELINE_PRIMARY,
            },
        )
        self.assertEqual(
            iterate._prediction_reference([first])["kind"],
            "official_baseline",
        )
        strong = self.outcome(2, 0.62)
        reference = iterate._prediction_reference([first, strong])
        self.assertEqual(reference["kind"], "eligible_incumbent")
        self.assertEqual(reference["entry_id"], strong["entry_id"])
        self.assertEqual(reference["primary"], 0.62)

    def test_prediction_boundaries_and_distance_are_deterministic(self):
        baseline = policy.OFFICIAL_VALIDATION_BASELINE_PRIMARY
        entry = self.outcome(1, baseline + 0.01)
        entry["card"]["expected_primary_delta"] = {"min": 0.01, "max": 0.02}
        within = iterate._prediction_assessment(entry, [])
        self.assertEqual(within["status"], "within_range")
        self.assertEqual(within["interval_distance"], 0.0)

        entry["valid_metrics"]["primary"] = baseline + 0.005
        below = iterate._prediction_assessment(entry, [])
        self.assertEqual(below["status"], "below_range")
        self.assertAlmostEqual(below["interval_distance"], 0.005)

        entry["valid_metrics"]["primary"] = baseline + 0.025
        above = iterate._prediction_assessment(entry, [])
        self.assertEqual(above["status"], "above_range")
        self.assertAlmostEqual(above["interval_distance"], 0.005)

    def test_failed_and_missing_metrics_are_not_given_causal_labels(self):
        failed = self.outcome(1, None)
        self.assertEqual(
            iterate._prediction_assessment(failed, [])["status"],
            "execution_failed",
        )
        missing = copy.deepcopy(failed)
        missing["error"] = None
        self.assertEqual(
            iterate._prediction_assessment(missing, [])["status"],
            "metric_missing",
        )

    def test_compact_prior_exposes_the_same_derived_assessment(self):
        rows = ledger_with_attempts(["seed_alpha"], [0.61])
        compact = iterate._compact_prior(rows)
        outcome = next(row for row in rows if row.get("type") == "iteration")
        self.assertEqual(
            compact[0]["prediction_assessment"],
            iterate._prediction_assessment(outcome, []),
        )


class LedgerStateMachineTests(unittest.TestCase):
    def validate(self, rows: list[dict], *, allow_open_attempt: bool = False) -> None:
        iterate.validate_ledger(
            rows,
            allow_open_attempt=allow_open_attempt,
            verify_review_evidence=False,
        )

    def test_minimal_started_ledger_is_valid(self):
        self.validate(signed(base_unsigned()))

    def test_accepted_reviews_are_single_use_adjacent_permits(self):
        base = base_unsigned()
        inserted = intervention_row("gap", BEGUN - 0.5, None)
        with self.assertRaisesRegex(iterate.ControllerError, "adjacent accepted portfolio"):
            self.validate(signed([base[0], inserted, base[1]]))

        one = ledger_with_attempts(["seed_alpha"], [0.61])
        opened_index = next(
            i for i, row in enumerate(one) if row["type"] == "attempt_started"
        )
        gap = intervention_row("attempt-gap", BEGUN + 9.5, RUN_ID)
        broken = one[:opened_index] + [gap] + one[opened_index:]
        with self.assertRaises(iterate.ControllerError):
            self.validate(signed(broken))

    def test_expired_attempt_review_is_only_valid_atomically_before_wall_terminal(self):
        rows = base_unsigned()
        expired = review_row(
            "attempt", 200, DEADLINE, iteration=1, expired=True
        )
        prospective = rows + [expired]
        terminal = journal_row(
            "terminal-expired-review",
            "run_terminated",
            DEADLINE,
            **policy.terminal_snapshot(prospective, DEADLINE),
        )
        legal = signed(prospective + [terminal])
        self.validate(legal)

        gap = intervention_row("expiry-gap", DEADLINE, RUN_ID)
        with self.assertRaisesRegex(iterate.ControllerError, "atomic wall terminal"):
            self.validate(signed(prospective + [gap, terminal]))

    def test_run_start_deadline_and_bank_portfolio_shapes_are_exact(self):
        cases = []
        wrong_deadline = base_unsigned()
        wrong_deadline[-1]["deadline_epoch"] += 1.0
        cases.append(wrong_deadline)

        extra_bank_field = base_unsigned()
        extra_bank_field[-1]["research_bank"]["unexpected"] = True
        cases.append(extra_bank_field)

        extra_portfolio_field = base_unsigned()
        extra_portfolio_field[-1]["portfolio"]["unexpected"] = True
        cases.append(extra_portfolio_field)

        for rows in cases:
            with self.subTest(mutated=rows[-1]):
                with self.assertRaises((iterate.ControllerError, policy.PolicyError)):
                    self.validate(signed(rows))

    def test_frozen_portfolio_hash_fields_must_be_real_hashes(self):
        rows = base_unsigned()
        rows[-1]["portfolio"]["canonical_sha256"] = "not-a-sha256"
        with self.assertRaises(iterate.ControllerError):
            self.validate(signed(rows))

    def test_open_and_outcome_bind_all_reviewed_evidence_exactly(self):
        field_mutations = {
            "solution": lambda value: value.update(sha256=digest(999_001)),
            "card": lambda value: value.update(hypothesis="changed after start"),
            "preflight_review": lambda value: value.update(summary="changed review"),
            "resolved_research_basis": lambda value: value.update(citations=[]),
            "family_registration": lambda _value: None,
        }
        for field, mutate in field_mutations.items():
            with self.subTest(field=field):
                rows = ledger_with_attempts(["seed_alpha"], [0.61])
                outcome = next(row for row in rows if row["type"] == "iteration")
                if field == "family_registration":
                    outcome[field] = {"forged": True}
                else:
                    mutate(outcome[field])
                with self.assertRaisesRegex(
                    iterate.ControllerError, "artifact binding mismatch"
                ):
                    self.validate(rows)

    def test_solution_and_card_records_have_exact_shapes(self):
        for field in ("solution", "card"):
            with self.subTest(field=field):
                rows = ledger_with_attempts(["seed_alpha"], [0.61])
                self.validate(rows)
                for row in rows:
                    if row["type"] in {"attempt_started", "iteration"}:
                        row[field]["unexpected"] = "authenticated but not controller-shaped"
                with self.assertRaises(iterate.ControllerError):
                    self.validate(rows)

    def test_attempt_embeds_the_exact_adjacent_review_result(self):
        rows = ledger_with_attempts(["seed_alpha"], [0.61])
        self.validate(rows)
        for row in rows:
            if row["type"] in {"attempt_started", "iteration"}:
                row["preflight_review"]["summary"] = "forged embedded summary"
        with self.assertRaises(iterate.ControllerError):
            self.validate(rows)

    def test_open_attempt_review_is_bound_to_the_same_iteration(self):
        rows = ledger_with_attempts(["seed_alpha"], [0.61])[:-1]
        self.validate(rows, allow_open_attempt=True)
        attempt_review = next(
            row
            for row in rows
            if row.get("type") == "preflight_review"
            and row.get("scope") == "attempt"
        )
        attempt_review["iteration"] = 999
        with self.assertRaises(iterate.ControllerError):
            self.validate(rows, allow_open_attempt=True)

    def test_accepted_review_requires_real_consensus_call_records(self):
        rows = base_unsigned()
        rows[0]["review"]["calls"] = [{}]
        with self.assertRaises(iterate.ControllerError):
            self.validate(signed(rows))

    def test_official_outcome_without_attempt_started_is_rejected(self):
        rows = base_unsigned()
        orphan = journal_row(
            "orphan-outcome",
            "iteration",
            BEGUN + 1.0,
            run_id=RUN_ID,
            attempt_id="orphan-attempt",
            iteration=1,
        )
        with self.assertRaisesRegex(iterate.ControllerError, "no preceding attempt_started"):
            self.validate(signed(rows + [orphan]))

    def test_legacy_setup_outcomes_before_run_start_remain_nonofficial(self):
        setup = {
            "entry_id": "legacy-setup",
            "type": "iteration",
            "timestamp": "2026-08-28T00:00:00+0800",
            "harness_version": "0.5.0-unfrozen",
            "phase": "setup",
            "iteration": 1,
            "valid_metrics": {"primary": 0.60},
        }
        self.validate([setup])
        self.assertEqual(policy.official_iterations([setup]), [])

    def test_orphan_accepted_portfolio_review_before_run_start_is_rejected(self):
        accepted = signed([review_row("portfolio", 909, BEGUN - 1.0)])
        with self.assertRaises(iterate.ControllerError):
            self.validate(accepted)

    def test_pre_run_intervention_and_failed_portfolio_review_are_allowed(self):
        failed = journal_row(
            "failed-portfolio-review",
            "preflight_review",
            BEGUN - 2.0,
            scope="portfolio",
            request_id=digest(90_001),
            review=None,
            error="ReviewError: synthetic transport failure",
            accepted=False,
            artifact_path="Project/research/portfolio.json",
            artifact_sha256=digest(90_002),
        )
        intervention = intervention_row("pre-run-intervention", BEGUN - 1.0, None)
        self.validate(signed([failed, intervention]))

    def test_unknown_or_attempt_scoped_pre_run_rows_are_rejected(self):
        unknown = journal_row(
            "unknown-pre-run",
            "controller_capability_grant",
            BEGUN - 2.0,
        )
        attempt_error = journal_row(
            "pre-run-attempt-review",
            "preflight_review",
            BEGUN - 1.0,
            scope="attempt",
            request_id=digest(91_001),
            review=None,
            error="ReviewError: impossible before a run",
            accepted=False,
            run_id=RUN_ID,
            iteration=1,
            artifact_path="Project/research/attempts/a001.json",
            card_sha256=digest(91_002),
            candidate_path="Project/solutions/s001.py",
            candidate_sha256=digest(91_003),
        )
        for row in (unknown, attempt_error):
            with self.subTest(row_type=row["type"], scope=row.get("scope")):
                with self.assertRaises(iterate.ControllerError):
                    self.validate(signed([row]))

    def test_crash_recovery_records_one_failed_outcome_without_rescoring(self):
        rows = ledger_with_attempts(["seed_alpha"], [0.61])
        rows = rows[:-1]
        auth = FakeAuthority(len(rows))

        with (
            mock.patch.object(iterate, "_review_evidence"),
            mock.patch.object(iterate.time, "time", return_value=BEGUN + 12.0),
            mock.patch.object(
                iterate, "DevelopmentTrusted", side_effect=AssertionError("rescored")
            ) as scorer,
            mock.patch.object(
                iterate.sandbox,
                "run_candidate",
                side_effect=AssertionError("candidate rerun"),
            ) as runner,
        ):
            recovered = iterate.recover_open_attempt(auth, rows)
            recovered_again = iterate.recover_open_attempt(auth, recovered)

        self.assertEqual(len(auth.calls), 1)
        self.assertEqual(recovered_again, recovered)
        scorer.assert_not_called()
        runner.assert_not_called()
        outcome = recovered[-1]
        self.assertEqual(outcome["type"], "iteration")
        self.assertEqual(outcome["attempt_id"], attempt_id(1))
        self.assertIsNone(outcome["valid_metrics"])
        self.assertIsNone(outcome["sealed_test_scores"])
        self.assertFalse(outcome["eligible_for_final"])
        self.assertIn("ControllerRecovery", outcome["error"])
        self.validate(recovered)

    def test_earliest_convergence_beats_a_later_improvement(self):
        attempts = []
        for iteration, score in enumerate([0.60, 0.60, 0.60, 0.60, 0.75], 1):
            attempts.append(
                {
                    "entry_id": f"policy-outcome-{iteration}",
                    "type": "iteration",
                    "run_id": RUN_ID,
                    "iteration": iteration,
                    "completed_epoch": BEGUN + iteration,
                    "error": None,
                    "eligible_for_final": True,
                    "valid_metrics": {"primary": score},
                    "solution": {"sha256": digest(iteration)},
                    "sealed_test_scores": {
                        "path": f"Project/results/sealed/p{iteration}.npy",
                        "sha256": digest(100 + iteration),
                    },
                }
            )
        rows = signed(base_unsigned() + attempts)
        snapshot = policy.terminal_snapshot(rows, BEGUN + 10.0)
        self.assertEqual(snapshot["reason"], "convergence")
        self.assertEqual(snapshot["terminal_iteration"], 4)
        self.assertEqual(snapshot["eligible_best_entry_id"], "policy-outcome-1")

    def test_no_attempt_may_open_after_a_terminal_marker(self):
        rows = wall_terminal(signed(base_unsigned()))
        unsigned = copy.deepcopy(rows)
        for row in unsigned:
            row.pop("journal_authority", None)
        late = attempt_rows(
            1, "seed_alpha", [], score=0.61
        )[:2]
        late[0]["recorded_epoch"] = DEADLINE + 1.0
        late[1]["recorded_epoch"] = DEADLINE + 2.0
        late[1]["started_epoch"] = BEGUN + 10.0
        with self.assertRaisesRegex(iterate.ControllerError, "terminal"):
            self.validate(signed(unsigned + late), allow_open_attempt=True)

    def test_first_four_attempts_are_distinct_seed_families(self):
        legal = ledger_with_attempts(SEED_FAMILIES, [0.61, 0.62, 0.63, 0.64])
        self.validate(legal)

        duplicate = ledger_with_attempts(
            ["seed_alpha", "seed_beta", "seed_gamma", "seed_alpha"],
            [0.61, 0.62, 0.63, 0.64],
        )
        with self.assertRaisesRegex(iterate.ControllerError, "seed-family|distinct seed"):
            self.validate(duplicate)

        extension = family_extension()
        registration = {
            "family_id": "new_family",
            "extension": extension,
            "extension_sha256": policy.canonical_sha256(extension),
            "first_iteration": 1,
        }
        rows = base_unsigned() + attempt_rows(
            1,
            "new_family",
            [],
            score=0.61,
            extension=extension,
            registration=registration,
        )
        with self.assertRaisesRegex(
            iterate.ControllerError, "seed portfolio|frozen seed-family order"
        ):
            self.validate(signed(rows))

    def test_new_family_registration_is_stable_from_first_use(self):
        families = SEED_FAMILIES + ["new_family", "new_family"]
        rows = ledger_with_attempts(
            families, [0.61, 0.62, 0.63, 0.64, 0.65, 0.66]
        )
        self.validate(rows)

        second = [
            row
            for row in rows
            if row.get("type") == "attempt_started" and row.get("iteration") == 6
        ][0]
        outcome = [
            row
            for row in rows
            if row.get("type") == "iteration" and row.get("iteration") == 6
        ][0]
        for row in (second, outcome):
            row["family_registration"]["extension"]["mechanism_delta"] += " changed"
            row["family_registration"]["extension_sha256"] = policy.canonical_sha256(
                row["family_registration"]["extension"]
            )
            row["card"]["family_extension"] = copy.deepcopy(
                row["family_registration"]["extension"]
            )
        with self.assertRaisesRegex(iterate.ControllerError, "changed after first use"):
            self.validate(rows)

    def test_new_family_registration_extension_has_an_exact_shape(self):
        rows = ledger_with_attempts(
            SEED_FAMILIES + ["new_family"],
            [0.61, 0.62, 0.63, 0.64, 0.65],
        )
        self.validate(rows)
        for row in rows:
            if row.get("type") not in {"attempt_started", "iteration"}:
                continue
            if (row.get("card") or {}).get("family_id") != "new_family":
                continue
            extension = row["family_registration"]["extension"]
            extension["unexpected"] = True
            row["family_registration"]["extension_sha256"] = policy.canonical_sha256(
                extension
            )
            row["card"]["family_extension"] = copy.deepcopy(extension)
        with self.assertRaises(iterate.ControllerError):
            self.validate(rows)

    def test_nonfinite_epochs_are_rejected(self):
        mutations = (
            ("recorded_epoch", math.nan),
            ("started_epoch", math.inf),
            ("deadline_epoch", -math.inf),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                rows = base_unsigned()
                rows[-1][field] = value
                with self.assertRaises((iterate.ControllerError, policy.PolicyError)):
                    self.validate(signed(rows))

        for field in ("started_epoch", "completed_epoch", "recorded_epoch"):
            with self.subTest(outcome_field=field):
                rows = ledger_with_attempts(["seed_alpha"], [0.61])
                outcome = next(row for row in rows if row["type"] == "iteration")
                outcome[field] = math.nan
                with self.assertRaises((iterate.ControllerError, policy.PolicyError)):
                    self.validate(rows)

    def test_nonfinite_persisted_timeout_and_wall_duration_are_rejected(self):
        for field in ("effective_timeout_seconds", "wall_seconds"):
            with self.subTest(field=field):
                rows = ledger_with_attempts(["seed_alpha"], [0.61])
                outcome = next(row for row in rows if row["type"] == "iteration")
                outcome[field] = math.nan
                with self.assertRaises(iterate.ControllerError):
                    self.validate(rows)

    def test_cli_rejects_nonfinite_timeouts_before_dispatch(self):
        cases = [
            ["iterate.py", "run", "--solution", "x.py", "--card", "x.json", "--timeout", "nan"],
            ["iterate.py", "final", "--review-timeout", "inf"],
        ]
        for argv in cases:
            with (
                self.subTest(argv=argv),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(sys, "stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    iterate.main()
                self.assertEqual(raised.exception.code, 2)

    def test_every_post_start_row_requires_a_finite_recorded_epoch(self):
        selectors = {
            "run_start": lambda row: row.get("type") == "run_start",
            "attempt_review": lambda row: row.get("type") == "preflight_review"
            and row.get("scope") == "attempt",
            "attempt_started": lambda row: row.get("type") == "attempt_started",
            "iteration": lambda row: row.get("type") == "iteration",
            "run_terminated": lambda row: row.get("type") == "run_terminated",
            "final_review": lambda row: row.get("type") == "preflight_review"
            and row.get("scope") == "final",
            "final_pending": lambda row: row.get("type") == "final_pending",
            "final": lambda row: row.get("type") == "final",
        }
        for label, selector in selectors.items():
            with self.subTest(row_type=label):
                rows = finalized_ledger()
                target = next(row for row in rows if selector(row))
                target["recorded_epoch"] = math.nan
                with self.assertRaises((iterate.ControllerError, policy.PolicyError)):
                    self.validate(rows)

    def test_controller_generated_rows_reject_extra_top_level_keys(self):
        cases = {
            "portfolio_review": (
                lambda: signed(base_unsigned()),
                lambda row: row["type"] == "preflight_review",
            ),
            "run_start": (
                lambda: signed(base_unsigned()),
                lambda row: row["type"] == "run_start",
            ),
            "attempt_started": (
                lambda: ledger_with_attempts(["seed_alpha"], [0.61]),
                lambda row: row["type"] == "attempt_started",
            ),
            "iteration": (
                lambda: ledger_with_attempts(["seed_alpha"], [0.61]),
                lambda row: row["type"] == "iteration",
            ),
            "run_terminated": (
                lambda: wall_terminal(signed(base_unsigned())),
                lambda row: row["type"] == "run_terminated",
            ),
            "final_review": (
                finalized_ledger,
                lambda row: row["type"] == "preflight_review"
                and row.get("scope") == "final",
            ),
            "final_pending": (
                finalized_ledger,
                lambda row: row["type"] == "final_pending",
            ),
            "final": (
                finalized_ledger,
                lambda row: row["type"] == "final",
            ),
        }

        for label, (build, selector) in cases.items():
            with self.subTest(row_type=label):
                rows = build()
                self.validate(rows)
                target = next(row for row in rows if selector(row))
                target["unexpected_authenticated_field"] = True
                with self.assertRaises(iterate.ControllerError):
                    self.validate(rows)

    def test_controller_generated_rows_reject_missing_required_keys(self):
        cases = {
            "portfolio_review": (
                lambda: signed(base_unsigned()),
                lambda row: row["type"] == "preflight_review",
                "artifact_path",
            ),
            "run_start": (
                lambda: signed(base_unsigned()),
                lambda row: row["type"] == "run_start",
                "git_revision",
            ),
            "attempt_started": (
                lambda: ledger_with_attempts(["seed_alpha"], [0.61]),
                lambda row: row["type"] == "attempt_started",
                "policy_id",
            ),
            "iteration": (
                lambda: ledger_with_attempts(["seed_alpha"], [0.61]),
                lambda row: row["type"] == "iteration",
                "git_revision",
            ),
            "run_terminated": (
                lambda: wall_terminal(signed(base_unsigned())),
                lambda row: row["type"] == "run_terminated",
                "harness_version",
            ),
            "final_review": (
                finalized_ledger,
                lambda row: row["type"] == "preflight_review"
                and row.get("scope") == "final",
                "designated_entry",
            ),
            "final_pending": (
                finalized_ledger,
                lambda row: row["type"] == "final_pending",
                "terminal_entry_id",
            ),
            "final": (
                finalized_ledger,
                lambda row: row["type"] == "final",
                "final_review_id",
            ),
        }

        for label, (build, selector, missing) in cases.items():
            with self.subTest(row_type=label, missing=missing):
                rows = build()
                self.validate(rows)
                target = next(row for row in rows if selector(row))
                target.pop(missing)
                with self.assertRaises(iterate.ControllerError):
                    self.validate(rows)

    def test_final_transition_binds_exact_review_and_terminal_ids(self):
        mutations = {
            "pending review": lambda rows: next(
                row for row in rows if row["type"] == "final_pending"
            )["review"].update(summary="forged embedded final review"),
            "pending terminal": lambda rows: next(
                row for row in rows if row["type"] == "final_pending"
            ).update(terminal_entry_id="wrong-terminal"),
            "final terminal": lambda rows: next(
                row for row in rows if row["type"] == "final"
            ).update(terminal_entry_id="wrong-terminal"),
            "final review": lambda rows: next(
                row for row in rows if row["type"] == "final"
            ).update(final_review_id="wrong-review"),
        }
        for label, mutate in mutations.items():
            with self.subTest(binding=label):
                rows = finalized_ledger()
                self.validate(rows)
                mutate(rows)
                with self.assertRaises(iterate.ControllerError):
                    self.validate(rows)

    def test_final_review_pending_final_is_once_only_and_adjacent(self):
        rows = finalized_ledger()
        self.validate(rows)

        duplicate_final = copy.deepcopy(rows[-1])
        duplicate_final["entry_id"] = "final-duplicate"
        duplicate_final["recorded_epoch"] += 1.0
        with self.assertRaisesRegex(iterate.ControllerError, "once-only"):
            self.validate(signed(rows + [duplicate_final]))

        gap = intervention_row("final-gap", DEADLINE + 2.5, RUN_ID)
        broken = rows[:-1] + [gap, rows[-1]]
        with self.assertRaisesRegex(iterate.ControllerError, "not adjacent"):
            self.validate(signed(broken))


if __name__ == "__main__":
    unittest.main()
