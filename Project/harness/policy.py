"""Pure policy helpers for the Track 2 official-run controller.

This module contains no model code and performs no scoring.  It exists so the
competition state machine can be tested against synthetic ledgers without
loading KuaiRand or consulting hidden labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


POLICY_ID = "track2.official-run.v2"
BENCHMARK = "KuaiRand-Pure"
OFFICIAL_VALIDATION_BASELINE_PRIMARY = 0.6016
EPSILON = 0.002
N_CONVERGE = 3
ITERATION_CAP = 50
WALL_CEILING_S = 6 * 3600
MAX_CONCLUSIVE_REVIEWS_PER_STAGE = 3
MAX_FAILED_REVIEWS_PER_STAGE = 3
MAX_OUTER_RUN_REQUESTS = ITERATION_CAP * (
    MAX_CONCLUSIVE_REVIEWS_PER_STAGE + MAX_FAILED_REVIEWS_PER_STAGE
)

FAMILY_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,48}")
CLAIM_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{2,63}")
TOPIC_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,31}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CARD_TYPES = {"explore", "refine", "recovery", "ensemble"}
MAX_PORTFOLIO_FAMILIES = 16
MAX_PORTFOLIO_SOURCE_BYTES = 256 * 1024
MAX_PORTFOLIO_VIEW_BYTES = 1024 * 1024
MAX_EVIDENCE_ITEMS = 16
MAX_TEXT = 12_000
MAX_EXPECTED_DELTA_ABS = 0.30
MAX_EXPECTED_DELTA_WIDTH = 0.10
RESEARCH_RELATIONSHIPS = {"supports", "constrains", "contradicts", "nearest"}
PORTFOLIO_RESEARCH_TARGETS = {"mechanism", "causal_claim", "falsifier", "risk"}
ATTEMPT_RESEARCH_TARGETS = {
    "mechanism", "hypothesis", "change", "falsifier", "why_now", "risk",
}


class PolicyError(ValueError):
    """A fail-closed policy or artifact-validation error."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def logical_entry(entry: dict) -> dict:
    """Remove transport-only journal authentication from policy semantics."""

    value = dict(entry)
    value.pop("journal_authority", None)
    return value


def first_run_start(entries: list[dict]) -> dict | None:
    return next((e for e in entries if e.get("type") == "run_start"), None)


def first_terminal(entries: list[dict]) -> dict | None:
    start = first_run_start(entries)
    if start is None:
        return None
    start_id = start.get("entry_id")
    seen = False
    for entry in entries:
        if entry.get("entry_id") == start_id:
            seen = True
            continue
        if seen and entry.get("type") == "run_terminated":
            return entry
    return None


def official_iterations(entries: list[dict], *, stop_at_terminal: bool = True) -> list[dict]:
    """Return official attempts after the first run_start.

    Setup rows before the marker are never official.  When ``stop_at_terminal``
    is true, later rows cannot contaminate the eligible prefix even if a ledger
    is accidentally appended after termination.
    """

    start = first_run_start(entries)
    if start is None:
        return []
    run_id = start.get("run_id")
    seen = False
    out: list[dict] = []
    for entry in entries:
        if entry.get("entry_id") == start.get("entry_id"):
            seen = True
            continue
        if not seen:
            continue
        if entry.get("type") == "run_terminated" and stop_at_terminal:
            break
        if entry.get("type") == "iteration" and entry.get("run_id") == run_id:
            out.append(entry)
    return out


def primary_score(entry: dict) -> float | None:
    metrics = entry.get("valid_metrics")
    if not isinstance(metrics, dict):
        return None
    score = metrics.get("primary")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    return score if math.isfinite(score) else None


def final_eligible(entry: dict) -> bool:
    seal = entry.get("sealed_test_scores")
    return (
        entry.get("error") is None
        and entry.get("eligible_for_final") is True
        and primary_score(entry) is not None
        and isinstance(seal, dict)
        and isinstance(seal.get("path"), str)
        and isinstance(seal.get("sha256"), str)
        and SHA256_RE.fullmatch(seal["sha256"]) is not None
    )


def best_eligible(attempts: list[dict]) -> dict | None:
    """Return the earliest checkpoint attaining the highest primary score."""

    best = None
    best_score = -math.inf
    for entry in attempts:
        if not final_eligible(entry):
            continue
        score = primary_score(entry)
        if score is not None and score > best_score:
            best = entry
            best_score = score
    return best


def convergence_reached(attempts: list[dict]) -> bool:
    """Apply the organizer rule to consecutive official attempts.

    The first attempt establishes an incumbent.  Once at least three later
    attempts exist, the run converges when none of the last three attempts
    improves the best checkpoint preceding that window by more than epsilon.
    A failed attempt is conservatively a non-improvement and remains one of the
    three consecutive iterations; it cannot be used to reset the window.
    """

    if len(attempts) <= N_CONVERGE:
        return False
    # Scores with a valid metrics object count for convergence even if their
    # sealed test artifact later proves unusable; convergence is a validation
    # rule, while final eligibility is a separate artifact rule.
    def successful_score(entry: dict) -> float | None:
        return primary_score(entry) if entry.get("error") is None else None

    prior_scores = [
        score for score in (successful_score(e) for e in attempts[:-N_CONVERGE])
        if score is not None
    ]
    if not prior_scores:
        return False
    window_scores = [
        score for score in (successful_score(e) for e in attempts[-N_CONVERGE:])
        if score is not None
    ]
    window_best = max(window_scores, default=-math.inf)
    return window_best <= max(prior_scores) + EPSILON


def earliest_attempt_terminal(attempts: list[dict]) -> tuple[int | None, list[str]]:
    """Return the first irreversible attempt boundary, never the latest one.

    This prefix scan is intentionally redundant with immediate controller
    latching.  Even if a caller ever appended an illegal later improvement,
    recomputation cannot make an earlier convergence disappear.
    """

    for count in range(1, len(attempts) + 1):
        prefix = attempts[:count]
        reasons: list[str] = []
        if convergence_reached(prefix):
            reasons.append("convergence")
        if count >= ITERATION_CAP:
            reasons.append("iteration_cap")
        if reasons:
            return count, reasons
    return None, []


def elapsed_seconds(entries: list[dict], now_epoch: float) -> float:
    start = first_run_start(entries)
    if start is None:
        return 0.0
    started = start.get("started_epoch")
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        raise PolicyError("run_start has no valid started_epoch")
    now = finite_epoch(now_epoch, "current epoch")
    begun = finite_epoch(started, "run_start.started_epoch")
    if now < begun:
        raise PolicyError("current epoch precedes run_start")
    return now - begun


def finite_epoch(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise PolicyError(f"{name} must be a finite number")
    return converted


def run_deadline(start: dict) -> float:
    begun = finite_epoch(start.get("started_epoch"), "run_start.started_epoch")
    deadline = finite_epoch(start.get("deadline_epoch"), "run_start.deadline_epoch")
    if deadline != begun + WALL_CEILING_S:
        raise PolicyError("run_start deadline is not exactly the six-hour ceiling")
    return deadline


def terminal_events(entries: list[dict], now_epoch: float) -> list[tuple[float, str]]:
    """Return triggered terminal events in chronological, deterministic order."""

    start = first_run_start(entries)
    if start is None:
        return []
    now = finite_epoch(now_epoch, "current epoch")
    deadline = run_deadline(start)
    attempts = official_iterations(entries, stop_at_terminal=False)
    boundary, attempt_reasons = earliest_attempt_terminal(attempts)
    events: list[tuple[float, str]] = []
    if boundary is not None:
        completed = finite_epoch(
            attempts[boundary - 1].get("completed_epoch"),
            f"iteration {boundary}.completed_epoch",
        )
        if completed <= now:
            events.extend((completed, reason) for reason in attempt_reasons)
    if deadline <= now:
        events.append((deadline, "wall_clock_ceiling"))
    order = {"convergence": 0, "iteration_cap": 1, "wall_clock_ceiling": 2}
    return sorted(events, key=lambda item: (item[0], order[item[1]]))


def triggered_reasons(entries: list[dict], now_epoch: float) -> list[str]:
    return [reason for _, reason in terminal_events(entries, now_epoch)]


def terminal_snapshot(entries: list[dict], now_epoch: float) -> dict:
    start = first_run_start(entries)
    if start is None:
        raise PolicyError("cannot terminate a run that has not started")
    all_attempts = official_iterations(entries, stop_at_terminal=False)
    events = terminal_events(entries, now_epoch)
    if not events:
        raise PolicyError("run is not terminal")
    attempt_limit, attempt_reasons = earliest_attempt_terminal(all_attempts)
    earliest_epoch = events[0][0]
    simultaneous = [reason for epoch, reason in events if epoch == earliest_epoch]
    deadline = run_deadline(start)
    attempt_event_epoch = None
    if attempt_limit is not None:
        attempt_event_epoch = finite_epoch(
            all_attempts[attempt_limit - 1].get("completed_epoch"),
            "attempt terminal completion",
        )
    if attempt_event_epoch is not None and attempt_event_epoch <= deadline:
        attempts = all_attempts[:attempt_limit]
    else:
        attempts = all_attempts
    eligible = [entry for entry in attempts if final_eligible(entry)]
    best = best_eligible(attempts)
    frozen = [
        {
            "entry_id": e.get("entry_id"),
            "iteration": e.get("iteration"),
            "primary": primary_score(e),
            "solution_sha256": (e.get("solution") or {}).get("sha256"),
            "sealed_sha256": (e.get("sealed_test_scores") or {}).get("sha256"),
        }
        for e in eligible
    ]
    return {
        "policy_id": POLICY_ID,
        "run_id": start.get("run_id"),
        "reason": events[0][1],
        "triggered_reasons": [reason for _, reason in events],
        "terminal_event_epoch": earliest_epoch,
        "simultaneous_reasons": simultaneous,
        "terminal_iteration": len(attempts),
        "eligible_entry_ids": [e.get("entry_id") for e in eligible],
        "eligible_best_entry_id": best.get("entry_id") if best else None,
        "eligible_best_validation_primary": primary_score(best) if best else None,
        "eligible_prefix_sha256": canonical_sha256(frozen),
        "official_prefix_sha256": canonical_sha256(
            [logical_entry(entry) for entry in attempts]
        ),
        "terminal_epoch": float(now_epoch),
    }


def validate_terminal_snapshot(entries: list[dict], terminal: dict) -> None:
    try:
        idx = entries.index(terminal)
    except ValueError as exc:
        raise PolicyError("terminal row is not in the ledger") from exc
    prefix = entries[:idx]
    expected = terminal_snapshot(prefix, float(terminal["terminal_epoch"]))
    fields = (
        "policy_id",
        "run_id",
        "reason",
        "triggered_reasons",
        "terminal_event_epoch",
        "simultaneous_reasons",
        "terminal_iteration",
        "eligible_entry_ids",
        "eligible_best_entry_id",
        "eligible_best_validation_primary",
        "eligible_prefix_sha256",
        "official_prefix_sha256",
    )
    mismatched = [name for name in fields if terminal.get(name) != expected.get(name)]
    if mismatched:
        raise PolicyError(f"terminal snapshot mismatch: {mismatched}")
    epoch = terminal.get("terminal_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(float(epoch)):
        raise PolicyError("terminal_epoch must be a finite number")
    prefix_attempts = official_iterations(prefix, stop_at_terminal=False)
    if len(prefix_attempts) != terminal.get("terminal_iteration"):
        raise PolicyError("official iteration was recorded past the first terminal boundary")
    later_attempts = [
        e for e in entries[idx + 1:]
        if e.get("type") == "iteration" and e.get("run_id") == terminal.get("run_id")
    ]
    if later_attempts:
        raise PolicyError("official iteration exists after the terminal row")


def _require_text(obj: dict, name: str, minimum: int, maximum: int = MAX_TEXT) -> str:
    value = obj.get(name)
    if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
        raise PolicyError(
            f"{name} must be text between {minimum} and {maximum} characters"
        )
    return value.strip()


def _validate_gain(value: Any, name: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise PolicyError(f"{name} must be an object with min/max")
    lo, hi = value.get("min"), value.get("max")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (lo, hi)):
        raise PolicyError(f"{name}.min/max must be numbers")
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi)) and float(lo) < float(hi)):
        raise PolicyError(f"{name} must satisfy finite min < max")
    if (
        abs(float(lo)) > MAX_EXPECTED_DELTA_ABS
        or abs(float(hi)) > MAX_EXPECTED_DELTA_ABS
        or float(hi) - float(lo) > MAX_EXPECTED_DELTA_WIDTH
    ):
        raise PolicyError(
            f"{name} must stay within +/-{MAX_EXPECTED_DELTA_ABS:.2f} and "
            f"have width <= {MAX_EXPECTED_DELTA_WIDTH:.2f}"
        )
    return {"min": float(lo), "max": float(hi)}


def _validate_topics(value: Any, name: str, *, maximum: int = 8) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise PolicyError(f"{name} must contain 1..{maximum} topics")
    if any(not isinstance(item, str) or TOPIC_RE.fullmatch(item) is None for item in value):
        raise PolicyError(f"{name} contains an invalid topic")
    if len(value) != len(set(value)):
        raise PolicyError(f"{name} contains duplicate topics")
    return value


def _validate_research_basis(
    value: Any, name: str, *, allowed_targets: set[str], primary_targets: set[str],
) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise PolicyError(f"{name} must contain 1..6 research-bank citations")
    seen: set[str] = set()
    primary = False
    for index, citation in enumerate(value):
        if not isinstance(citation, dict) or set(citation) != {
            "claim_id", "relationship", "target"
        }:
            raise PolicyError(f"{name}[{index}] has missing or extra fields")
        claim_id = citation.get("claim_id")
        if not isinstance(claim_id, str) or CLAIM_ID_RE.fullmatch(claim_id) is None:
            raise PolicyError(f"{name}[{index}].claim_id is invalid")
        if claim_id in seen:
            raise PolicyError(f"{name} contains duplicate claim_id {claim_id}")
        seen.add(claim_id)
        if citation.get("relationship") not in RESEARCH_RELATIONSHIPS:
            raise PolicyError(f"{name}[{index}].relationship is invalid")
        target = citation.get("target")
        if target not in allowed_targets:
            raise PolicyError(f"{name}[{index}].target is invalid")
        primary = primary or target in primary_targets
    if not primary:
        raise PolicyError(f"{name} lacks a mechanism/hypothesis-level citation")
    return value


def _validate_family_extension(
    value: Any, *, family_id: str, known_family_ids: set[str], name: str,
) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "nearest_family_ids", "bank_topics", "novel_topics", "mechanism_delta"
    }:
        raise PolicyError(f"{name} has missing or extra fields")
    nearest = value.get("nearest_family_ids")
    if (
        not isinstance(nearest, list)
        or not 1 <= len(nearest) <= 3
        or any(not isinstance(item, str) for item in nearest)
        or len(nearest) != len(set(nearest))
    ):
        raise PolicyError(f"{name}.nearest_family_ids must contain 1..3 unique ids")
    if family_id in nearest or any(item not in known_family_ids for item in nearest):
        raise PolicyError(f"{name}.nearest_family_ids must name other known families")
    _validate_topics(value.get("bank_topics"), f"{name}.bank_topics", maximum=4)
    novel = value.get("novel_topics")
    if (
        not isinstance(novel, list)
        or len(novel) > 4
        or any(not isinstance(item, str) or TOPIC_RE.fullmatch(item) is None for item in novel)
        or len(novel) != len(set(novel))
    ):
        raise PolicyError(f"{name}.novel_topics must contain 0..4 unique topics")
    _require_text(value, "mechanism_delta", 80, 2000)
    return value


def validate_portfolio(portfolio: dict) -> dict:
    if (
        not isinstance(portfolio, dict)
        or type(portfolio.get("schema_version")) is not int
        or portfolio.get("schema_version") != 2
    ):
        raise PolicyError("portfolio schema_version must be 2")
    base_keys = {
        "schema_version", "benchmark", "selection_rubric", "opening_order", "families"
    }
    correction_keys = {"corrects_review_id", "correction_summary"}
    if frozenset(portfolio) not in {
        frozenset(base_keys), frozenset(base_keys | correction_keys)
    }:
        raise PolicyError("portfolio has missing or extra fields")
    if bool(set(portfolio) & correction_keys) and not correction_keys <= set(portfolio):
        raise PolicyError("portfolio correction fields must appear together")
    if correction_keys <= set(portfolio):
        _require_text(portfolio, "corrects_review_id", 8, 128)
        _require_text(portfolio, "correction_summary", 30, 4000)
    if portfolio.get("benchmark") != BENCHMARK:
        raise PolicyError(f"portfolio benchmark must be {BENCHMARK}")
    families = portfolio.get("families")
    if (
        not isinstance(families, list)
        or not 4 <= len(families) <= MAX_PORTFOLIO_FAMILIES
    ):
        raise PolicyError(
            f"portfolio needs 4..{MAX_PORTFOLIO_FAMILIES} independent family cards"
        )
    ids: list[str] = []
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            raise PolicyError(f"families[{index}] must be an object")
        if set(family) != {
            "family_id", "mechanism", "causal_claim", "smallest_experiment",
            "falsifier", "known_risks", "expected_primary_delta", "bank_topics",
            "research_basis",
        }:
            raise PolicyError(f"families[{index}] has missing or extra fields")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or FAMILY_ID_RE.fullmatch(family_id) is None:
            raise PolicyError(f"families[{index}].family_id is invalid")
        if family_id in ids:
            raise PolicyError(f"duplicate family_id: {family_id}")
        ids.append(family_id)
        _require_text(family, "mechanism", 60)
        _require_text(family, "causal_claim", 50)
        _require_text(family, "smallest_experiment", 40)
        _require_text(family, "falsifier", 30)
        _require_text(family, "known_risks", 20)
        _validate_gain(family.get("expected_primary_delta"), "expected_primary_delta")
        _validate_topics(family.get("bank_topics"), f"families[{index}].bank_topics")
        _validate_research_basis(
            family.get("research_basis"), f"families[{index}].research_basis",
            allowed_targets=PORTFOLIO_RESEARCH_TARGETS,
            primary_targets={"mechanism", "causal_claim"},
        )
    opening = portfolio.get("opening_order")
    if (
        not isinstance(opening, list)
        or len(opening) < 4
        or any(not isinstance(item, str) for item in opening)
    ):
        raise PolicyError("opening_order must name at least four families")
    try:
        unique_opening = len(set(opening)) == len(opening)
    except TypeError as exc:
        raise PolicyError("opening_order values must be strings") from exc
    if not unique_opening or any(item not in ids for item in opening):
        raise PolicyError("opening_order must contain unique declared family ids")
    _require_text(portfolio, "selection_rubric", 100)
    return portfolio


def validate_attempt_card(
    card: dict,
    *,
    run_id: str,
    expected_iteration: int,
    portfolio: dict,
    candidate_path: str,
    candidate_sha256: str,
    registered_families: dict[str, dict] | None = None,
) -> dict:
    if (
        not isinstance(card, dict)
        or type(card.get("schema_version")) is not int
        or card.get("schema_version") != 2
    ):
        raise PolicyError("attempt card schema_version must be 2")
    base_keys = {
        "schema_version", "benchmark", "run_id", "iteration", "family_id",
        "card_type", "candidate_path", "candidate_sha256", "mechanism",
        "hypothesis", "change_summary", "falsifier", "why_now",
        "expected_primary_delta", "research_basis", "family_extension",
        "prior_outcomes_considered",
    }
    correction_keys = {"corrects_review_id", "correction_summary"}
    if frozenset(card) not in {
        frozenset(base_keys), frozenset(base_keys | correction_keys)
    }:
        raise PolicyError("attempt card has missing or extra fields")
    if bool(set(card) & correction_keys) and not correction_keys <= set(card):
        raise PolicyError("attempt correction fields must appear together")
    if correction_keys <= set(card):
        _require_text(card, "corrects_review_id", 8, 128)
        _require_text(card, "correction_summary", 30, 4000)
    if card.get("benchmark") != BENCHMARK:
        raise PolicyError(f"attempt benchmark must be {BENCHMARK}")
    if card.get("run_id") != run_id:
        raise PolicyError("attempt card run_id does not match the official run")
    if type(card.get("iteration")) is not int or card.get("iteration") != expected_iteration:
        raise PolicyError("attempt card is not bound to the next official iteration")
    seed_ids = {f["family_id"] for f in validate_portfolio(portfolio)["families"]}
    registrations = registered_families or {}
    if not isinstance(registrations, dict):
        raise PolicyError("registered_families must be an object")
    family_id = card.get("family_id")
    if not isinstance(family_id, str) or FAMILY_ID_RE.fullmatch(family_id) is None:
        raise PolicyError("attempt family_id is invalid")
    extension = card.get("family_extension")
    if family_id in seed_ids:
        if extension is not None:
            raise PolicyError("seed portfolio family forbids family_extension")
    elif family_id in registrations:
        expected_extension = registrations[family_id]
        if extension != expected_extension:
            raise PolicyError("registered family_extension does not match first use")
        _validate_family_extension(
            extension, family_id=family_id,
            known_family_ids=seed_ids | set(registrations),
            name="family_extension",
        )
    else:
        if expected_iteration <= 4:
            raise PolicyError(
                "a new family may only be registered after four seed-family attempts"
            )
        if card.get("card_type") != "explore":
            raise PolicyError("a new family must first appear on an explore card")
        _validate_family_extension(
            extension, family_id=family_id,
            known_family_ids=seed_ids | set(registrations),
            name="family_extension",
        )
    if card.get("card_type") not in CARD_TYPES:
        raise PolicyError(f"card_type must be one of {sorted(CARD_TYPES)}")
    if card.get("candidate_path") != candidate_path:
        raise PolicyError("attempt card candidate_path mismatch")
    if card.get("candidate_sha256") != candidate_sha256:
        raise PolicyError("attempt card candidate_sha256 mismatch")
    _require_text(card, "mechanism", 50)
    _require_text(card, "hypothesis", 50)
    _require_text(card, "change_summary", 40)
    _require_text(card, "falsifier", 30)
    _require_text(card, "why_now", 40)
    _validate_gain(card.get("expected_primary_delta"), "expected_primary_delta")
    _validate_research_basis(
        card.get("research_basis"), "attempt research_basis",
        allowed_targets=ATTEMPT_RESEARCH_TARGETS,
        primary_targets={"mechanism", "hypothesis"},
    )
    prior = card.get("prior_outcomes_considered")
    if not isinstance(prior, list):
        raise PolicyError("prior_outcomes_considered must be a list")
    if len(prior) > ITERATION_CAP or any(
        not isinstance(item, str) or not 1 <= len(item.strip()) <= 128 for item in prior
    ):
        raise PolicyError("prior_outcomes_considered must contain non-empty entry ids")
    return card
