#!/usr/bin/env python3
"""Track 2 official-run controller, v2.0.0-rc1.

The controller, not an agent prompt, owns every state transition.  There are
no production override flags and no alternate production ledger.  A terminal
condition is appended beside the triggering outcome and remains irreversible.
Candidate code executes from the exact reviewed bytes in an OS sandbox that
cannot see raw test labels, the parent repository, or the network.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HARNESS_VERSION = "2.0.0-rc1"
ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = ROOT / "Project" / "harness"
TOOLS_DIR = ROOT / "Project" / "tools"
KIT = ROOT / "kuairand-starter-kit"
RAW_DATA_DIR = KIT / "KuaiRand-Pure" / "data"
SANITIZED_DIR = KIT / "KuaiRand-Pure" / "data_sanitized"
MANIFEST_PATH = ROOT / "Project" / "manifest.json"
JOURNAL_PATH = ROOT / "Project" / "results" / "JOURNAL.jsonl"
LOCK_PATH = ROOT / "Project" / "results" / ".controller.lock"
INHERITED_LOCK_FD_ENV = "TRACK2_CONTROLLER_LOCK_FD"
SEALED_DIR = ROOT / "Project" / "results" / "sealed"
FINAL_CSV = ROOT / "Project" / "results" / "final_submission_test.csv"
BASELINE_TEST_PRIMARY = 0.5946
GIT_BIN = Path("/usr/bin/git")
STATE_ROOT = Path.home() / ".local" / "state" / "tiktok-techjam-2026-track2"

def _load_exact_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trusted module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError(f"trusted module path mismatch for {name}")
    return module


policy = _load_exact_module("track2_policy", HARNESS_DIR / "policy.py")
MAX_CONCLUSIVE_REVIEWS_PER_STAGE = policy.MAX_CONCLUSIVE_REVIEWS_PER_STAGE
MAX_FAILED_REVIEWS_PER_STAGE = policy.MAX_FAILED_REVIEWS_PER_STAGE
sandbox = _load_exact_module("track2_sandbox", HARNESS_DIR / "sandbox.py")
authority = _load_exact_module("track2_authority", HARNESS_DIR / "authority.py")
input_snapshot = _load_exact_module(
    "track2_input_snapshot", HARNESS_DIR / "input_snapshot.py"
)
research_bank = _load_exact_module(
    "track2_research_bank", HARNESS_DIR / "research_bank.py"
)
preflight_review = _load_exact_module(
    "track2_preflight_review", TOOLS_DIR / "preflight_review.py"
)


TRUSTED_COMPONENTS = (
    "Project/harness/iterate.py",
    "Project/harness/policy.py",
    "Project/harness/sandbox.py",
    "Project/harness/candidate_worker.py",
    "Project/harness/authority.py",
    "Project/harness/input_snapshot.py",
    "Project/harness/research_bank.py",
    "Project/harness/controller_service.py",
    "Project/harness/researcher_shell.py",
    "Project/harness/controller_mcp_config.json",
    "Project/harness/claude_runtime.json",
    "Project/RESEARCHER_BRIEF.md",
    "Project/research/templates/attempt.template.json",
    "Project/tools/control.py",
    "Project/tools/controller_mcp.py",
    "Project/tools/init_researcher_workspace.py",
    "Project/tools/preflight_review.py",
    "Project/audits/preflight_schema.json",
    "Project/manifest.json",
    "kuairand-starter-kit/data.py",
    "kuairand-starter-kit/evaluate.py",
    "kuairand-starter-kit/submit.py",
    "kuairand-starter-kit/baseline.py",
    "kuairand-starter-kit/ablation_features.py",
    "kuairand-starter-kit/baseline_scores.json",
)

ALLOWED_IMPORTS = {
    "__future__", "array", "baseline", "bisect", "collections", "copy", "csv",
    "data", "dataclasses", "decimal", "evaluate", "fractions", "functools",
    "heapq", "itertools", "math", "numpy", "operator", "random", "statistics",
    "typing",
}
BLOCKED_CALLS = {
    "breakpoint", "compile", "eval", "exec", "input", "open", "__import__"
}
BLOCKED_ATTRIBUTES = {
    "__builtins__", "__code__", "__globals__", "__subclasses__", "_getframe",
}
BLOCKED_TEXT = {
    "raw dataset path": r"KuaiRand-Pure/data(?:/|['\"])",
    "network address": r"(?:https?://|ftp://|localhost|127\.0\.0\.1)",
    "process or network module": r"\b(?:subprocess|socket|requests|urllib|httpx|os|sys|inspect|ctypes|multiprocessing|signal|resource|importlib)\b",
    "kernel pseudo-filesystem": r"/(?:proc|sys|dev)(?:/|['\"])",
}

FEEDBACK_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time",
    "is_profile_enter",
]
TEST_DATE_START = 20220429


class ControllerError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def strict_json_bytes(data: bytes, label: str) -> dict:
    def no_duplicates(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ControllerError(f"{label} contains duplicate JSON key {key!r}")
            obj[key] = value
        return obj

    def no_constants(value):
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            data, object_pairs_hook=no_duplicates, parse_constant=no_constants
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ControllerError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{label} must contain one JSON object")
    return value


def strict_json_file(path: Path, label: str) -> dict:
    return strict_json_bytes(path.read_bytes(), label)


def git_revision() -> str:
    result = subprocess.run(
        [str(GIT_BIN), "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
        timeout=10, check=False,
    )
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", result.stdout.strip()):
        raise ControllerError("repository HEAD cannot be resolved")
    return result.stdout.strip()


def require_standalone_repository() -> None:
    git_dir = ROOT / ".git"
    if not git_dir.is_dir() or git_dir.is_symlink():
        raise ControllerError(
            "official execution requires a standalone clone with its own .git directory"
        )
    alternates = git_dir / "objects" / "info" / "alternates"
    if alternates.exists():
        raise ControllerError("shared/alternate Git object stores are forbidden")


def require_isolated_interpreter() -> None:
    """Keep owner commands and service commands on one exact Python runtime."""

    expected = Path("/usr/bin/python3").resolve(strict=True)
    try:
        actual = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise ControllerError("controller Python executable is unavailable") from exc
    if actual != expected or sys.flags.isolated != 1:
        raise ControllerError(
            "official controller commands require /usr/bin/python3 -I"
        )


def _run_start_runtime_manifest(start: dict) -> str:
    capability = start.get("sandbox")
    expected_keys = {
        "engine", "mount_namespace", "new_pid_namespace",
        "network_namespace", "raw_dataset_mounted", "parent_repo_mounted",
        "runtime_manifest_sha256", "runtime",
    }
    if (
        not isinstance(capability, dict)
        or set(capability) != expected_keys
        or capability.get("engine") != "bubblewrap"
        or capability.get("mount_namespace") is not True
        or capability.get("new_pid_namespace") is not True
        or capability.get("network_namespace") is not True
        or capability.get("raw_dataset_mounted") is not False
        or capability.get("parent_repo_mounted") is not False
    ):
        raise ControllerError("run_start sandbox capability binding is malformed")
    attestation = {
        "runtime_manifest_sha256": capability.get("runtime_manifest_sha256"),
        "runtime": capability.get("runtime"),
    }
    try:
        return sandbox.validate_runtime_attestation(attestation)
    except sandbox.SandboxError as exc:
        raise ControllerError(f"run_start runtime binding is invalid: {exc}") from exc


def authority_state_dir() -> Path:
    identity = sha256_bytes(str(ROOT.resolve()).encode("utf-8"))[:24]
    return STATE_ROOT / identity


def open_authority(*, create: bool, reconcile: bool) -> Any:
    state_dir = authority_state_dir()
    if create and not state_dir.exists():
        auth = authority.JournalAuthority.create(
            journal_path=JOURNAL_PATH, state_dir=state_dir, repo_root=ROOT
        )
    else:
        auth = authority.JournalAuthority.open_existing(
            journal_path=JOURNAL_PATH, state_dir=state_dir, repo_root=ROOT
        )
    auth.verify(reconcile_crash_suffix=reconcile)
    return auth


def open_input_snapshot(auth: Any, *, create: bool, manifest: dict | None = None):
    if create:
        if manifest is None:
            raise ControllerError("manifest is required to create the input snapshot")
        snapshot = input_snapshot.InputSnapshot.create(
            root=ROOT, state_dir=auth.state_dir, manifest=manifest
        )
    else:
        snapshot = input_snapshot.InputSnapshot.open_existing(state_dir=auth.state_dir)
    snapshot.verify()
    return snapshot


def open_research_bank(expected: dict | None = None):
    """Load exact committed bank bytes and optionally match run-start state."""

    bank = research_bank.load(ROOT)
    if expected is not None:
        if not isinstance(expected, dict) or set(expected) != {
            "snapshot_sha256", "descriptor", "claim_count", "known_topics"
        }:
            raise ControllerError("run_start research-bank binding is malformed")
        if (
            expected.get("snapshot_sha256") != bank.snapshot_sha256
            or expected.get("descriptor") != bank.descriptor
            or expected.get("claim_count") != len(bank.known_claims)
            or expected.get("known_topics") != list(bank.known_topics)
        ):
            raise ControllerError("frozen research bank differs from run_start")
    return bank


def _resolve_portfolio_research(bank: Any, portfolio: dict) -> dict:
    families = []
    for family in portfolio["families"]:
        resolved = bank.resolve_basis(
            family["research_basis"],
            allowed_relationships=policy.RESEARCH_RELATIONSHIPS,
            allowed_targets=policy.PORTFOLIO_RESEARCH_TARGETS,
        )
        cited_topics = {
            topic for citation in resolved["citations"]
            for topic in citation["topics"]
        }
        declared = set(family["bank_topics"])
        if not declared.issubset(cited_topics):
            raise ControllerError(
                f"family {family['family_id']} bank_topics are not covered by its "
                "controller-resolved citations"
            )
        families.append({
            "family_id": family["family_id"],
            "bank_topics": family["bank_topics"],
            "resolved_research_basis": resolved,
        })
    return {
        "bank_snapshot_sha256": bank.snapshot_sha256,
        "families": families,
    }


def _registered_families(entries: list[dict]) -> dict[str, dict]:
    """Return exact first-use extension records from authenticated starts."""

    registrations: dict[str, dict] = {}
    for row in entries:
        if row.get("type") != "attempt_started":
            continue
        registration = row.get("family_registration")
        if registration is None:
            continue
        if not isinstance(registration, dict) or set(registration) != {
            "family_id", "extension", "extension_sha256", "first_iteration"
        }:
            raise ControllerError("family registration has the wrong shape")
        family_id = registration.get("family_id")
        extension = registration.get("extension")
        if (
            not isinstance(family_id, str)
            or registration.get("extension_sha256")
            != policy.canonical_sha256(extension)
            or type(registration.get("first_iteration")) is not int
        ):
            raise ControllerError("family registration has an invalid binding")
        previous = registrations.get(family_id)
        if previous is None:
            if registration["first_iteration"] != row.get("iteration"):
                raise ControllerError("family first-use iteration is inconsistent")
            registrations[family_id] = registration
        elif previous != registration:
            raise ControllerError("family registration changed after first use")
    return registrations


def _resolve_attempt_research(
    bank: Any, card: dict, portfolio: dict, registrations: dict[str, dict]
) -> dict:
    resolved = bank.resolve_basis(
        card["research_basis"],
        allowed_relationships=policy.RESEARCH_RELATIONSHIPS,
        allowed_targets=policy.ATTEMPT_RESEARCH_TARGETS,
    )
    cited_topics = {
        topic for citation in resolved["citations"] for topic in citation["topics"]
    }
    seed = next(
        (family for family in portfolio["families"]
         if family["family_id"] == card["family_id"]),
        None,
    )
    if seed is not None:
        applicable_topics = set(seed["bank_topics"])
    else:
        registration = registrations.get(card["family_id"])
        extension = (
            registration["extension"] if registration is not None
            else card["family_extension"]
        )
        applicable_topics = set(extension["bank_topics"])
    if not applicable_topics.intersection(cited_topics):
        raise ControllerError(
            "attempt citations share no topic with their applicable mechanism family"
        )
    if seed is None and not applicable_topics.issubset(cited_topics):
        raise ControllerError(
            "new-family bank_topics are not fully covered by attempt citations"
        )
    return resolved


def committed_bytes(
    path: Path, allowed_parent: Path, suffix: str, *, maximum_bytes: int
) -> tuple[str, bytes]:
    if path.is_symlink():
        raise ControllerError(f"symlink artifacts are forbidden: {path}")
    resolved = path.resolve(strict=True)
    parent = allowed_parent.resolve(strict=True)
    if not resolved.is_relative_to(parent) or resolved.suffix != suffix:
        raise ControllerError(f"artifact must be a {suffix} file under {allowed_parent}")
    rel = resolved.relative_to(ROOT).as_posix()
    result = subprocess.run(
        [str(GIT_BIN), "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
        timeout=30, check=False,
    )
    if result.returncode != 0:
        raise ControllerError(f"artifact is not committed at HEAD: {rel}")
    if len(result.stdout) > maximum_bytes:
        raise ControllerError(f"artifact exceeds {maximum_bytes} bytes: {rel}")
    live = resolved.read_bytes()
    if live != result.stdout:
        raise ControllerError(f"artifact bytes differ from committed HEAD: {rel}")
    return rel, live


def verify_manifest(require_sanitized: bool = True) -> dict:
    manifest = strict_json_file(MANIFEST_PATH, "manifest")
    groups = [manifest.get("files", {}), manifest.get("dataset_files", {})]
    if require_sanitized:
        sanitized = manifest.get("dataset_files_sanitized")
        if not isinstance(sanitized, dict) or not sanitized:
            raise ControllerError("manifest has no sanitized dataset hashes")
        groups.append(sanitized)
    bad = []
    for group in groups:
        if not isinstance(group, dict):
            raise ControllerError("manifest hash section is malformed")
        for rel, expected in group.items():
            path = ROOT / rel
            if (
                not path.is_file()
                or path.is_symlink()
                or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                or sha256_file(path) != expected
            ):
                bad.append(rel)
    if bad:
        raise ControllerError(f"organizer/data integrity failure: {bad}")
    return manifest


def current_component_hashes(*, require_committed: bool) -> dict[str, str]:
    hashes = {}
    for rel in TRUSTED_COMPONENTS:
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            raise ControllerError(f"trusted component missing or symlinked: {rel}")
        live = path.read_bytes()
        if require_committed:
            result = subprocess.run(
                [str(GIT_BIN), "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
                timeout=30, check=False,
            )
            if result.returncode != 0 or result.stdout != live:
                raise ControllerError(f"trusted component is not exact committed HEAD: {rel}")
        hashes[rel] = sha256_bytes(live)
    return hashes


def verify_frozen_components(start: dict) -> None:
    expected = start.get("trusted_components")
    if not isinstance(expected, dict) or expected != current_component_hashes(
        require_committed=False
    ):
        raise ControllerError("trusted controller/component drift since run_start")


def base_entry(entry_type: str, **fields: Any) -> dict:
    epoch = time.time()
    return {
        "entry_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}",
        "type": entry_type,
        "timestamp": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
        "recorded_epoch": epoch,
        "harness_version": HARNESS_VERSION,
        **fields,
    }


def read_journal() -> list[dict]:
    if not JOURNAL_PATH.exists():
        return []
    entries = []
    for line_number, raw in enumerate(JOURNAL_PATH.read_bytes().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            entries.append(strict_json_bytes(raw, f"journal line {line_number}"))
        except ControllerError as exc:
            raise ControllerError(f"ledger integrity failure: {exc}") from exc
    return entries


def append_batch(auth: Any, history: list[dict], new_entries: list[dict]) -> list[dict]:
    if not new_entries:
        raise ControllerError("journal append batch may not be empty")
    ids = {entry.get("entry_id") for entry in history}
    if any(entry.get("entry_id") in ids for entry in new_entries):
        raise ControllerError("journal append would duplicate an entry_id")
    try:
        return auth.append(new_entries)
    except authority.AuthorityError as exc:
        raise ControllerError(f"external journal authority refused append: {exc}") from exc


@contextlib.contextmanager
def controller_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    inherited = os.environ.get(INHERITED_LOCK_FD_ENV)
    if inherited is not None:
        if re.fullmatch(r"[0-9]{1,9}", inherited) is None:
            raise ControllerError("inherited controller lock descriptor is malformed")
        fd = int(inherited)
        try:
            opened = os.fstat(fd)
            path_metadata = LOCK_PATH.stat(follow_symlinks=False)
        except OSError as exc:
            raise ControllerError("inherited controller lock is unavailable") from exc
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or opened.st_mode & 0o022
            or (opened.st_dev, opened.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise ControllerError("inherited controller lock identity is unsafe")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("inherited controller lock is not held") from exc
        yield
        return
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(LOCK_PATH, flags, 0o600)
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o022
        ):
            raise ControllerError("controller lock file is unsafe")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _finite_epoch(value: Any, label: str) -> float:
    try:
        return policy.finite_epoch(value, label)
    except policy.PolicyError as exc:
        raise ControllerError(str(exc)) from exc


def _review_evidence(review: dict) -> None:
    """Revalidate the private packet, raw log, and their exact consensus."""

    review_id = review.get("review_id")
    if not isinstance(review_id, str) or preflight_review.REVIEW_ID_RE.fullmatch(
        review_id
    ) is None:
        raise ControllerError("conclusive review has an invalid review_id")
    try:
        packet_bytes = preflight_review._read_review_evidence(
            root=ROOT,
            relative=review.get("packet_path"),
            expected_name=f"packet_{review_id}.json",
            maximum=preflight_review.MAX_PACKET_BYTES,
        )
        preflight_review._validate_result(
            result=review,
            root=ROOT,
            kind=review.get("kind"),
            request_id=review.get("request_id"),
            packet_sha256=review.get("packet_sha256"),
            packet_bytes=packet_bytes,
        )
    except preflight_review.ReviewError as exc:
        raise ControllerError(f"review evidence is invalid: {exc}") from exc


def _validate_review_row(row: dict, *, verify_evidence: bool) -> None:
    request_id = row.get("request_id")
    if not isinstance(request_id, str) or re.fullmatch(r"[0-9a-f]{64}", request_id) is None:
        raise ControllerError("preflight review row has no canonical request_id")
    review = row.get("review")
    error = row.get("error")
    if review is None:
        if not isinstance(error, str) or not error:
            raise ControllerError("non-conclusive review row must record an error")
        if row.get("accepted") is not False:
            raise ControllerError("review error cannot be accepted")
        return
    if error is not None or not isinstance(review, dict):
        raise ControllerError("conclusive review row has inconsistent error/review fields")
    required = {
        "review_id", "request_id", "kind", "requested_model", "reasoning_effort",
        "verdict", "findings", "summary", "packet_sha256", "calls", "accepted",
        "packet_path", "raw_log_path", "raw_log_sha256", "reviewer_tool_sha256",
    }
    if set(review) != required:
        raise ControllerError("conclusive review has missing or extra fields")
    if review.get("request_id") != request_id:
        raise ControllerError("review request_id differs from its journal row")
    expected_kind = {
        "portfolio": "portfolio", "attempt": "attempt", "final": "final"
    }.get(row.get("scope"))
    if expected_kind is None or review.get("kind") != expected_kind:
        raise ControllerError("review scope/kind mismatch")
    if (
        review.get("requested_model") != preflight_review.MODEL
        or review.get("reasoning_effort") != preflight_review.REASONING_EFFORT
        or not isinstance(review.get("review_id"), str)
        or preflight_review.REVIEW_ID_RE.fullmatch(review["review_id"]) is None
        or not isinstance(review.get("packet_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", review["packet_sha256"]) is None
        or not isinstance(review.get("raw_log_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", review["raw_log_sha256"]) is None
        or not isinstance(review.get("reviewer_tool_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", review["reviewer_tool_sha256"]) is None
        or not isinstance(review.get("packet_path"), str)
        or not isinstance(review.get("raw_log_path"), str)
    ):
        raise ControllerError("conclusive review has invalid identity metadata")
    try:
        preflight_review._validate_response({
            "verdict": review.get("verdict"),
            "findings": review.get("findings"),
            "summary": review.get("summary"),
        })
    except ValueError as exc:
        raise ControllerError(f"stored review verdict is invalid: {exc}") from exc
    calls = review.get("calls")
    if not isinstance(calls, list) or len(calls) not in {2, 3}:
        raise ControllerError("stored review has an invalid independent-call count")
    try:
        for call in calls:
            preflight_review._validate_call_record(call)
        first_two_agree = preflight_review._accept_class(
            calls[0]
        ) == preflight_review._accept_class(calls[1])
        if (len(calls) == 2) != first_two_agree:
            raise preflight_review.ReviewError(
                "review call sequence violates the consensus protocol"
            )
        verdict, findings, summary = preflight_review._consensus(calls)
    except preflight_review.ReviewError as exc:
        raise ControllerError(f"stored review calls are invalid: {exc}") from exc
    if {
        "verdict": verdict,
        "findings": findings,
        "summary": summary,
    } != {
        "verdict": review.get("verdict"),
        "findings": review.get("findings"),
        "summary": review.get("summary"),
    }:
        raise ControllerError("stored review result does not match its calls")
    accepted = verdict in preflight_review.ACCEPTED
    if review.get("accepted") is not accepted or row.get("accepted") is not accepted:
        raise ControllerError("review acceptance was not derived from its calls")
    if verify_evidence:
        _review_evidence(review)


def _validate_bank_binding(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "snapshot_sha256", "descriptor", "claim_count", "known_topics"
    }:
        raise ControllerError("run_start research-bank binding has the wrong shape")
    descriptor = value.get("descriptor")
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"schema_version", "catalog", "notes"}
        or descriptor.get("schema_version") != 1
        or type(value.get("claim_count")) is not int
        or not 1 <= value["claim_count"] <= research_bank.MAX_CLAIMS
        or not isinstance(value.get("known_topics"), list)
        or not 1 <= len(value["known_topics"]) <= (
            research_bank.MAX_CLAIMS * research_bank.MAX_TOPICS
        )
        or value["known_topics"] != sorted(set(value["known_topics"]))
        or any(
            not isinstance(topic, str) or policy.TOPIC_RE.fullmatch(topic) is None
            for topic in value["known_topics"]
        )
    ):
        raise ControllerError("run_start research-bank descriptor is malformed")
    catalog = descriptor.get("catalog")
    if (
        not isinstance(catalog, dict)
        or set(catalog) != {"path", "sha256"}
        or catalog.get("path") != research_bank.CATALOG_PATH
        or not isinstance(catalog.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", catalog["sha256"]) is None
    ):
        raise ControllerError("run_start research-bank catalog binding is malformed")
    notes = descriptor.get("notes")
    if not isinstance(notes, list) or not 1 <= len(notes) <= research_bank.MAX_NOTES:
        raise ControllerError("run_start research-bank notes are malformed")
    note_hashes: dict[str, str] = {}
    for note in notes:
        if (
            not isinstance(note, dict)
            or set(note) != {"path", "sha256"}
            or not isinstance(note.get("path"), str)
            or research_bank.NOTE_PATH_RE.fullmatch(note["path"]) is None
            or not isinstance(note.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", note["sha256"]) is None
            or note["path"] in note_hashes
        ):
            raise ControllerError("run_start research-bank note binding is malformed")
        note_hashes[note["path"]] = note["sha256"]
    if [note["path"] for note in notes] != sorted(note_hashes):
        raise ControllerError("run_start research-bank notes are not canonical")
    snapshot_sha = value.get("snapshot_sha256")
    if (
        not isinstance(snapshot_sha, str)
        or snapshot_sha != policy.canonical_sha256(descriptor)
    ):
        raise ControllerError("run_start research-bank snapshot hash is invalid")
    return note_hashes


def _validate_resolved_basis(
    value: Any, *, bank_snapshot_sha256: str, note_hashes: dict[str, str]
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"bank_snapshot_sha256", "citations"}
        or value.get("bank_snapshot_sha256") != bank_snapshot_sha256
        or not isinstance(value.get("citations"), list)
        or not 1 <= len(value["citations"]) <= 6
    ):
        raise ControllerError("resolved research basis has the wrong snapshot/shape")
    seen: set[str] = set()
    for citation in value["citations"]:
        if not isinstance(citation, dict) or set(citation) != {
            "claim_id", "relationship", "target", "note_path", "note_sha256",
            "line_start", "line_end", "topics", "excerpt", "excerpt_sha256",
        }:
            raise ControllerError("resolved research citation has wrong fields")
        claim_id = citation.get("claim_id")
        note_path = citation.get("note_path")
        excerpt = citation.get("excerpt")
        if (
            not isinstance(claim_id, str)
            or policy.CLAIM_ID_RE.fullmatch(claim_id) is None
            or claim_id in seen
            or citation.get("relationship") not in policy.RESEARCH_RELATIONSHIPS
            or citation.get("target") not in (
                policy.PORTFOLIO_RESEARCH_TARGETS | policy.ATTEMPT_RESEARCH_TARGETS
            )
            or not isinstance(note_path, str)
            or note_hashes.get(note_path) != citation.get("note_sha256")
            or type(citation.get("line_start")) is not int
            or type(citation.get("line_end")) is not int
            or not 1 <= citation["line_start"] <= citation["line_end"]
            or citation["line_end"] - citation["line_start"] + 1 > 12
            or not isinstance(citation.get("topics"), list)
            or not 1 <= len(citation["topics"]) <= research_bank.MAX_TOPICS
            or len(citation["topics"]) != len(set(citation["topics"]))
            or any(
                not isinstance(topic, str) or policy.TOPIC_RE.fullmatch(topic) is None
                for topic in citation["topics"]
            )
            or not isinstance(excerpt, str)
            or not excerpt.strip()
            or len(excerpt.encode("utf-8")) > research_bank.MAX_EXCERPT_BYTES
            or citation.get("excerpt_sha256")
            != sha256_bytes(excerpt.encode("utf-8"))
        ):
            raise ControllerError("resolved research citation binding is invalid")
        seen.add(claim_id)


_BASE_ROW_KEYS = {
    "entry_id", "type", "timestamp", "recorded_epoch", "harness_version",
    "journal_authority",
}
_SOLUTION_KEYS = {"path", "sha256", "normalized_ast_sha256", "source"}
_CARD_BASE_KEYS = {
    "schema_version", "benchmark", "run_id", "iteration", "family_id",
    "card_type", "candidate_path", "candidate_sha256", "mechanism",
    "hypothesis", "change_summary", "falsifier", "why_now",
    "expected_primary_delta", "research_basis", "family_extension",
    "prior_outcomes_considered", "path", "sha256",
}
_CARD_CORRECTION_KEYS = {"corrects_review_id", "correction_summary"}


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_common_row(row: dict, expected_type: str) -> None:
    if row.get("type") != expected_type or row.get("harness_version") != HARNESS_VERSION:
        raise ControllerError(f"{expected_type} row has invalid type/version")
    if not isinstance(row.get("timestamp"), str) or not row["timestamp"]:
        raise ControllerError(f"{expected_type} row has no timestamp")
    _finite_epoch(row.get("recorded_epoch"), f"{expected_type}.recorded_epoch")


def _validate_review_row_shape(row: dict) -> None:
    scope = row.get("scope")
    common = _BASE_ROW_KEYS | {
        "scope", "request_id", "review", "error", "accepted",
    }
    variants = {
        "portfolio": {"artifact_path", "artifact_sha256"},
        "attempt": {
            "run_id", "iteration", "artifact_path", "card_sha256",
            "candidate_path", "candidate_sha256",
        },
        "attempt_static": {
            "run_id", "iteration", "artifact_path", "card_sha256",
            "candidate_path", "candidate_sha256", "findings",
        },
        "final": {"run_id", "designated_entry"},
    }
    if scope not in variants:
        raise ControllerError("preflight review row has an invalid scope")
    allowed = common | variants[scope]
    if scope == "attempt":
        allowed_with_expiry = allowed | {"expired_before_use"}
        if frozenset(row) not in {frozenset(allowed), frozenset(allowed_with_expiry)}:
            raise ControllerError("preflight review row has missing or extra fields")
    elif set(row) != allowed:
        raise ControllerError("preflight review row has missing or extra fields")
    _validate_common_row(row, "preflight_review")
    if scope == "portfolio":
        if (
            not isinstance(row.get("artifact_path"), str)
            or not _valid_sha256(row.get("artifact_sha256"))
        ):
            raise ControllerError("portfolio review artifact binding is invalid")
    elif scope in {"attempt", "attempt_static"}:
        if (
            not isinstance(row.get("run_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", row["run_id"]) is None
            or type(row.get("iteration")) is not int
            or not 1 <= row["iteration"] <= policy.ITERATION_CAP
            or not isinstance(row.get("artifact_path"), str)
            or not isinstance(row.get("candidate_path"), str)
            or not _valid_sha256(row.get("card_sha256"))
            or not _valid_sha256(row.get("candidate_sha256"))
        ):
            raise ControllerError("attempt review artifact binding is invalid")
        if scope == "attempt_static" and (
            not isinstance(row.get("findings"), list)
            or not row["findings"]
            or len(row["findings"]) > 64
            or any(not isinstance(item, str) or not item for item in row["findings"])
        ):
            raise ControllerError("static review findings are invalid")
    elif (
        not isinstance(row.get("run_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", row["run_id"]) is None
        or not isinstance(row.get("designated_entry"), str)
        or not row["designated_entry"]
    ):
        raise ControllerError("final review identity binding is invalid")


def _validate_intervention_row(row: dict, run_id: str | None) -> None:
    if set(row) != _BASE_ROW_KEYS | {"run_id", "description"}:
        raise ControllerError("intervention row has missing or extra fields")
    _validate_common_row(row, "intervention")
    if (
        row.get("run_id") != run_id
        or not isinstance(row.get("description"), str)
        or not 1 <= len(row["description"].strip()) <= 4000
    ):
        raise ControllerError("intervention row is malformed")


def _validate_solution_and_card(
    solution: Any, card: Any, *, run_id: str, iteration: int
) -> None:
    if not isinstance(solution, dict) or set(solution) != _SOLUTION_KEYS:
        raise ControllerError("attempt solution record has missing or extra fields")
    source = solution.get("source")
    if (
        not isinstance(solution.get("path"), str)
        or re.fullmatch(
            r"Project/solutions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.py",
            solution["path"],
        ) is None
        or not _valid_sha256(solution.get("sha256"))
        or not _valid_sha256(solution.get("normalized_ast_sha256"))
        or not isinstance(source, str)
        or not 1 <= len(source.encode("utf-8")) <= 512 * 1024
        or sha256_bytes(source.encode("utf-8")) != solution["sha256"]
    ):
        raise ControllerError("attempt solution record is invalid")
    try:
        normalized = _candidate_fingerprint(source)
    except ControllerError as exc:
        raise ControllerError(f"attempt solution record is invalid: {exc}") from exc
    if normalized != solution["normalized_ast_sha256"]:
        raise ControllerError("attempt solution AST binding is invalid")

    if not isinstance(card, dict):
        raise ControllerError("attempt card record is not an object")
    allowed = {frozenset(_CARD_BASE_KEYS), frozenset(_CARD_BASE_KEYS | _CARD_CORRECTION_KEYS)}
    if frozenset(card) not in allowed:
        raise ControllerError("attempt card record has missing or extra fields")
    if (
        card.get("run_id") != run_id
        or card.get("iteration") != iteration
        or card.get("candidate_path") != solution["path"]
        or card.get("candidate_sha256") != solution["sha256"]
        or not isinstance(card.get("path"), str)
        or re.fullmatch(
            r"Project/research/attempts/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json",
            card["path"],
        ) is None
        or not _valid_sha256(card.get("sha256"))
    ):
        raise ControllerError("attempt card/solution identity binding is invalid")


def _validate_attempt_row_shapes(opened: dict, outcome: dict | None) -> None:
    opened_keys = _BASE_ROW_KEYS | {
        "policy_id", "run_id", "attempt_id", "iteration", "started_epoch",
        "solution", "card", "resolved_research_basis", "family_registration",
        "preflight_review",
    }
    if set(opened) != opened_keys:
        raise ControllerError("attempt_started has missing or extra fields")
    _validate_common_row(opened, "attempt_started")
    if outcome is None:
        return
    outcome_keys = _BASE_ROW_KEYS | {
        "policy_id", "run_id", "attempt_id", "iteration", "started_epoch",
        "completed_epoch", "wall_seconds", "effective_timeout_seconds",
        "solution", "card", "resolved_research_basis", "family_registration",
        "preflight_review", "hypothesis", "sandbox", "valid_metrics",
        "sealed_test_scores", "eligible_for_final", "error", "git_revision",
    }
    if set(outcome) != outcome_keys:
        raise ControllerError("iteration outcome has missing or extra fields")
    _validate_common_row(outcome, "iteration")


def validate_ledger(
    entries: list[dict], *, allow_open_attempt: bool = False,
    verify_review_evidence: bool = True,
) -> None:
    ids = [entry.get("entry_id") for entry in entries]
    if any(not isinstance(entry_id, str) or not entry_id for entry_id in ids):
        raise ControllerError("every journal row must have a non-empty entry_id")
    if len(ids) != len(set(ids)):
        raise ControllerError("journal contains duplicate entry_id values")
    starts = [entry for entry in entries if entry.get("type") == "run_start"]
    if len(starts) > 1:
        raise ControllerError("journal contains multiple run_start rows")
    if not starts:
        if any(
            entry.get("type") in {
                "run_terminated", "attempt_started", "final_pending", "final"
            }
            or (
                entry.get("type") == "iteration"
                and "journal_authority" in entry
            )
            for entry in entries
        ):
            raise ControllerError("official state exists before run_start")
        for row in entries:
            row_type = row.get("type")
            if row_type == "iteration" and "journal_authority" not in row:
                continue  # frozen legacy setup prefix
            if row_type == "intervention":
                _validate_intervention_row(row, None)
                continue
            if row_type == "preflight_review":
                _validate_review_row_shape(row)
                _validate_review_row(row, verify_evidence=verify_review_evidence)
                if row.get("scope") != "portfolio" or row.get("accepted") is not False:
                    raise ControllerError(
                        "orphan or non-portfolio review exists before run_start"
                    )
                continue
            raise ControllerError("unsupported authenticated state exists before run_start")
        return

    start = starts[0]
    start_index = entries.index(start)
    if (
        start.get("policy_id") != policy.POLICY_ID
        or not isinstance(start.get("run_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", start["run_id"]) is None
        or start.get("benchmark") != policy.BENCHMARK
        or set(start) != _BASE_ROW_KEYS | {
            "policy_id", "run_id", "benchmark", "started_epoch",
            "deadline_epoch", "git_revision", "trusted_components",
            "portfolio", "research_bank", "input_snapshot", "sandbox",
        }
    ):
        raise ControllerError("run_start policy/run_id/benchmark is invalid")
    _validate_common_row(start, "run_start")
    if "journal_authority" not in start:
        raise ControllerError("run_start is not protected by external authority")
    for row in entries[start_index:]:
        if "journal_authority" not in row:
            raise ControllerError("official journal row lacks external authentication")
    recorded = [
        _finite_epoch(row.get("recorded_epoch"), f"{row.get('entry_id')}.recorded_epoch")
        for row in entries[start_index:]
    ]
    if any(later < earlier for earlier, later in zip(recorded, recorded[1:])):
        raise ControllerError("official journal recorded_epoch moves backwards")

    run_id = start["run_id"]
    begun = _finite_epoch(start.get("started_epoch"), "run_start.started_epoch")
    deadline = policy.run_deadline(start)
    if recorded[0] < begun:
        raise ControllerError("run_start recorded before its started_epoch")
    snapshot = start.get("input_snapshot")
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"manifest_sha256", "candidate_data_manifest_sha256"}
        or any(
            not isinstance(snapshot.get(name), str)
            or re.fullmatch(r"[0-9a-f]{64}", snapshot[name]) is None
            for name in snapshot
        )
    ):
        raise ControllerError("run_start input snapshot binding is invalid")
    components = start.get("trusted_components")
    if (
        not isinstance(components, dict)
        or set(components) != set(TRUSTED_COMPONENTS)
        or any(not _valid_sha256(value) for value in components.values())
        or not isinstance(start.get("git_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", start["git_revision"]) is None
    ):
        raise ControllerError("run_start trusted-component binding is invalid")
    _run_start_runtime_manifest(start)
    bank_binding = start.get("research_bank")
    note_hashes = _validate_bank_binding(bank_binding)

    reviews = [row for row in entries if row.get("type") == "preflight_review"]
    for row in reviews:
        _validate_review_row_shape(row)
        _validate_review_row(row, verify_evidence=verify_review_evidence)
    conclusive: dict[str, dict] = {}
    for row in reviews:
        if row.get("review") is None:
            continue
        request_id = row["request_id"]
        if request_id in conclusive:
            raise ControllerError("canonical review request has multiple conclusive verdicts")
        conclusive[request_id] = row
    if len(_conclusive_stage_reviews(entries, scope="portfolio")) > (
        MAX_CONCLUSIVE_REVIEWS_PER_STAGE
    ):
        raise ControllerError("portfolio semantic-review budget was exceeded")
    for iteration in range(1, policy.ITERATION_CAP + 1):
        if len(_conclusive_stage_reviews(
            entries, scope="attempt", iteration=iteration
        )) > MAX_CONCLUSIVE_REVIEWS_PER_STAGE:
            raise ControllerError(
                f"attempt iteration {iteration} semantic-review budget was exceeded"
            )

    portfolio_review = entries[start_index - 1] if start_index else None
    frozen_portfolio = start.get("portfolio")
    if (
        not isinstance(portfolio_review, dict)
        or portfolio_review.get("type") != "preflight_review"
        or portfolio_review.get("scope") != "portfolio"
        or portfolio_review.get("accepted") is not True
        or not isinstance(frozen_portfolio, dict)
        or frozen_portfolio.get("review_id")
        != (portfolio_review.get("review") or {}).get("review_id")
        or frozen_portfolio.get("request_id") != portfolio_review.get("request_id")
        or frozen_portfolio.get("sha256") != portfolio_review.get("artifact_sha256")
        or set(frozen_portfolio) != {
            "path", "sha256", "canonical_sha256", "review_id", "request_id",
            "family_ids", "opening_order", "resolved_research_sha256",
        }
        or not isinstance(frozen_portfolio.get("family_ids"), list)
        or len(frozen_portfolio["family_ids"]) < 4
        or len(frozen_portfolio["family_ids"])
        != len(set(frozen_portfolio["family_ids"]))
        or any(
            not isinstance(family_id, str)
            or policy.FAMILY_ID_RE.fullmatch(family_id) is None
            for family_id in frozen_portfolio["family_ids"]
        )
        or not isinstance(frozen_portfolio.get("opening_order"), list)
        or len(frozen_portfolio["opening_order"]) < 4
        or len(frozen_portfolio["opening_order"])
        != len(set(frozen_portfolio["opening_order"]))
        or any(
            family_id not in frozen_portfolio["family_ids"]
            for family_id in frozen_portfolio["opening_order"]
        )
        or not isinstance(frozen_portfolio.get("resolved_research_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", frozen_portfolio["resolved_research_sha256"]
        ) is None
        or not _valid_sha256(frozen_portfolio.get("sha256"))
        or not _valid_sha256(frozen_portfolio.get("canonical_sha256"))
        or not isinstance(frozen_portfolio.get("path"), str)
        or re.fullmatch(r"Project/research/[A-Za-z0-9][A-Za-z0-9._/-]{0,190}\.json", frozen_portfolio["path"]) is None
        or not isinstance(frozen_portfolio.get("request_id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", frozen_portfolio["request_id"]) is None
        or not isinstance(frozen_portfolio.get("review_id"), str)
        or preflight_review.REVIEW_ID_RE.fullmatch(frozen_portfolio["review_id"]) is None
        or "journal_authority" not in portfolio_review
    ):
        raise ControllerError("run_start lacks one adjacent accepted portfolio review")

    official_types = {
        "attempt_started", "iteration", "run_terminated", "final_pending", "final"
    }
    for row in entries[start_index + 1:]:
        if row.get("type") not in official_types | {"preflight_review", "intervention"}:
            raise ControllerError("unsupported row type exists after run_start")
        if row.get("type") in official_types and row.get("run_id") != run_id:
            raise ControllerError("official row belongs to another run")
        if row.get("type") == "intervention":
            _validate_intervention_row(row, run_id)

    attempts = policy.official_iterations(entries, stop_at_terminal=False)
    started_rows = [
        row for row in entries[start_index + 1:]
        if row.get("type") == "attempt_started"
    ]
    if len({row.get("attempt_id") for row in started_rows}) != len(started_rows):
        raise ControllerError("duplicate attempt_started attempt_id")
    if len({row.get("attempt_id") for row in attempts}) != len(attempts):
        raise ControllerError("duplicate iteration attempt_id")
    by_started = {row.get("attempt_id"): row for row in started_rows}
    by_outcome = {row.get("attempt_id"): row for row in attempts}
    if any(attempt_id not in by_started for attempt_id in by_outcome):
        raise ControllerError("official outcome has no preceding attempt_started")
    open_ids = [attempt_id for attempt_id in by_started if attempt_id not in by_outcome]
    if len(open_ids) > 1 or (open_ids and not allow_open_attempt):
        raise ControllerError("journal has an incomplete official attempt")

    previous_completion = begun
    for expected_iteration, outcome in enumerate(attempts, 1):
        if outcome.get("iteration") != expected_iteration:
            raise ControllerError("official attempts have an iteration sequence gap")
        opened = by_started[outcome.get("attempt_id")]
        _validate_attempt_row_shapes(opened, outcome)
        if opened.get("iteration") != expected_iteration:
            raise ControllerError("attempt_started/outcome iteration mismatch")
        opened_index = entries.index(opened)
        outcome_index = entries.index(outcome)
        if outcome_index != opened_index + 1:
            raise ControllerError("attempt outcome is not adjacent to its start marker")
        review_row = entries[opened_index - 1] if opened_index else None
        if (
            not isinstance(review_row, dict)
            or review_row.get("type") != "preflight_review"
            or review_row.get("scope") != "attempt"
            or review_row.get("accepted") is not True
            or review_row.get("iteration") != expected_iteration
            or review_row.get("request_id")
            != (opened.get("preflight_review") or {}).get("request_id")
            or (review_row.get("review") or {}).get("review_id")
            != (opened.get("preflight_review") or {}).get("review_id")
            or review_row.get("review") != opened.get("preflight_review")
        ):
            raise ControllerError("attempt start lacks one adjacent accepted review")
        if (
            opened.get("policy_id") != policy.POLICY_ID
            or outcome.get("policy_id") != policy.POLICY_ID
            or opened.get("run_id") != run_id
            or outcome.get("run_id") != run_id
            or not isinstance(opened.get("attempt_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", opened["attempt_id"]) is None
            or outcome.get("attempt_id") != opened.get("attempt_id")
            or opened.get("solution") != outcome.get("solution")
            or opened.get("card") != outcome.get("card")
            or opened.get("resolved_research_basis")
            != outcome.get("resolved_research_basis")
            or opened.get("family_registration")
            != outcome.get("family_registration")
            or opened.get("preflight_review") != outcome.get("preflight_review")
        ):
            raise ControllerError("attempt start/outcome artifact binding mismatch")
        _validate_solution_and_card(
            opened.get("solution"), opened.get("card"),
            run_id=run_id, iteration=expected_iteration,
        )
        started_epoch = _finite_epoch(opened.get("started_epoch"), "attempt started_epoch")
        if _finite_epoch(outcome.get("started_epoch"), "outcome started_epoch") != started_epoch:
            raise ControllerError("outcome started_epoch differs from attempt start")
        completed_epoch = _finite_epoch(
            outcome.get("completed_epoch"), "attempt completed_epoch"
        )
        wall_seconds = _finite_epoch(outcome.get("wall_seconds"), "outcome wall_seconds")
        if wall_seconds < 0 or not math.isclose(
            wall_seconds, completed_epoch - started_epoch, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ControllerError("outcome wall_seconds is inconsistent")
        effective_timeout = outcome.get("effective_timeout_seconds")
        if effective_timeout is not None:
            effective_timeout = _finite_epoch(
                effective_timeout, "outcome effective_timeout_seconds"
            )
            if not 0 < effective_timeout <= policy.WALL_CEILING_S:
                raise ControllerError("outcome effective timeout is out of range")
        elif not str(outcome.get("error") or "").startswith("ControllerRecovery:"):
            raise ControllerError("only crash recovery may omit effective timeout")
        if (
            not isinstance(outcome.get("hypothesis"), str)
            or len(outcome["hypothesis"].encode("utf-8")) > 4000
            or not isinstance(outcome.get("git_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", outcome["git_revision"]) is None
        ):
            raise ControllerError("outcome hypothesis/revision is invalid")
        error = outcome.get("error")
        if error is not None and (
            not isinstance(error, str) or not 1 <= len(error) <= 12_000
        ):
            raise ControllerError("outcome error record is invalid")
        metrics = outcome.get("valid_metrics")
        if metrics is not None and (
            not isinstance(metrics, dict)
            or "primary" not in metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise ControllerError("outcome validation metrics are invalid")
        if (
            started_epoch < previous_completion
            or started_epoch >= deadline
            or completed_epoch < started_epoch
        ):
            raise ControllerError("attempt chronology crosses an illegal boundary")
        if _finite_epoch(opened.get("recorded_epoch"), "attempt start recorded_epoch") < started_epoch:
            raise ControllerError("attempt start was recorded before it began")
        if _finite_epoch(outcome.get("recorded_epoch"), "outcome recorded_epoch") < completed_epoch:
            raise ControllerError("attempt outcome was recorded before completion")
        derived_eligible = (
            outcome.get("error") is None
            and completed_epoch <= deadline
            and policy.primary_score(outcome) is not None
            and isinstance(outcome.get("sealed_test_scores"), dict)
            and policy.final_eligible({**outcome, "eligible_for_final": True})
        )
        if outcome.get("eligible_for_final") is not derived_eligible:
            raise ControllerError("outcome eligibility was not derived from evidence/time")
        previous_completion = completed_epoch

    seed_family_ids = set(frozen_portfolio["family_ids"])
    registered: dict[str, dict] = {}
    for opened in started_rows:
        if opened.get("attempt_id") not in by_outcome:
            _validate_attempt_row_shapes(opened, None)
            _validate_solution_and_card(
                opened.get("solution"), opened.get("card"),
                run_id=run_id, iteration=opened.get("iteration"),
            )
        _validate_resolved_basis(
            opened.get("resolved_research_basis"),
            bank_snapshot_sha256=bank_binding["snapshot_sha256"],
            note_hashes=note_hashes,
        )
        card = opened.get("card")
        if not isinstance(card, dict):
            raise ControllerError("attempt start has no bound card")
        family_id = card.get("family_id")
        registration = opened.get("family_registration")
        if family_id in seed_family_ids:
            if registration is not None or card.get("family_extension") is not None:
                raise ControllerError("seed family illegally carries an extension")
            continue
        if not isinstance(registration, dict) or set(registration) != {
            "family_id", "extension", "extension_sha256", "first_iteration"
        }:
            raise ControllerError("new family lacks a strict registration")
        if (
            registration.get("family_id") != family_id
            or registration.get("extension") != card.get("family_extension")
            or registration.get("extension_sha256")
            != policy.canonical_sha256(registration.get("extension"))
        ):
            raise ControllerError("family registration/card binding is inconsistent")
        try:
            policy._validate_family_extension(
                registration.get("extension"),
                family_id=family_id,
                known_family_ids=seed_family_ids | set(registered),
                name="family_registration.extension",
            )
        except policy.PolicyError as exc:
            raise ControllerError(
                f"family registration extension is invalid: {exc}"
            ) from exc
        previous = registered.get(family_id)
        if previous is None:
            if registration.get("first_iteration") != opened.get("iteration"):
                raise ControllerError("new family first-use iteration is inconsistent")
            registered[family_id] = registration
        elif previous != registration:
            raise ControllerError("family registration changed after first use")
    opening_sample = [
        (row.get("card") or {}).get("family_id") for row in started_rows[:4]
    ]
    if (
        opening_sample != frozen_portfolio["opening_order"][:len(opening_sample)]
    ):
        raise ControllerError(
            "opening attempts do not follow the frozen seed-family order"
        )

    if open_ids:
        opened = by_started[open_ids[0]]
        if opened.get("iteration") != len(attempts) + 1 or entries[-1] is not opened:
            raise ControllerError("incomplete attempt is not the unique official tail")
        opened_index = entries.index(opened)
        review_row = entries[opened_index - 1] if opened_index else None
        if (
            opened.get("policy_id") != policy.POLICY_ID
            or opened.get("run_id") != run_id
            or not isinstance(opened.get("attempt_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}", opened["attempt_id"]) is None
            or not isinstance(review_row, dict)
            or review_row.get("type") != "preflight_review"
            or review_row.get("scope") != "attempt"
            or review_row.get("accepted") is not True
            or review_row.get("iteration") != opened.get("iteration")
            or review_row.get("review") != opened.get("preflight_review")
        ):
            raise ControllerError("open attempt lacks its exact adjacent review")
        open_started = _finite_epoch(
            opened.get("started_epoch"), "open attempt started_epoch"
        )
        if open_started < previous_completion or open_started >= deadline:
            raise ControllerError("open attempt began at/after the wall deadline")
        if _finite_epoch(
            opened.get("recorded_epoch"), "open attempt recorded_epoch"
        ) < open_started:
            raise ControllerError("open attempt was recorded before it began")

    boundary, _ = policy.earliest_attempt_terminal(attempts)
    if boundary is not None and len(attempts) > boundary:
        raise ControllerError("official iteration exists past the first terminal boundary")

    terminals = [row for row in entries if row.get("type") == "run_terminated"]
    if len(terminals) > 1:
        raise ControllerError("journal contains multiple terminal rows")
    if boundary is not None and not terminals:
        raise ControllerError("attempt terminal boundary lacks an atomic terminal row")
    if terminals:
        terminal = terminals[0]
        if set(terminal) != _BASE_ROW_KEYS | {
            "policy_id", "run_id", "reason", "triggered_reasons",
            "terminal_event_epoch", "simultaneous_reasons", "terminal_iteration",
            "eligible_entry_ids", "eligible_best_entry_id",
            "eligible_best_validation_primary", "eligible_prefix_sha256",
            "official_prefix_sha256", "terminal_epoch",
        }:
            raise ControllerError("terminal row has missing or extra fields")
        _validate_common_row(terminal, "run_terminated")
        if any(entries.index(opened) > entries.index(terminal) for opened in started_rows):
            raise ControllerError("attempt_started exists after terminal state")
        try:
            policy.validate_terminal_snapshot(entries, terminal)
        except policy.PolicyError as exc:
            raise ControllerError(f"invalid terminal snapshot: {exc}") from exc
        terminal_index = entries.index(terminal)
        if terminal.get("reason") in {"convergence", "iteration_cap"}:
            trigger = attempts[terminal["terminal_iteration"] - 1]
            if terminal_index != entries.index(trigger) + 1:
                raise ControllerError("attempt-triggered terminal was not latched atomically")

    pending = [row for row in entries if row.get("type") == "final_pending"]
    finals = [row for row in entries if row.get("type") == "final"]
    if len(pending) > 1 or len(finals) > 1:
        raise ControllerError("once-only final state is duplicated")
    if pending:
        marker = pending[0]
        if set(marker) != _BASE_ROW_KEYS | {
            "policy_id", "run_id", "designated_entry", "terminal_entry_id", "review"
        }:
            raise ControllerError("final_pending has missing or extra fields")
        _validate_common_row(marker, "final_pending")
        marker_index = entries.index(marker)
        prior = entries[marker_index - 1] if marker_index else None
        if (
            not terminals
            or marker.get("designated_entry")
            != terminals[0].get("eligible_best_entry_id")
            or marker.get("policy_id") != policy.POLICY_ID
            or marker.get("run_id") != run_id
            or marker.get("terminal_entry_id") != terminals[0].get("entry_id")
            or not isinstance(prior, dict)
            or prior.get("type") != "preflight_review"
            or prior.get("scope") != "final"
            or prior.get("accepted") is not True
            or prior.get("request_id") != (marker.get("review") or {}).get("request_id")
            or (prior.get("review") or {}).get("review_id")
            != (marker.get("review") or {}).get("review_id")
            or prior.get("review") != marker.get("review")
        ):
            raise ControllerError("final_pending lacks its exact terminal/final review link")
    if finals:
        final = finals[0]
        if set(final) != _BASE_ROW_KEYS | {
            "policy_id", "run_id", "designated_entry", "designated_solution",
            "valid_metrics", "test_metrics_from_submitted_csv",
            "baseline_test_primary", "delta_over_baseline", "submission_csv",
            "submission_csv_sha256", "terminal_entry_id", "final_review_id",
        }:
            raise ControllerError("final row has missing or extra fields")
        _validate_common_row(final, "final")
        if not pending or entries.index(finals[0]) != entries.index(pending[0]) + 1:
            raise ControllerError("final is not adjacent to its once-only pending marker")
        target = next(
            (entry for entry in attempts if entry.get("entry_id") == final.get("designated_entry")),
            None,
        )
        test_metrics = final.get("test_metrics_from_submitted_csv")
        delta = final.get("delta_over_baseline")
        if (
            final.get("designated_entry") != pending[0].get("designated_entry")
            or final.get("policy_id") != policy.POLICY_ID
            or final.get("run_id") != run_id
            or final.get("terminal_entry_id") != pending[0].get("terminal_entry_id")
            or final.get("final_review_id")
            != (pending[0].get("review") or {}).get("review_id")
            or target is None
            or final.get("designated_solution") != {
                "path": (target.get("solution") or {}).get("path"),
                "sha256": (target.get("solution") or {}).get("sha256"),
            }
            or final.get("valid_metrics") != target.get("valid_metrics")
            or not isinstance(test_metrics, dict)
            or "primary" not in test_metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in test_metrics.values()
            )
            or final.get("baseline_test_primary") != BASELINE_TEST_PRIMARY
            or isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            or not math.isclose(
                float(delta), float(test_metrics["primary"]) - BASELINE_TEST_PRIMARY,
                rel_tol=0.0, abs_tol=1e-12,
            )
            or final.get("submission_csv")
            != "Project/results/final_submission_test.csv"
            or not _valid_sha256(final.get("submission_csv_sha256"))
        ):
            raise ControllerError("final does not match its final_pending marker")

    # An accepted semantic permit is single-use and must be consumed by the
    # exact adjacent state transition.  It can never sit in the journal as a
    # reusable capability.  The only exception is an attempt review that
    # expired while the wall clock crossed; that row must atomically precede
    # the wall-clock terminal marker.
    for review_row in reviews:
        if review_row.get("accepted") is not True:
            continue
        index = entries.index(review_row)
        following = entries[index + 1] if index + 1 < len(entries) else None
        scope = review_row.get("scope")
        if scope == "portfolio":
            if following is not start:
                raise ControllerError("accepted portfolio review is not consumed once")
        elif scope == "attempt":
            expired = review_row.get("expired_before_use")
            if expired is True:
                if (
                    not isinstance(following, dict)
                    or following.get("type") != "run_terminated"
                    or following.get("reason") != "wall_clock_ceiling"
                ):
                    raise ControllerError("expired attempt review lacks atomic wall terminal")
            elif expired is not None:
                raise ControllerError("expired_before_use may only be the literal true")
            elif (
                not isinstance(following, dict)
                or following.get("type") != "attempt_started"
                or (following.get("preflight_review") or {}).get("request_id")
                != review_row.get("request_id")
                or (following.get("preflight_review") or {}).get("review_id")
                != (review_row.get("review") or {}).get("review_id")
            ):
                raise ControllerError("accepted attempt review is not consumed once")
        elif scope == "final":
            if (
                not isinstance(following, dict)
                or following.get("type") != "final_pending"
                or (following.get("review") or {}).get("request_id")
                != review_row.get("request_id")
            ):
                raise ControllerError("accepted final review is not consumed once")


def _make_terminal(history: list[dict], now_epoch: float) -> dict:
    return {
        **base_entry("run_terminated"),
        **policy.terminal_snapshot(history, now_epoch),
    }


def append_outcome_and_terminal(
    auth: Any, history: list[dict], outcome: dict, now_epoch: float
) -> list[dict]:
    prospective = history + [outcome]
    batch = [outcome]
    if policy.triggered_reasons(prospective, now_epoch):
        batch.append(_make_terminal(prospective, now_epoch))
    return append_batch(auth, history, batch)


def latch_terminal_if_due(
    auth: Any, entries: list[dict], now_epoch: float
) -> tuple[list[dict], dict | None]:
    terminal = policy.first_terminal(entries)
    if terminal is not None:
        return entries, terminal
    if not policy.triggered_reasons(entries, now_epoch):
        return entries, None
    written = append_batch(auth, entries, [_make_terminal(entries, now_epoch)])
    return entries + written, written[0]


def recover_open_attempt(auth: Any, entries: list[dict]) -> list[dict]:
    validate_ledger(entries, allow_open_attempt=True)
    start = policy.first_run_start(entries)
    if start is None:
        return entries
    attempts = policy.official_iterations(entries, stop_at_terminal=False)
    outcomes = {entry.get("attempt_id") for entry in attempts}
    open_attempts = [
        entry for entry in entries
        if entry.get("type") == "attempt_started"
        and entry.get("run_id") == start.get("run_id")
        and entry.get("attempt_id") not in outcomes
    ]
    if not open_attempts:
        return entries
    opened = open_attempts[0]
    completed_epoch = time.time()
    outcome = base_entry(
        "iteration",
        policy_id=policy.POLICY_ID,
        run_id=start["run_id"],
        attempt_id=opened["attempt_id"],
        iteration=opened["iteration"],
        solution=opened["solution"],
        card=opened["card"],
        resolved_research_basis=opened["resolved_research_basis"],
        family_registration=opened["family_registration"],
        preflight_review=opened["preflight_review"],
        valid_metrics=None,
        sealed_test_scores=None,
        eligible_for_final=False,
        error="ControllerRecovery: attempt began but no outcome was durably recorded",
        started_epoch=opened["started_epoch"],
        completed_epoch=completed_epoch,
        wall_seconds=max(0.0, completed_epoch - float(opened["started_epoch"])),
        effective_timeout_seconds=None,
        hypothesis=(opened.get("card") or {}).get("hypothesis", "")[:4000],
        sandbox=None,
        git_revision=git_revision(),
    )
    written = append_outcome_and_terminal(auth, entries, outcome, completed_epoch)
    return entries + written


def scan_candidate(source: str) -> list[str]:
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    findings.append(f"import is outside allowlist: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                findings.append(f"import is outside allowlist: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                findings.append(f"blocked dynamic-execution call: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRIBUTES:
            findings.append(f"blocked introspection attribute: {node.attr}")
    for label, pattern in BLOCKED_TEXT.items():
        if re.search(pattern, source, re.IGNORECASE):
            findings.append(f"blocked source reference: {label}")
    return sorted(set(findings))


def _load_frozen_source(name: str, path: Path):
    """Execute one exact snapshot file without consulting import search paths."""

    payload = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(payload, str(path), "exec"), module.__dict__)
    if Path(module.__file__).resolve() != path.resolve():
        raise ControllerError(f"organizer module identity changed during load: {path}")
    return module


class DevelopmentTrusted:
    """Validation scorer with sanitized data only; no raw test label is loaded."""

    def __init__(self, snapshot):
        data_module = _load_frozen_source(
            "track2_organizer_data_dev",
            snapshot.file("kuairand-starter-kit/data.py"),
        )
        evaluate_module = _load_frozen_source(
            "track2_organizer_evaluate_dev",
            snapshot.file("kuairand-starter-kit/evaluate.py"),
        )
        self._evaluate = evaluate_module.evaluate
        self.splits = data_module.load(str(snapshot.sanitized_dir))
        if any(row[6] != 0 for row in self.splits["test"]):
            raise ControllerError("sanitized test rows contain a nonzero label")
        if any(row[0] > 20220421 for row in self.splits["train"]):
            raise ControllerError("training split crosses 2022-04-21")

    @staticmethod
    def _array(scores: Any, expected: int, label: str):
        import numpy as np

        arr = np.asarray(scores, dtype=np.float64)
        if arr.shape != (expected,) or not np.all(np.isfinite(arr)):
            raise ControllerError(f"{label} predictions fail shape/finiteness checks")
        return arr

    def score_valid(self, scores: Any) -> dict:
        rows = self.splits["valid"]
        arr = self._array(scores, len(rows), "validation")
        metrics = self._evaluate([row[1] for row in rows], [row[6] for row in rows], arr)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in metrics.values()):
            raise ControllerError("organizer evaluator returned non-finite metrics")
        return metrics

    def seal_test(self, entry_id: str, scores: Any) -> dict:
        import numpy as np

        arr = self._array(scores, len(self.splits["test"]), "test")
        SEALED_DIR.mkdir(parents=True, exist_ok=True)
        path = SEALED_DIR / f"{entry_id}.npy"
        if path.exists():
            raise ControllerError("unique sealed artifact path already exists")
        np.save(path, arr, allow_pickle=False)
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}


class FinalTrusted:
    """The only object that loads raw test labels; constructed after final_pending."""

    def __init__(self, snapshot):
        data_module = _load_frozen_source(
            "track2_organizer_data_final",
            snapshot.file("kuairand-starter-kit/data.py"),
        )
        evaluate_module = _load_frozen_source(
            "track2_organizer_evaluate_final",
            snapshot.file("kuairand-starter-kit/evaluate.py"),
        )
        previous_data = sys.modules.get("data")
        previous_evaluate = sys.modules.get("evaluate")
        sys.modules["data"] = data_module
        sys.modules["evaluate"] = evaluate_module
        try:
            submit_module = _load_frozen_source(
                "track2_organizer_submit_final",
                snapshot.file("kuairand-starter-kit/submit.py"),
            )
        finally:
            if previous_data is None:
                sys.modules.pop("data", None)
            else:
                sys.modules["data"] = previous_data
            if previous_evaluate is None:
                sys.modules.pop("evaluate", None)
            else:
                sys.modules["evaluate"] = previous_evaluate

        self._evaluate = evaluate_module.evaluate
        self._read_submission = submit_module.read_submission
        self._write_submission = submit_module.write_submission
        self.splits = data_module.load(str(snapshot.raw_dir))
        self._test_labels = [row[6] for row in self.splits["test"]]

    def load_sealed(self, target: dict):
        import numpy as np

        seal = target["sealed_test_scores"]
        expected = SEALED_DIR / f"{target['entry_id']}.npy"
        unresolved = ROOT / seal["path"]
        actual = unresolved.resolve(strict=True)
        if actual != expected.resolve() or unresolved.is_symlink():
            raise ControllerError("sealed artifact path is not the frozen entry path")
        if sha256_file(actual) != seal["sha256"]:
            raise ControllerError("sealed artifact hash differs from terminal snapshot")
        arr = np.load(actual, allow_pickle=False)
        if arr.shape != (len(self.splits["test"]),) or not np.all(np.isfinite(arr)):
            raise ControllerError("sealed artifact fails final shape/finiteness checks")
        return arr

    def write_check_score(self, scores: Any) -> dict:
        rows = self.splits["test"]
        temporary = FINAL_CSV.with_name(
            f".{FINAL_CSV.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        if FINAL_CSV.exists():
            raise ControllerError("final submission path already exists before final scoring")
        self._write_submission(str(temporary), rows, scores)
        parsed = self._read_submission(str(temporary), rows)
        metrics = self._evaluate([row[1] for row in rows], self._test_labels, parsed)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, FINAL_CSV)
        directory_fd = os.open(FINAL_CSV.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return metrics


def _portfolio_from_start(start: dict) -> dict:
    frozen = start.get("portfolio")
    if not isinstance(frozen, dict):
        raise ControllerError("run_start has no frozen portfolio")
    path = ROOT / frozen.get("path", "")
    live = path.read_bytes()
    if sha256_bytes(live) != frozen.get("sha256"):
        raise ControllerError("frozen portfolio bytes changed after run_start")
    value = strict_json_bytes(live, "frozen portfolio")
    policy.validate_portfolio(value)
    if policy.canonical_sha256(value) != frozen.get("canonical_sha256"):
        raise ControllerError("frozen portfolio semantic hash changed")
    if (
        [family["family_id"] for family in value["families"]]
        != frozen.get("family_ids")
        or value["opening_order"] != frozen.get("opening_order")
    ):
        raise ControllerError("frozen portfolio family registry changed")
    return value


PREDICTION_STATUSES = (
    "within_range", "below_range", "above_range",
    "execution_failed", "metric_missing",
)


def _prediction_reference(prior_attempts: list[dict]) -> dict:
    """Return the public score against which the next delta is interpreted."""

    baseline = float(policy.OFFICIAL_VALIDATION_BASELINE_PRIMARY)
    incumbent = policy.best_eligible(prior_attempts)
    incumbent_score = policy.primary_score(incumbent) if incumbent else None
    if incumbent_score is not None and incumbent_score > baseline:
        return {
            "kind": "eligible_incumbent",
            "entry_id": incumbent["entry_id"],
            "primary": incumbent_score,
        }
    return {
        "kind": "official_baseline",
        "entry_id": None,
        "primary": baseline,
    }


def _prediction_assessment(entry: dict, prior_attempts: list[dict]) -> dict:
    """Derive measurement/calibration facts without claiming causal meaning."""

    reference = _prediction_reference(prior_attempts)
    expected = (entry.get("card") or {}).get("expected_primary_delta")
    if (
        not isinstance(expected, dict)
        or set(expected) != {"min", "max"}
        or any(
            isinstance(expected.get(name), bool)
            or not isinstance(expected.get(name), (int, float))
            or not math.isfinite(float(expected[name]))
            for name in ("min", "max")
        )
    ):
        raise ControllerError("official outcome has no valid expected delta range")
    expected_delta = {
        "min": float(expected["min"]),
        "max": float(expected["max"]),
    }
    expected_primary = {
        "min": reference["primary"] + expected_delta["min"],
        "max": reference["primary"] + expected_delta["max"],
    }
    observed = policy.primary_score(entry)
    observed_delta = (
        observed - reference["primary"] if observed is not None else None
    )
    if entry.get("error") is not None:
        status = "execution_failed"
        interval_distance = None
    elif observed is None:
        status = "metric_missing"
        interval_distance = None
    elif observed_delta < expected_delta["min"]:
        status = "below_range"
        interval_distance = expected_delta["min"] - observed_delta
    elif observed_delta > expected_delta["max"]:
        status = "above_range"
        interval_distance = observed_delta - expected_delta["max"]
    else:
        status = "within_range"
        interval_distance = 0.0
    prior_best = policy.best_eligible(prior_attempts)
    prior_best_score = policy.primary_score(prior_best) if prior_best else None
    return {
        "reference": reference,
        "expected_delta": expected_delta,
        "expected_primary": expected_primary,
        "observed_primary": observed,
        "observed_delta": observed_delta,
        "status": status,
        "interval_distance": interval_distance,
        "material_improvement_over_reference": (
            observed_delta is not None and observed_delta > policy.EPSILON
        ),
        "new_validation_best": bool(
            policy.final_eligible(entry)
            and observed is not None
            and (prior_best_score is None or observed > prior_best_score)
        ),
    }


def _prediction_assessments(attempts: list[dict]) -> list[dict]:
    assessments = []
    prior: list[dict] = []
    for entry in attempts:
        assessments.append(_prediction_assessment(entry, prior))
        prior.append(entry)
    return assessments


def _prediction_calibration_summary(attempts: list[dict]) -> dict:
    global_counts = {status: 0 for status in PREDICTION_STATUSES}
    by_family: dict[str, dict[str, int]] = {}
    for entry, assessment in zip(attempts, _prediction_assessments(attempts)):
        status = assessment["status"]
        global_counts[status] += 1
        family_id = (entry.get("card") or {}).get("family_id")
        if isinstance(family_id, str):
            counts = by_family.setdefault(
                family_id, {name: 0 for name in PREDICTION_STATUSES}
            )
            counts[status] += 1
    return {"global": global_counts, "by_family": by_family}


def _compact_prior(entries: list[dict]) -> list[dict]:
    attempts = policy.official_iterations(entries)
    assessments = _prediction_assessments(attempts)
    return [
        {
            "entry_id": entry.get("entry_id"),
            "iteration": entry.get("iteration"),
            "solution": {
                "path": (entry.get("solution") or {}).get("path"),
                "sha256": (entry.get("solution") or {}).get("sha256"),
                "normalized_ast_sha256": (
                    entry.get("solution") or {}
                ).get("normalized_ast_sha256"),
            },
            "card": {
                key: (entry.get("card") or {}).get(key)
                for key in (
                    "family_id", "card_type", "mechanism", "hypothesis",
                    "change_summary", "falsifier", "why_now",
                    "expected_primary_delta", "research_basis", "family_extension",
                )
            },
            "resolved_research_basis": entry.get("resolved_research_basis"),
            "family_registration": entry.get("family_registration"),
            "valid_metrics": entry.get("valid_metrics"),
            "error": entry.get("error"),
            "prediction_assessment": assessment,
            "preflight": {
                "review_id": (entry.get("preflight_review") or {}).get("review_id"),
                "request_id": (entry.get("preflight_review") or {}).get("request_id"),
                "verdict": (entry.get("preflight_review") or {}).get("verdict"),
                "findings": (entry.get("preflight_review") or {}).get("findings"),
            },
        }
        for entry, assessment in zip(attempts, assessments)
    ]


def _review_request_id(kind: str, semantics: dict) -> str:
    return policy.canonical_sha256({
        "review_protocol": "track2.no-tools.consensus.v1",
        "kind": kind,
        "semantics": semantics,
    })


def _conclusive_review(entries: list[dict], request_id: str) -> dict | None:
    matches = [
        row for row in entries
        if row.get("type") == "preflight_review"
        and row.get("request_id") == request_id
        and isinstance(row.get("review"), dict)
    ]
    if len(matches) > 1:
        raise ControllerError("canonical review request has multiple verdicts")
    return matches[0] if matches else None


def _conclusive_stage_reviews(
    entries: list[dict], *, scope: str, iteration: int | None = None
) -> list[dict]:
    return [
        row for row in entries
        if row.get("type") == "preflight_review"
        and row.get("scope") == scope
        and isinstance(row.get("review"), dict)
        and (iteration is None or row.get("iteration") == iteration)
    ]


def _require_review_budget(
    entries: list[dict], *, scope: str, iteration: int | None = None
) -> None:
    used = len(_conclusive_stage_reviews(
        entries, scope=scope, iteration=iteration
    ))
    if used >= MAX_CONCLUSIVE_REVIEWS_PER_STAGE:
        label = scope if iteration is None else f"{scope} iteration {iteration}"
        raise ControllerError(
            f"{label} exhausted its {MAX_CONCLUSIVE_REVIEWS_PER_STAGE} "
            "conclusive semantic reviews; transport failures do not count"
        )
    failed = len([
        row for row in entries
        if row.get("type") == "preflight_review"
        and row.get("scope") == scope
        and row.get("review") is None
        and isinstance(row.get("error"), str)
        and (iteration is None or row.get("iteration") == iteration)
    ])
    if failed >= MAX_FAILED_REVIEWS_PER_STAGE:
        label = scope if iteration is None else f"{scope} iteration {iteration}"
        raise ControllerError(
            f"{label} exhausted its {MAX_FAILED_REVIEWS_PER_STAGE} failed-review "
            "retry budget; owner inspection is required"
        )


def _review_record(
    scope: str, request_id: str, review: dict | None, error: str | None,
    **fields: Any,
) -> dict:
    return base_entry(
        "preflight_review", scope=scope, request_id=request_id,
        review=review, error=error,
        accepted=bool(review and review.get("accepted")), **fields,
    )


def _candidate_fingerprint(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ControllerError(f"candidate is not parseable: {exc}") from exc
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return sha256_bytes(normalized.encode("utf-8"))


def _rejected_reviews(
    entries: list[dict], *, scope: str, iteration: int | None = None,
    candidate_sha256: str | None = None,
) -> list[dict]:
    result = []
    for row in entries:
        review = row.get("review")
        if (
            row.get("type") != "preflight_review"
            or row.get("scope") != scope
            or not isinstance(review, dict)
            or review.get("verdict") != "REJECT"
        ):
            continue
        if iteration is not None and row.get("iteration") != iteration:
            continue
        if candidate_sha256 is not None and row.get("candidate_sha256") != candidate_sha256:
            continue
        result.append({
            "review_id": review.get("review_id"),
            "request_id": row.get("request_id"),
            "candidate_sha256": row.get("candidate_sha256"),
            "artifact_sha256": row.get("artifact_sha256"),
            "findings": review.get("findings"),
            "summary": review.get("summary"),
        })
    return result


def _final_cache_path(auth: Any) -> Path:
    return auth.state_dir / "final-result.json"


def _write_final_cache(auth: Any, pending: dict, final: dict) -> None:
    path = _final_cache_path(auth)
    payload = json.dumps(
        {"pending_entry_id": pending["entry_id"], "final": final},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ControllerError("external final result cache already exists") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ControllerError("final result cache write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(auth.state_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_final_cache(auth: Any, pending: dict) -> dict | None:
    path = _final_cache_path(auth)
    if not path.exists():
        return None
    if path.is_symlink():
        raise ControllerError("external final result cache is a symlink")
    metadata = path.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size > 4 * 1024 * 1024
    ):
        raise ControllerError("external final result cache is not a private regular file")
    value = strict_json_bytes(path.read_bytes(), "external final result cache")
    if set(value) != {"pending_entry_id", "final"}:
        raise ControllerError("external final result cache has the wrong shape")
    if value.get("pending_entry_id") != pending.get("entry_id"):
        raise ControllerError("external final result cache belongs to another pending marker")
    final = value.get("final")
    if not isinstance(final, dict) or final.get("type") != "final":
        raise ControllerError("external final result cache has no final row")
    return final


def cmd_start_run(args) -> int:
    with controller_lock():
        require_standalone_repository()
        raw_entries = read_journal()
        if policy.first_run_start(raw_entries) is not None and not authority_state_dir().exists():
            raise ControllerError("run_start exists but its external authority is missing")
        auth = open_authority(create=True, reconcile=True)
        entries = read_journal()
        validate_ledger(entries)
        if policy.first_run_start(entries) is not None:
            raise ControllerError("run_start already exists and can never be reset")
        manifest = verify_manifest()
        components = current_component_hashes(require_committed=True)
        snapshot = open_input_snapshot(auth, create=True, manifest=manifest)
        portfolio_path = Path(args.portfolio)
        rel, portfolio_bytes = committed_bytes(
            portfolio_path, ROOT / "Project" / "research", ".json",
            maximum_bytes=policy.MAX_PORTFOLIO_SOURCE_BYTES,
        )
        portfolio = strict_json_bytes(portfolio_bytes, "research portfolio")
        policy.validate_portfolio(portfolio)
        bank = open_research_bank()
        resolved_portfolio = _resolve_portfolio_research(bank, portfolio)
        rejected = _rejected_reviews(entries, scope="portfolio")
        correction_id = portfolio.get("corrects_review_id")
        if rejected and correction_id not in {row["review_id"] for row in rejected}:
            raise ControllerError(
                "a revised portfolio must cite the prior rejected review_id in "
                "corrects_review_id"
            )
        semantics = {
            "policy_id": policy.POLICY_ID,
            "portfolio_canonical_sha256": policy.canonical_sha256(portfolio),
            "trusted_components": components,
            "input_snapshot_sha256": snapshot.canonical_sha256,
            "research_bank_snapshot_sha256": bank.snapshot_sha256,
            "resolved_portfolio_research_sha256": policy.canonical_sha256(
                resolved_portfolio
            ),
        }
        request_id = _review_request_id("portfolio", semantics)
        prior = _conclusive_review(entries, request_id)
        if prior is not None:
            raise ControllerError(
                f"this exact portfolio already has sticky verdict "
                f"{prior['review']['verdict']} ({prior['review']['review_id']})"
            )
        _require_review_budget(entries, scope="portfolio")
        packet = {
            "request_id": request_id,
            "kind": "portfolio",
            "policy_id": policy.POLICY_ID,
            "organizer_stop_rule": {
                "epsilon": policy.EPSILON,
                "consecutive_iterations": policy.N_CONVERGE,
                "iteration_cap": policy.ITERATION_CAP,
                "wall_clock_seconds": policy.WALL_CEILING_S,
                "selection": "validation-best eligible checkpoint at first terminal point",
            },
            "portfolio_path": rel,
            "portfolio_sha256": sha256_bytes(portfolio_bytes),
            "portfolio": portfolio,
            "resolved_portfolio_research": resolved_portfolio,
            "research_bank": {
                "snapshot_sha256": bank.snapshot_sha256,
                "descriptor": bank.descriptor,
                "claim_count": len(bank.known_claims),
                "known_topics": list(bank.known_topics),
            },
            "trusted_components": components,
            "input_snapshot_sha256": snapshot.canonical_sha256,
            "prior_rejections": rejected,
            "existing_rows_are_setup_only": len([
                entry for entry in entries if entry.get("type") == "iteration"
            ]),
        }
        try:
            review = preflight_review.run_review(
                root=ROOT, kind="portfolio", request_id=request_id, packet=packet,
                timeout_seconds=args.review_timeout,
                cache_dir=auth.state_dir / "review-cache",
            )
            error = None
        except preflight_review.ReviewError as exc:
            review, error = None, f"{type(exc).__name__}: {exc}"
        review_row = _review_record(
            "portfolio", request_id, review, error, artifact_path=rel,
            artifact_sha256=sha256_bytes(portfolio_bytes),
        )
        if error or not review or not review["accepted"]:
            append_batch(auth, entries, [review_row])
            raise ControllerError(error or f"portfolio reviewer returned {review['verdict']}")

        # Recheck after the external read-only review, then start the clock.
        verify_manifest()
        snapshot.verify()
        bank.verify()
        if components != current_component_hashes(require_committed=True):
            raise ControllerError("trusted components changed during portfolio review")
        if portfolio_path.resolve().read_bytes() != portfolio_bytes:
            raise ControllerError("portfolio changed during its review")
        # Runtime inventory and the real namespace probe happen before the
        # official six-hour clock starts. Every later attempt must match this
        # exact interpreter/stdlib/NumPy/bubblewrap root.
        sandbox_capability = sandbox.check_capability()
        run_id = uuid.uuid4().hex
        started_epoch = time.time()
        start = base_entry(
            "run_start",
            policy_id=policy.POLICY_ID,
            run_id=run_id,
            benchmark=policy.BENCHMARK,
            started_epoch=started_epoch,
            deadline_epoch=started_epoch + policy.WALL_CEILING_S,
            git_revision=git_revision(),
            trusted_components=components,
            portfolio={
                "path": rel,
                "sha256": sha256_bytes(portfolio_bytes),
                "canonical_sha256": policy.canonical_sha256(portfolio),
                "review_id": review["review_id"],
                "request_id": request_id,
                "family_ids": [family["family_id"] for family in portfolio["families"]],
                "opening_order": portfolio["opening_order"],
                "resolved_research_sha256": policy.canonical_sha256(
                    resolved_portfolio
                ),
            },
            research_bank={
                "snapshot_sha256": bank.snapshot_sha256,
                "descriptor": bank.descriptor,
                "claim_count": len(bank.known_claims),
                "known_topics": list(bank.known_topics),
            },
            input_snapshot={
                "manifest_sha256": snapshot.canonical_sha256,
                "candidate_data_manifest_sha256": policy.canonical_sha256(
                    snapshot.candidate_sha256()
                ),
            },
            sandbox=sandbox_capability,
        )
        sandbox.verify_runtime(sandbox_capability["runtime_manifest_sha256"])
        written = append_batch(auth, entries, [review_row, start])
        print(json.dumps({
            "run_id": written[-1]["run_id"],
            "started_epoch": written[-1]["started_epoch"],
            "portfolio_review": review["verdict"],
            "message": "official clock started; terminal conditions have no override",
        }, indent=2))
        return 0


def cmd_run(args) -> int:
    with controller_lock():
        require_standalone_repository()
        auth = open_authority(create=False, reconcile=True)
        entries = read_journal()
        validate_ledger(entries, allow_open_attempt=True)
        start = policy.first_run_start(entries)
        if start is None:
            raise ControllerError("official run has not started; setup scoring is disabled")
        runtime_manifest_sha256 = _run_start_runtime_manifest(start)
        sandbox.verify_runtime(runtime_manifest_sha256)
        verify_manifest()
        verify_frozen_components(start)
        snapshot = open_input_snapshot(auth, create=False)
        if snapshot.canonical_sha256 != start["input_snapshot"]["manifest_sha256"]:
            raise ControllerError("external input snapshot differs from run_start")
        bank = open_research_bank(start.get("research_bank"))
        entries = recover_open_attempt(auth, entries)
        validate_ledger(entries)
        entries, terminal = latch_terminal_if_due(auth, entries, time.time())
        if terminal is not None:
            raise ControllerError(
                f"run is irreversibly terminal ({terminal['reason']}) at iteration "
                f"{terminal['terminal_iteration']}"
            )

        portfolio = _portfolio_from_start(start)
        solution_rel, candidate_bytes = committed_bytes(
            Path(args.solution), ROOT / "Project" / "solutions", ".py",
            maximum_bytes=512 * 1024,
        )
        card_rel, card_bytes = committed_bytes(
            Path(args.card), ROOT / "Project" / "research" / "attempts", ".json",
            maximum_bytes=256 * 1024,
        )
        candidate_sha = sha256_bytes(candidate_bytes)
        card_sha = sha256_bytes(card_bytes)
        card = strict_json_bytes(card_bytes, "attempt card")
        attempts = policy.official_iterations(entries)
        expected_iteration = len(attempts) + 1
        registrations = _registered_families(entries)
        policy.validate_attempt_card(
            card,
            run_id=start["run_id"],
            expected_iteration=expected_iteration,
            portfolio=portfolio,
            candidate_path=solution_rel,
            candidate_sha256=candidate_sha,
            registered_families={
                family_id: value["extension"]
                for family_id, value in registrations.items()
            },
        )
        resolved_research = _resolve_attempt_research(
            bank, card, portfolio, registrations
        )
        seed_family_ids = {family["family_id"] for family in portfolio["families"]}
        if card["family_id"] in seed_family_ids:
            family_registration = None
        elif card["family_id"] in registrations:
            family_registration = registrations[card["family_id"]]
        else:
            family_registration = {
                "family_id": card["family_id"],
                "extension": card["family_extension"],
                "extension_sha256": policy.canonical_sha256(
                    card["family_extension"]
                ),
                "first_iteration": expected_iteration,
            }
        expected_prior = [entry["entry_id"] for entry in attempts]
        if card["prior_outcomes_considered"] != expected_prior:
            raise ControllerError(
                "prior_outcomes_considered must list every prior official outcome "
                "once, in chronological order"
            )
        if expected_iteration <= 4:
            expected_family = portfolio["opening_order"][expected_iteration - 1]
            if card.get("family_id") != expected_family:
                raise ControllerError(
                    "the first four official attempts must follow the portfolio's "
                    f"frozen opening_order (expected {expected_family})"
                )
        source = candidate_bytes.decode("utf-8", errors="strict")
        normalized_ast_sha = _candidate_fingerprint(source)
        prior_candidate_rejects = _rejected_reviews(
            entries, scope="attempt", iteration=expected_iteration,
            candidate_sha256=candidate_sha,
        )
        correction_id = card.get("corrects_review_id")
        if (
            prior_candidate_rejects
            and correction_id not in {
                row["review_id"] for row in prior_candidate_rejects
            }
        ):
            raise ControllerError(
                "a previously rejected candidate needs a correction card naming the "
                "rejected review_id"
            )
        semantics = {
            "policy_id": policy.POLICY_ID,
            "run_id": start["run_id"],
            "iteration": expected_iteration,
            "candidate_sha256": candidate_sha,
            "candidate_ast_sha256": normalized_ast_sha,
            "card_canonical_sha256": policy.canonical_sha256(card),
            "prior_outcome_ids": expected_prior,
            "research_bank_snapshot_sha256": bank.snapshot_sha256,
            "resolved_research_sha256": policy.canonical_sha256(resolved_research),
            "family_registration": family_registration,
        }
        request_id = _review_request_id("attempt", semantics)
        prior_verdict = _conclusive_review(entries, request_id)
        if prior_verdict is not None:
            raise ControllerError(
                f"this exact attempt already has sticky verdict "
                f"{prior_verdict['review']['verdict']} "
                f"({prior_verdict['review']['review_id']})"
            )
        _require_review_budget(
            entries, scope="attempt", iteration=expected_iteration
        )
        static_findings = scan_candidate(source)
        if static_findings:
            rejected = _review_record(
                "attempt_static", request_id, None,
                "deterministic source policy rejection",
                run_id=start["run_id"], iteration=expected_iteration,
                artifact_path=card_rel, card_sha256=card_sha,
                candidate_path=solution_rel, candidate_sha256=candidate_sha,
                findings=static_findings,
            )
            append_batch(auth, entries, [rejected])
            raise ControllerError(f"candidate source rejected: {static_findings}")

        remaining = policy.WALL_CEILING_S - policy.elapsed_seconds(entries, time.time())
        if remaining <= 0:
            entries, terminal = latch_terminal_if_due(auth, entries, time.time())
            raise ControllerError(f"run is terminal ({terminal['reason']})")
        packet = {
            "request_id": request_id,
            "kind": "attempt",
            "policy_id": policy.POLICY_ID,
            "run_id": start["run_id"],
            "next_iteration": expected_iteration,
            "remaining_wall_seconds_before_review": remaining,
            "portfolio": portfolio,
            "card_path": card_rel,
            "card_sha256": card_sha,
            "card": card,
            "resolved_research_basis": resolved_research,
            "family_registration": family_registration,
            "candidate_path": solution_rel,
            "candidate_sha256": candidate_sha,
            "candidate_normalized_ast_sha256": normalized_ast_sha,
            "candidate_source": source,
            "static_findings": static_findings,
            "prior_official_outcomes": _compact_prior(entries),
            "prior_rejections_for_same_candidate": prior_candidate_rejects,
        }
        try:
            review = preflight_review.run_review(
                root=ROOT, kind="attempt", request_id=request_id, packet=packet,
                timeout_seconds=min(float(args.review_timeout), remaining),
                cache_dir=auth.state_dir / "review-cache",
            )
            error = None
        except preflight_review.ReviewError as exc:
            review, error = None, f"{type(exc).__name__}: {exc}"
        review_row = _review_record(
            "attempt", request_id, review, error, run_id=start["run_id"],
            iteration=expected_iteration, artifact_path=card_rel,
            card_sha256=card_sha, candidate_path=solution_rel,
            candidate_sha256=candidate_sha,
        )
        if error or not review or not review["accepted"]:
            written = append_batch(auth, entries, [review_row])
            entries += written
            entries, _ = latch_terminal_if_due(auth, entries, time.time())
            raise ControllerError(error or f"attempt reviewer returned {review['verdict']}")

        verify_manifest()
        verify_frozen_components(start)
        bank.verify()
        sandbox.verify_runtime(runtime_manifest_sha256)
        if Path(args.solution).resolve().read_bytes() != candidate_bytes:
            raise ControllerError("candidate changed during independent review")
        if Path(args.card).resolve().read_bytes() != card_bytes:
            raise ControllerError("attempt card changed during independent review")
        remaining = policy.WALL_CEILING_S - policy.elapsed_seconds(entries, time.time())
        if remaining <= 0:
            review_row["expired_before_use"] = True
            terminal_epoch = time.time()
            prospective = entries + [review_row]
            terminal_row = _make_terminal(prospective, terminal_epoch)
            written = append_batch(auth, entries, [review_row, terminal_row])
            entries += written
            raise ControllerError("six-hour ceiling elapsed during preflight review")

        attempt_id = uuid.uuid4().hex
        started_epoch = time.time()
        if started_epoch >= policy.run_deadline(start):
            review_row["expired_before_use"] = True
            prospective = entries + [review_row]
            terminal_row = _make_terminal(prospective, started_epoch)
            written = append_batch(auth, entries, [review_row, terminal_row])
            entries += written
            raise ControllerError("six-hour ceiling elapsed before attempt start")
        solution_record = {
            "path": solution_rel,
            "sha256": candidate_sha,
            "normalized_ast_sha256": normalized_ast_sha,
            "source": source,
        }
        started = base_entry(
            "attempt_started",
            policy_id=policy.POLICY_ID,
            run_id=start["run_id"],
            attempt_id=attempt_id,
            iteration=expected_iteration,
            started_epoch=started_epoch,
            solution=solution_record,
            card={**card, "path": card_rel, "sha256": card_sha},
            resolved_research_basis=resolved_research,
            family_registration=family_registration,
            preflight_review=review,
        )
        written = append_batch(auth, entries, [review_row, started])
        entries += written

        entry_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:10]}"
        valid_metrics = None
        seal = None
        sandbox_record = None
        hypothesis = card["hypothesis"]
        error_text = None
        effective_timeout = min(
            float(args.timeout),
            policy.WALL_CEILING_S - policy.elapsed_seconds(entries, time.time()),
        )
        try:
            trusted = DevelopmentTrusted(snapshot)
            result = sandbox.run_candidate(
                root=ROOT,
                candidate_name=Path(solution_rel).name,
                candidate_bytes=candidate_bytes,
                candidate_sha256=candidate_sha,
                organizer_snapshot=snapshot.kit_dir,
                organizer_sha256=snapshot.kit_sha256(),
                worker_bytes=(HARNESS_DIR / "candidate_worker.py").read_bytes(),
                worker_sha256=start["trusted_components"][
                    "Project/harness/candidate_worker.py"
                ],
                sanitized_snapshot=snapshot.candidate_dir,
                sanitized_sha256=snapshot.candidate_sha256(),
                timeout_seconds=effective_timeout,
                expected_valid_rows=len(trusted.splits["valid"]),
                expected_test_rows=len(trusted.splits["test"]),
                expected_runtime_manifest_sha256=runtime_manifest_sha256,
            )
            sandbox_record = result["sandbox"]
            if policy.elapsed_seconds(entries, time.time()) >= policy.WALL_CEILING_S:
                raise TimeoutError("six-hour ceiling reached before trusted scoring")
            verify_manifest()
            verify_frozen_components(start)
            bank.verify()
            valid_metrics = trusted.score_valid(result["valid"])
            seal = trusted.seal_test(entry_id, result["test"])
            sandbox.verify_runtime(runtime_manifest_sha256)
        except BaseException as exc:  # failure evidence must survive
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            error_text = f"{type(exc).__name__}: {exc}"[:12_000]
        completed_epoch = time.time()
        eligible = (
            error_text is None
            and completed_epoch <= policy.run_deadline(start)
        )
        if not eligible and error_text is None:
            error_text = "WallClockExceeded: result completed after the six-hour ceiling"
        outcome = {
            **base_entry("iteration"),
            "entry_id": entry_id,
            "policy_id": policy.POLICY_ID,
            "run_id": start["run_id"],
            "attempt_id": attempt_id,
            "iteration": expected_iteration,
            "started_epoch": started_epoch,
            "completed_epoch": completed_epoch,
            "wall_seconds": max(0.0, completed_epoch - started_epoch),
            "effective_timeout_seconds": effective_timeout,
            "solution": solution_record,
            "card": {**card, "path": card_rel, "sha256": card_sha},
            "resolved_research_basis": resolved_research,
            "family_registration": family_registration,
            "preflight_review": review,
            "hypothesis": hypothesis,
            "sandbox": sandbox_record,
            "valid_metrics": valid_metrics,
            "sealed_test_scores": seal,
            "eligible_for_final": bool(eligible),
            "error": error_text,
            "git_revision": git_revision(),
        }
        appended = append_outcome_and_terminal(
            auth, entries, outcome, completed_epoch
        )
        state_entries = entries + appended
        terminal = policy.first_terminal(state_entries)
        print(json.dumps({
            "iteration": expected_iteration,
            "entry_id": entry_id,
            "valid_primary": policy.primary_score(outcome),
            "prediction_assessment": _prediction_assessment(outcome, attempts),
            "error": error_text,
            "eligible_for_final": eligible,
            "terminal": ({
                "reason": terminal["reason"],
                "terminal_iteration": terminal["terminal_iteration"],
                "best_entry_id": terminal["eligible_best_entry_id"],
            } if terminal else None),
        }, indent=2))
        return 0 if error_text is None else 2


def cmd_final(args) -> int:
    with controller_lock():
        require_standalone_repository()
        auth = open_authority(create=False, reconcile=True)
        entries = read_journal()
        validate_ledger(entries, allow_open_attempt=True)
        start = policy.first_run_start(entries)
        if start is None:
            raise ControllerError("official run has not started")
        runtime_manifest_sha256 = _run_start_runtime_manifest(start)
        sandbox.verify_runtime(runtime_manifest_sha256)
        verify_manifest()
        verify_frozen_components(start)
        snapshot = open_input_snapshot(auth, create=False)
        if snapshot.canonical_sha256 != start["input_snapshot"]["manifest_sha256"]:
            raise ControllerError("external input snapshot differs from run_start")
        bank = open_research_bank(start.get("research_bank"))
        entries = recover_open_attempt(auth, entries)
        validate_ledger(entries)
        entries, terminal = latch_terminal_if_due(auth, entries, time.time())
        if terminal is None:
            raise ControllerError("run is still ACTIVE; final is legal only after termination")
        validate_ledger(entries)
        existing_final = next(
            (entry for entry in entries if entry.get("type") == "final"), None
        )
        if existing_final is not None:
            raise ControllerError("hidden test final already completed once")
        existing_pending = next(
            (entry for entry in entries if entry.get("type") == "final_pending"), None
        )
        if existing_pending is not None:
            cached = _read_final_cache(auth, existing_pending)
            if cached is None:
                raise ControllerError(
                    "final_pending exists without a durable result; hidden scoring will "
                    "not be repeated"
                )
            if (
                cached.get("designated_entry") != existing_pending.get("designated_entry")
                or cached.get("final_review_id")
                != (existing_pending.get("review") or {}).get("review_id")
                or cached.get("submission_csv_sha256") != sha256_file(FINAL_CSV)
            ):
                raise ControllerError("durable final result does not match pending state")
            append_batch(auth, entries, [cached])
            print(json.dumps({**cached, "recovered_without_rescoring": True}, indent=2))
            return 0
        if FINAL_CSV.exists():
            raise ControllerError(
                "submission CSV already exists before the once-only final transition"
            )
        target_id = terminal.get("eligible_best_entry_id")
        if not target_id:
            raise ControllerError("terminal snapshot contains no eligible checkpoint")
        target = next(
            entry for entry in policy.official_iterations(entries)
            if entry.get("entry_id") == target_id
        )
        if not policy.final_eligible(target):
            raise ControllerError("terminal-selected checkpoint is not final-eligible")
        semantics = {
            "policy_id": policy.POLICY_ID,
            "run_id": start["run_id"],
            "terminal_entry_id": terminal.get("entry_id"),
            "official_prefix_sha256": terminal.get("official_prefix_sha256"),
            "designated_entry": target_id,
            "solution_sha256": target["solution"]["sha256"],
            "sealed_sha256": target["sealed_test_scores"]["sha256"],
        }
        request_id = _review_request_id("final", semantics)
        prior_verdict = _conclusive_review(entries, request_id)
        if prior_verdict is not None:
            raise ControllerError(
                f"the exact frozen final already has sticky verdict "
                f"{prior_verdict['review']['verdict']} "
                f"({prior_verdict['review']['review_id']})"
            )
        _require_review_budget(entries, scope="final")
        packet = {
            "request_id": request_id,
            "kind": "final",
            "policy_id": policy.POLICY_ID,
            "run_id": start["run_id"],
            "terminal": terminal,
            "selected_entry": target,
            "official_outcomes": _compact_prior(entries),
            "journal_sha256": sha256_file(JOURNAL_PATH),
        }
        try:
            review = preflight_review.run_review(
                root=ROOT, kind="final", request_id=request_id, packet=packet,
                timeout_seconds=args.review_timeout,
                cache_dir=auth.state_dir / "review-cache",
            )
            error = None
        except preflight_review.ReviewError as exc:
            review, error = None, f"{type(exc).__name__}: {exc}"
        review_row = _review_record(
            "final", request_id, review, error, run_id=start["run_id"],
            designated_entry=target_id,
        )
        if error or not review or not review["accepted"]:
            append_batch(auth, entries, [review_row])
            raise ControllerError(error or f"final reviewer returned {review['verdict']}")
        verify_manifest()
        verify_frozen_components(start)
        bank.verify()
        sandbox.verify_runtime(runtime_manifest_sha256)
        pending = base_entry(
            "final_pending", policy_id=policy.POLICY_ID, run_id=start["run_id"],
            designated_entry=target_id, terminal_entry_id=terminal["entry_id"],
            review=review,
        )
        written = append_batch(auth, entries, [review_row, pending])
        entries += written
        pending = written[-1]

        trusted = FinalTrusted(snapshot)
        scores = trusted.load_sealed(target)
        metrics = trusted.write_check_score(scores)
        sandbox.verify_runtime(runtime_manifest_sha256)
        final = base_entry(
            "final", policy_id=policy.POLICY_ID, run_id=start["run_id"],
            designated_entry=target_id,
            designated_solution={
                "path": target["solution"]["path"],
                "sha256": target["solution"]["sha256"],
            },
            valid_metrics=target["valid_metrics"],
            test_metrics_from_submitted_csv=metrics,
            baseline_test_primary=BASELINE_TEST_PRIMARY,
            delta_over_baseline=metrics["primary"] - BASELINE_TEST_PRIMARY,
            submission_csv=FINAL_CSV.relative_to(ROOT).as_posix(),
            submission_csv_sha256=sha256_file(FINAL_CSV),
            terminal_entry_id=terminal["entry_id"],
            final_review_id=review["review_id"],
        )
        _write_final_cache(auth, pending, final)
        append_batch(auth, entries, [final])
        print(json.dumps(final, indent=2))
        return 0


MAX_MODEL_LOG_BYTES = 96 * 1024
MAX_PRIVATE_ADMISSION_STATE_BYTES = 1536 * 1024


def _state_value(*, include_outcomes: bool) -> dict:
    if authority_state_dir().exists():
        open_authority(create=False, reconcile=False)
    entries = read_journal()
    validate_ledger(entries, allow_open_attempt=True)
    start = policy.first_run_start(entries)
    attempts = policy.official_iterations(entries)
    prediction_assessments = _prediction_assessments(attempts)
    terminal = policy.first_terminal(entries)
    best = policy.best_eligible(attempts)
    due = policy.triggered_reasons(entries, time.time()) if start else []
    portfolio = start.get("portfolio", {}) if start else {}
    portfolio_value = _portfolio_from_start(start) if start else None
    registrations = _registered_families(entries) if start else {}
    next_iteration = len(attempts) + 1
    attempt_reviews_used = len(_conclusive_stage_reviews(
        entries, scope="attempt", iteration=next_iteration
    )) if start and not terminal else 0
    attempt_review_failures = len([
        row for row in entries
        if row.get("type") == "preflight_review"
        and row.get("scope") == "attempt"
        and row.get("iteration") == next_iteration
        and row.get("review") is None
        and isinstance(row.get("error"), str)
    ]) if start and not terminal else 0
    return {
        "policy_id": policy.POLICY_ID,
        "official_run_started": start is not None,
        "run_id": start.get("run_id") if start else None,
        "run_start_git_revision": start.get("git_revision") if start else None,
        "state": (
            "TERMINAL" if terminal else
            ("TERMINAL_DUE" if due else ("ACTIVE" if start else "NOT_STARTED"))
        ),
        "official_iterations": len(attempts),
        "next_iteration": next_iteration if start and not terminal else None,
        "prior_outcomes_considered": [
            entry["entry_id"] for entry in attempts
        ],
        "next_prediction_reference": (
            _prediction_reference(attempts) if start and not terminal else None
        ),
        "prediction_calibration": _prediction_calibration_summary(attempts),
        "portfolio_family_ids": portfolio.get("family_ids", []),
        "portfolio_opening_order": portfolio.get("opening_order", []),
        "portfolio": portfolio_value,
        "research_bank": start.get("research_bank") if start else None,
        "registered_family_extensions": (
            {
                family_id: registration["extension"]
                for family_id, registration in registrations.items()
            }
            if start else {}
        ),
        "registered_family_records": registrations,
        "official_outcomes": ([
            {
                "entry_id": entry["entry_id"],
                "iteration": entry["iteration"],
                "family_id": (entry.get("card") or {}).get("family_id"),
                "card_type": (entry.get("card") or {}).get("card_type"),
                "mechanism": (entry.get("card") or {}).get("mechanism"),
                "hypothesis": (entry.get("card") or {}).get("hypothesis"),
                "falsifier": (entry.get("card") or {}).get("falsifier"),
                "expected_primary_delta": (
                    entry.get("card") or {}
                ).get("expected_primary_delta"),
                "valid_metrics": entry.get("valid_metrics"),
                "prediction_assessment": assessment,
                "eligible_for_final": entry.get("eligible_for_final"),
                "error": entry.get("error"),
            }
            for entry, assessment in zip(attempts, prediction_assessments)
        ] if include_outcomes else []),
        "current_attempt_rejections": [
            {
                "review_id": (row.get("review") or {}).get("review_id"),
                "request_id": row.get("request_id"),
                "candidate_sha256": row.get("candidate_sha256"),
                "card_sha256": row.get("card_sha256"),
                "findings": (row.get("review") or {}).get("findings"),
                "summary": (row.get("review") or {}).get("summary"),
            }
            for row in _conclusive_stage_reviews(
                entries, scope="attempt", iteration=next_iteration
            )
            if isinstance(row.get("review"), dict)
        ],
        "attempt_review_budget_remaining": (
            max(0, MAX_CONCLUSIVE_REVIEWS_PER_STAGE - attempt_reviews_used)
            if start and not terminal else 0
        ),
        "attempt_review_failure_budget_remaining": (
            max(0, MAX_FAILED_REVIEWS_PER_STAGE - attempt_review_failures)
            if start and not terminal else 0
        ),
        "best_entry_id": best.get("entry_id") if best else None,
        "best_validation_primary": policy.primary_score(best) if best else None,
        "elapsed_seconds": policy.elapsed_seconds(entries, time.time()) if start else 0,
        "would_trigger_now": due,
        "terminal": terminal,
        "open_attempt": any(
            entry.get("type") == "attempt_started"
            and entry.get("attempt_id") not in {row.get("attempt_id") for row in attempts}
            for entry in entries
        ),
    }


def _clip_log_text(value: Any, maximum: int) -> Any:
    if not isinstance(value, str):
        return value
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= maximum:
        return value
    suffix = "…[truncated]"
    prefix = encoded[: maximum - len(suffix.encode("utf-8"))].decode(
        "utf-8", errors="ignore"
    )
    return prefix + suffix


def _compact_portfolio(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    families = []
    for family in value.get("families", []):
        if not isinstance(family, dict):
            continue
        families.append({
            key: (
                _clip_log_text(family.get(key), 384)
                if key in {
                    "mechanism", "causal_claim", "smallest_experiment",
                    "falsifier", "known_risks",
                }
                else family.get(key)
            )
            for key in (
                "family_id", "mechanism", "causal_claim", "smallest_experiment",
                "falsifier", "known_risks", "expected_primary_delta",
                "bank_topics", "research_basis",
            )
        })
    return {
        "schema_version": value.get("schema_version"),
        "benchmark": value.get("benchmark"),
        "selection_rubric": _clip_log_text(value.get("selection_rubric"), 1000),
        "opening_order": value.get("opening_order"),
        "families": families,
    }


def cmd_log(_args) -> int:
    state = _state_value(include_outcomes=True)
    state.pop("run_start_git_revision", None)
    exact_portfolio = state.pop("portfolio", None)
    state["portfolio_summary"] = _compact_portfolio(exact_portfolio)
    portfolio_copy = (
        json.dumps(
            exact_portfolio, sort_keys=True, indent=2, ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        if isinstance(exact_portfolio, dict) else None
    )
    state["frozen_portfolio"] = (
        {
            "path": "Project/research/portfolio.json",
            "canonical_sha256": policy.canonical_sha256(exact_portfolio),
            "file_sha256": sha256_bytes(portfolio_copy),
            "size": len(portfolio_copy),
        }
        if portfolio_copy is not None else None
    )
    state.pop("registered_family_records", None)
    state["registered_family_extensions"] = {
        family_id: {
            **extension,
            "mechanism_delta": _clip_log_text(
                extension.get("mechanism_delta"), 384
            ),
        }
        for family_id, extension in state["registered_family_extensions"].items()
        if isinstance(extension, dict)
    }
    for outcome in state["official_outcomes"]:
        outcome["mechanism"] = _clip_log_text(outcome.get("mechanism"), 384)
        outcome["hypothesis"] = _clip_log_text(outcome.get("hypothesis"), 384)
        outcome["falsifier"] = _clip_log_text(outcome.get("falsifier"), 256)
        outcome["error"] = _clip_log_text(outcome.get("error"), 512)
    for rejection in state["current_attempt_rejections"]:
        rejection["findings"] = _clip_log_text(
            json.dumps(
                rejection.get("findings"), ensure_ascii=False, allow_nan=False
            ),
            4000,
        )
        rejection["summary"] = _clip_log_text(rejection.get("summary"), 1000)
    payload = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_MODEL_LOG_BYTES:
        raise ControllerError("bounded model-facing state summary exceeded its invariant")
    print(payload)
    return 0


def cmd_admission_state(_args) -> int:
    """Private, bounded deterministic pre-admission input for the outer service."""

    state = _state_value(include_outcomes=False)
    payload = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > MAX_PRIVATE_ADMISSION_STATE_BYTES:
        raise ControllerError("private admission state exceeded its fixed size invariant")
    print(payload)
    return 0


def cmd_recover(_args) -> int:
    """Owner-only reconciliation after an interrupted consuming transition."""

    with controller_lock():
        require_standalone_repository()
        auth = open_authority(create=False, reconcile=True)
        entries = read_journal()
        validate_ledger(entries, allow_open_attempt=True)
        start = policy.first_run_start(entries)
        if start is None:
            raise ControllerError("official run has not started")
        verify_manifest()
        verify_frozen_components(start)
        snapshot = open_input_snapshot(auth, create=False)
        if snapshot.canonical_sha256 != start["input_snapshot"]["manifest_sha256"]:
            raise ControllerError("external input snapshot differs from run_start")
        open_research_bank(start.get("research_bank"))
        before = len(policy.official_iterations(entries, stop_at_terminal=False))
        recovered = recover_open_attempt(auth, entries)
        if recovered == entries:
            raise ControllerError("there is no open official attempt to recover")
        validate_ledger(recovered)
        recovered, terminal = latch_terminal_if_due(auth, recovered, time.time())
        validate_ledger(recovered)
        attempts = policy.official_iterations(recovered, stop_at_terminal=False)
        if len(attempts) != before + 1 or attempts[-1].get("error") is None:
            raise ControllerError("owner recovery did not create one failed outcome")
        print(json.dumps({
            "recovered": True,
            "iteration": attempts[-1]["iteration"],
            "entry_id": attempts[-1]["entry_id"],
            "eligible_for_final": False,
            "terminal": ({
                "reason": terminal["reason"],
                "terminal_iteration": terminal["terminal_iteration"],
                "best_entry_id": terminal["eligible_best_entry_id"],
            } if terminal else None),
        }, indent=2))
    return 0


def cmd_intervention(args) -> int:
    with controller_lock():
        require_standalone_repository()
        auth = open_authority(create=True, reconcile=True)
        entries = read_journal()
        validate_ledger(entries, allow_open_attempt=True)
        start = policy.first_run_start(entries)
        if start is not None:
            verify_manifest()
            verify_frozen_components(start)
            open_research_bank(start.get("research_bank"))
            entries = recover_open_attempt(auth, entries)
            validate_ledger(entries)
        description = args.describe.strip()
        if not 1 <= len(description) <= 4000:
            raise ControllerError("intervention description must be 1..4000 characters")
        append_batch(auth, entries, [base_entry(
            "intervention", run_id=start.get("run_id") if start else None,
            description=description,
        )])
    print("intervention recorded; it grants no controller authority")
    return 0


def cmd_sanitize(_args) -> int:
    with controller_lock():
        entries = read_journal()
        validate_ledger(entries)
        if policy.first_run_start(entries) is not None:
            raise ControllerError("sanitized data cannot be regenerated after run_start")
        verify_manifest(require_sanitized=False)
        SANITIZED_DIR.mkdir(parents=True, exist_ok=True)
        logs = {
            "log_standard_4_22_to_5_08_pure.csv",
            "log_random_4_22_to_5_08_pure.csv",
        }
        for source in sorted(RAW_DATA_DIR.iterdir()):
            destination = SANITIZED_DIR / source.name
            if source.name not in logs:
                shutil.copyfile(source, destination)
                continue
            with source.open(newline="") as src, destination.open("w", newline="") as dst:
                reader = csv.DictReader(src)
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    if int(row["date"]) >= TEST_DATE_START:
                        for column in FEEDBACK_COLUMNS:
                            if column in row:
                                row[column] = "0"
                    writer.writerow(row)
        verify_manifest()
    print("sanitized dataset regenerated and hash-verified")
    return 0


def cmd_check(_args) -> int:
    verify_manifest()
    components = current_component_hashes(require_committed=False)
    capabilities = sandbox.check_capability()
    print(json.dumps({
        "policy_id": policy.POLICY_ID,
        "integrity": "OK",
        "trusted_components": components,
        "sandbox": capabilities,
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 2 irreversible official-run controller")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify integrity and required OS isolation")
    start = sub.add_parser("start-run", help="review portfolio and start the one official clock")
    start.add_argument("--portfolio", required=True)
    start.add_argument("--review-timeout", type=float, default=1800)
    run = sub.add_parser("run", help="review and consume exactly one official attempt")
    run.add_argument("--solution", required=True)
    run.add_argument("--card", required=True)
    run.add_argument("--timeout", type=float, default=1800)
    run.add_argument("--review-timeout", type=float, default=1200)
    final = sub.add_parser("final", help="score the frozen validation-best artifact once")
    final.add_argument("--review-timeout", type=float, default=1200)
    sub.add_parser("log", help="read-only official state summary")
    sub.add_parser("_admission-state", help=argparse.SUPPRESS)
    sub.add_parser(
        "recover", help="owner-only fail-closed recovery of one interrupted attempt"
    )
    sub.add_parser("sanitize-data", help="regenerate sanitized data before run_start only")
    intervention = sub.add_parser("intervention", help="record human help; grants no authority")
    intervention.add_argument("--describe", required=True)
    args = parser.parse_args()
    for name in ("timeout", "review_timeout"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if not math.isfinite(float(value)) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    require_isolated_interpreter()
    return {
        "check": cmd_check,
        "start-run": cmd_start_run,
        "run": cmd_run,
        "final": cmd_final,
        "log": cmd_log,
        "_admission-state": cmd_admission_state,
        "recover": cmd_recover,
        "sanitize-data": cmd_sanitize,
        "intervention": cmd_intervention,
    }[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ControllerError,
        authority.AuthorityError,
        input_snapshot.SnapshotError,
        policy.PolicyError,
        preflight_review.ReviewError,
        research_bank.BankError,
        sandbox.SandboxError,
    ) as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
