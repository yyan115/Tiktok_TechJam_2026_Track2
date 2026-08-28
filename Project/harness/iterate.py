#!/usr/bin/env python3
"""Iteration harness ("the lab bench"), v0.2.0-unfrozen.

v0.2.0 is a rebuild after the first codex audit (verdict NO, 8 findings):

  F1  MECHANICAL leak-guard: the harness loads the data itself and hands
      solutions a restricted copy — test rows have their label zeroed out.
      Solutions never receive test labels. (Development may use train fully
      and validation labels for early stopping, per the competition rules;
      only the test labels are off-limits.)
  F2  Trusted-callable snapshots + evaluator tamper probe: the organizers'
      evaluate() and data loader are imported and probed BEFORE candidate
      code executes; the probe is re-checked after the candidate runs and
      any drift aborts the run. Same-process residual documented below.
  F3  Test-exactly-once is enforced: `final` refuses when a final entry
      already exists (an override flag exists but is itself journaled), and
      a `final_pending` marker is journaled BEFORE test scoring so even a
      crash leaves evidence that test was consumed. `run` refuses after a
      final exists (journaled override for explicitly-labeled post-final work).
  F4  The scored final IS the measured artifact: every run SEALS the
      solution's test-split scores (its own outputs, no labels) to
      results/sealed/<entry_id>.npy with the sha journaled. `final --entry`
      scores those sealed bytes — no retraining, no stochastic drift.
  F5  Convergence and budgets are enforced, not just recorded: `run` refuses
      once converged (organizers' rule, window over successful iterations),
      past the 50-iteration cap, or past the 6h wall ceiling — each override
      flag is journaled. Malformed journal lines warn loudly.
  F6  Journal completeness: harness sha, git rev + dirty flag, dataset
      verification against manifest hashes, full solution source embedded,
      sealed-scores sha; solution load and data load are INSIDE the
      journaled try so import/data failures still produce entries.
  F7  `final` validates the written CSV with the organizers' own checker and
      scores are finiteness-checked before any evaluation.

Same-process residual (documented deliberately, as in the sister Track 3
repo): solution code runs in this process; a truly adversarial solution
could attack channels the probes don't watch. Trust model is cooperative —
guards against mistakes, not malice; solution sources are short, journaled
verbatim, and reviewed at audit checkpoints.

Solution contract (a .py file in Project/solutions/):
  HYPOTHESIS = "one line: what this tries and why"           (required)
  def run(splits) -> {'valid': scores, 'test': scores}       (required)
    - splits = {'train': rows, 'valid': rows, 'test': rows}, the organizers'
      row-tuple format, EXCEPT test rows carry label 0 (stripped).
    - scores are row-aligned real numbers; only relative order matters.

This file is part of the trusted lab bench. After its freeze it must not be
modified without user approval (see Project/PLAN.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path

HARNESS_VERSION = "0.5.0-unfrozen"

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "kuairand-starter-kit"
DATA_DIR = KIT / "KuaiRand-Pure" / "data"
MANIFEST_PATH = ROOT / "Project" / "manifest.json"
RESULTS_DIR = ROOT / "Project" / "results"


def _default_ledger() -> bool:
    return JOURNAL_PATH == RESULTS_DIR / "JOURNAL.jsonl"


def sealed_dir() -> Path:
    # Namespaced by LEDGER IDENTITY, not just directory, so two scratch
    # ledgers side by side (or one placed next to production) can never share
    # or clobber artifacts (codex round 4). Production keeps its stable path.
    if _default_ledger():
        return RESULTS_DIR / "sealed"
    return JOURNAL_PATH.parent / (JOURNAL_PATH.stem + "_sealed")


def final_csv_path() -> Path:
    if _default_ledger():
        return RESULTS_DIR / "final_submission_test.csv"
    return JOURNAL_PATH.parent / (JOURNAL_PATH.stem + "_final_submission_test.csv")

EPSILON = 0.002     # organizers' convergence rule
N_CONVERGE = 3
ITERATION_CAP = 50
WALL_CEILING_S = 6 * 3600

BASELINE_TEST_PRIMARY = 0.5946  # organizers' published FM hidden-test primary

sys.path.insert(0, str(KIT))  # organizers' modules: data, evaluate, submit

# Module-level ledger path; `--ledger` swaps it for scratch/test runs.
JOURNAL_PATH = RESULTS_DIR / "JOURNAL.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_hashes(require_sanitized: bool = True) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    bad = []
    for name, expected in manifest["files"].items():
        if sha256_file(ROOT / name) != expected:
            bad.append(name)
    for name, expected in manifest.get("dataset_files", {}).items():
        p = ROOT / name
        if not p.exists() or sha256_file(p) != expected:
            bad.append(name)
    if require_sanitized:
        sanitized = manifest.get("dataset_files_sanitized", {})
        for name, expected in sanitized.items():
            p = ROOT / name
            if not p.exists() or sha256_file(p) != expected:
                bad.append(name + " (run `iterate.py sanitize-data` to regenerate)")
    if bad:
        raise SystemExit(f"INTEGRITY FAILURE: organizer/dataset files changed or missing: {bad}")


def git_state() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return {"git_rev": rev or "unknown", "git_dirty": dirty}
    except Exception:
        return {"git_rev": "unknown", "git_dirty": True}


# ---------------- trusted scoring core (snapshotted before candidate code) ---

class Trusted:
    """Loads organizer modules, snapshots callables, keeps PRIVATE test labels,
    and owns a tamper probe for the evaluator."""

    _PROBE_USERS = ["u1", "u1", "u1", "u2", "u2", "u3"]
    _PROBE_LABELS = [1, 0, 0, 1, 1, 0]
    _PROBE_SCORES = [0.9, 0.5, 0.1, 0.3, 0.7, 0.4]

    def __init__(self):
        from data import load  # organizers' modules, hash-verified beforehand
        from evaluate import evaluate
        from submit import write_submission, read_submission
        self._evaluate = evaluate
        self._write_submission = write_submission
        self._read_submission = read_submission
        self.splits = load(str(DATA_DIR))
        # Private labels; solutions receive label-stripped test rows.
        self._test_labels = [x[6] for x in self.splits["test"]]
        self._probe_expected = json.dumps(
            self._evaluate(self._PROBE_USERS, self._PROBE_LABELS, self._PROBE_SCORES),
            sort_keys=True)

    def probe(self, stage: str) -> None:
        now = json.dumps(
            self._evaluate(self._PROBE_USERS, self._PROBE_LABELS, self._PROBE_SCORES),
            sort_keys=True)
        if now != self._probe_expected:
            raise SystemExit(
                f"TAMPER DETECTED ({stage}): the evaluator no longer reproduces "
                "its pre-candidate probe result. Run aborted; nothing recorded "
                "as a scored result.")

    def restricted_splits(self) -> dict:
        stripped_test = [x[:6] + (0,) + x[7:] for x in self.splits["test"]]
        return {"train": list(self.splits["train"]),
                "valid": list(self.splits["valid"]),
                "test": stripped_test}

    def _check_scores(self, rows, scores):
        import numpy as np
        arr = np.asarray(list(scores), dtype=float)
        if len(arr) != len(rows):
            raise SystemExit(f"solution returned {len(arr)} scores for {len(rows)} rows")
        if not np.all(np.isfinite(arr)):
            raise SystemExit("solution returned NaN/Inf scores")
        return arr

    def score_valid(self, scores):
        rows = self.splits["valid"]
        arr = self._check_scores(rows, scores)
        self.probe("before validation scoring")
        return self._evaluate([x[1] for x in rows], [x[6] for x in rows], list(arr))

    def seal_test_scores(self, entry_id: str, scores) -> dict:
        import numpy as np
        arr = self._check_scores(self.splits["test"], scores)
        d = sealed_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{entry_id}.npy"
        rel = (str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        np.save(path, arr)
        return {"path": rel, "sha256": sha256_file(path)}

    def load_sealed(self, sealed_path_str: str, expected_sha: str):
        """Load + verify a sealed prediction file WITHOUT evaluating it —
        hidden-test labels are consulted exactly once, in score_csv()
        (codex round 3, finding 1)."""
        import numpy as np
        path = Path(sealed_path_str)
        if not path.is_absolute():
            path = ROOT / path
        if sha256_file(path) != expected_sha:
            raise SystemExit("sealed test scores do not match their journaled hash")
        arr = np.load(path)
        if len(arr) != len(self.splits["test"]):
            raise SystemExit("sealed score count does not match test split")
        if not np.all(np.isfinite(arr)):
            raise SystemExit("sealed scores contain NaN/Inf")
        return arr

    def write_and_check_submission(self, csv_path: Path, arr) -> None:
        self._write_submission(str(csv_path), self.splits["test"], list(arr))
        # The organizers' own checker: header, row count, row_id continuity,
        # alignment, numeric/finite scores.
        self._read_submission(str(csv_path), self.splits["test"])

    def score_csv(self, csv_path: Path):
        """Score the checker-PARSED CSV — the exact submitted artifact."""
        parsed = self._read_submission(str(csv_path), self.splits["test"])
        rows = self.splits["test"]
        self.probe("before final CSV scoring")
        return self._evaluate([x[1] for x in rows], self._test_labels, parsed)


# ---------------- journal ----------------

def read_journal(fail_closed: bool = False) -> list:
    """fail_closed=True (used by `final` and by run's gate checks): ANY
    malformed line is treated as potentially-hidden state and blocks the
    operation, instead of being skipped (codex round 2, finding 5)."""
    if not JOURNAL_PATH.exists():
        return []
    out, malformed = [], 0
    for line in JOURNAL_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            malformed += 1
    if malformed:
        message = (f"{JOURNAL_PATH.name} contains {malformed} malformed line(s)")
        if fail_closed:
            raise SystemExit(f"LEDGER INTEGRITY: {message} — refusing to proceed; "
                             "repair/inspect the journal first")
        print(f"[warning] {message} — investigate before trusting derived results",
              file=sys.stderr)
    return out


def append_journal(entry: dict) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def convergence_state(entries: list) -> dict:
    """Organizers' rule (epsilon=0.002, N=3) applied over SUCCESSFUL iteration
    scores in order. Failed iterations count against the 50-cap but contribute
    no score to the window (documented interpretation). The 6h ceiling runs
    from the first iteration's timestamp."""
    all_iterations = [e for e in entries if e.get("type") == "iteration"]
    starts = [e for e in entries if e.get("type") == "run_start"]
    if starts:
        marker_id = starts[0]["entry_id"]
        seen = False
        official = []
        for e in entries:
            if e.get("entry_id") == marker_id:
                seen = True
            if seen and e.get("type") == "iteration":
                official.append(e)
        iteration_entries = official
    else:
        # No official run yet: everything so far is SETUP. Budgets and
        # convergence bind the official run only (organizers' "50 iterations
        # per benchmark run"); setup history stays in the journal but does not
        # consume the budget or trigger convergence (codex round 4).
        iteration_entries = []
    setup_iterations = len(all_iterations) - len(iteration_entries)
    primaries = [e["valid_metrics"]["primary"] for e in iteration_entries
                 if e.get("valid_metrics")]
    # best_entry_id points at the best FINALIZABLE entry (error-free + sealed,
    # last among tied maxima) so the final gate and the digest agree; falls
    # back to any scored entry when nothing is finalizable yet.
    best = None
    best_entry_id = None
    if primaries:
        candidates = [e for e in iteration_entries
                      if e.get("valid_metrics") and e.get("error") is None
                      and e.get("sealed_test_scores")]
        pool = candidates or [e for e in iteration_entries if e.get("valid_metrics")]
        best = max(e["valid_metrics"]["primary"] for e in pool)
        best_entry_id = [e for e in pool
                         if e["valid_metrics"]["primary"] == best][-1]["entry_id"]
    converged = False
    if len(primaries) > N_CONVERGE:
        converged = max(primaries[-N_CONVERGE:]) <= max(primaries[:-N_CONVERGE]) + EPSILON
    # The 6h clock runs from the journaled run_start marker (written by
    # `start-run` when the official autonomous run begins); setup/baseline
    # iterations before the marker do not consume the allowance (codex round 2,
    # finding 6). Falls back to the first iteration if no marker exists.
    elapsed = 0.0
    anchor = starts[0] if starts else None
    if anchor:
        try:
            first = time.mktime(time.strptime(
                anchor["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
            elapsed = time.time() - first
        except Exception:
            pass
    return {
        "iterations_used": len(iteration_entries),
        "setup_iterations": setup_iterations,
        "official_run_started": bool(starts),
        "iteration_cap": ITERATION_CAP,
        "best_valid_primary": best,
        "best_entry_id": best_entry_id,
        "converged": converged,
        "elapsed_seconds": elapsed,  # raw, ungated by rounding (codex round 4)
        "elapsed_hours": round(elapsed / 3600, 2),
        "wall_ceiling_hours": WALL_CEILING_S / 3600,
    }


def base_entry_fields(entry_type: str) -> dict:
    import secrets
    return {
        "entry_id": time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": entry_type,
        "harness_version": HARNESS_VERSION,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }


def final_exists(entries: list) -> bool:
    return any(e.get("type") in ("final", "final_pending") for e in entries)


SUSPICIOUS_SOURCE_PATTERNS = [
    # Detection, not prevention (cooperative trust model): patterns that would
    # let a solution reach RAW test labels or harness internals. File-level
    # data access is legitimate via the sanitized copy (data_sanitized/);
    # only the raw dir is flagged. Also flags signal handling (a cooperative
    # solution must not catch the harness timeout).
    r"data\s*\.\s*load|from\s+data\s+import\s+load",
    r"_test_labels", r"_getframe", r"inspect\.",
    r"KuaiRand-Pure/data/",
    r"['\"]data['\"]",          # path-join forms: Path(...)/'data'/...
    r"read_csv|loadtxt|genfromtxt",  # tabular readers aimed at files
    r"log_standard|log_random",      # raw log filenames (sanitized dir is fine via 'data_sanitized')
    r"\bsignal\b", r"TimeoutError",
]


def scan_source(source_text: str) -> list:
    return sorted({pat for pat in SUSPICIOUS_SOURCE_PATTERNS
                   if re.search(pat, source_text)})


def read_solution_source(path: Path):
    """Read + hash BEFORE any execution so even a failing solution keeps full
    provenance in the journal (codex round 2, finding 7)."""
    source_bytes = path.read_bytes()
    return (sha256_bytes(source_bytes),
            source_bytes.decode("utf-8", errors="replace"), source_bytes)


def load_solution(path: Path, source_bytes: bytes):
    module = types.ModuleType(path.stem)
    module.__file__ = str(path)
    sys.modules[path.stem] = module
    exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    if not hasattr(module, "run") or not hasattr(module, "HYPOTHESIS"):
        raise SystemExit(f"{path} must define HYPOTHESIS and run(splits)")
    return module


# ---------------- commands ----------------

def cmd_run(args) -> int:
    verify_hashes()
    entries = read_journal(fail_closed=True)
    state = convergence_state(entries)

    if (args.post_final or args.continue_past_convergence) and not args.override_reason.strip():
        raise SystemExit("run override flags require a non-empty --override-reason")
    if final_exists(entries) and not args.post_final:
        raise SystemExit("a final entry exists — development is closed. "
                         "(--post-final overrides; needs --override-reason)")
    if state["official_run_started"] and state["iterations_used"] >= ITERATION_CAP:
        raise SystemExit("iteration cap (50) reached — no further runs allowed")
    if state["converged"] and not args.continue_past_convergence:
        raise SystemExit(
            "converged by the organizers' rule (no >0.002 improvement over the "
            "last 3 scores) — designate a final. "
            "(--continue-past-convergence overrides and is journaled as such)")
    if state["elapsed_seconds"] > WALL_CEILING_S and not args.continue_past_convergence:
        raise SystemExit("6h wall-clock ceiling exceeded — designate a final")

    entry = {
        **base_entry_fields("iteration"),
        "iteration": state["iterations_used"] + 1,
        **git_state(),
        "llm_tokens_reported": args.tokens,
        "overrides": {
            "post_final": bool(args.post_final),
            "continue_past_convergence": bool(args.continue_past_convergence),
            "reason": args.override_reason or "",
        },
        "timeout_seconds": args.timeout,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "leak_guard": "mechanical: test labels stripped before solution code runs",
        "phase": "official" if state["official_run_started"] else "setup",
    }

    # Provenance survives every failure path: source read + hashed pre-exec.
    solution_path = Path(args.solution).resolve()
    try:
        sha, source_text, source_bytes = read_solution_source(solution_path)
        rel = (str(solution_path.relative_to(ROOT))
               if solution_path.is_relative_to(ROOT) else str(solution_path))
        entry["solution"] = {"path": rel, "sha256": sha, "source": source_text}
        entry["source_flags"] = scan_source(source_text)
    except Exception as exc:
        entry["solution"] = {"path": args.solution}
        entry["error"] = f"source unreadable: {exc}"
        entry["valid_metrics"] = None
        entry["wall_seconds"] = 0.0
        entry["convergence"] = convergence_state(entries + [entry])
        append_journal(entry)
        print(json.dumps({"iteration": entry["iteration"], "error": entry["error"]}, indent=2))
        return 2
    if entry["source_flags"]:
        print(f"[audit-flag] solution source matches suspicious patterns: "
              f"{entry['source_flags']} — journaled for review", file=sys.stderr)

    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"iteration exceeded --timeout {args.timeout}s")

    t0 = time.time()
    # The alarm brackets the ENTIRE iteration — data load, candidate import,
    # run, lazy-score consumption, validation scoring, probing, sealing — so a
    # hang anywhere is bounded (codex round 3, finding 2). A cooperative
    # solution must not catch harness TimeoutError (scanner flags signal use).
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(1, int(args.timeout)))
    try:
        trusted = Trusted()          # loads data + probes evaluator FIRST
        module = load_solution(solution_path, source_bytes)  # candidate code runs here
        entry["hypothesis"] = getattr(module, "HYPOTHESIS", "")
        result = module.run(trusted.restricted_splits())
        entry["valid_metrics"] = trusted.score_valid(result["valid"])
        # Probe BEFORE sealing: a tamper-detected run must not leave a seal
        # behind (codex round 2, finding 2 ordering caveat).
        trusted.probe("after solution run")
        entry["sealed_test_scores"] = trusted.seal_test_scores(
            entry["entry_id"], result["test"])
        entry["error"] = None
    except BaseException as exc:  # noqa: BLE001 — recovery evidence IS graded
        if isinstance(exc, KeyboardInterrupt):
            raise
        if "hypothesis" not in entry:
            m = re.search(r"HYPOTHESIS\s*=\s*[\"\']([^\"\']*)", source_text)
            entry["hypothesis"] = m.group(1) if m else ""
        entry["valid_metrics"] = None
        entry["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    entry["wall_seconds"] = round(time.time() - t0, 1)
    entry["convergence"] = convergence_state(entries + [entry])
    if entry["convergence"]["elapsed_seconds"] > WALL_CEILING_S:
        entry["over_ceiling"] = True

    append_journal(entry)
    print(json.dumps({
        "iteration": entry["iteration"],
        "hypothesis": entry.get("hypothesis", ""),
        "valid_primary": (entry["valid_metrics"] or {}).get("primary"),
        "error": entry["error"],
        "wall_seconds": entry["wall_seconds"],
        "convergence": entry["convergence"],
    }, indent=2))
    return 0 if entry["error"] is None else 2


def cmd_final(args) -> int:
    """Score the designated iteration's SEALED test scores, exactly once.

    Enforced (codex round 2, findings 3/5): fail-closed ledger read; an
    exclusive lockfile against concurrent finals; the designated entry must be
    error-free and SHOULD be the validation-best (deviating requires
    --not-best with a journaled reason); the run should have terminated
    (converged / cap / ceiling) unless --early-final with a journaled reason;
    --force (re-final) requires a non-empty reason and is recorded on the
    final entry itself."""
    verify_hashes()
    lock_path = JOURNAL_PATH.parent / ".final.lock"
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(lock_fd)
    except FileExistsError:
        raise SystemExit(f"another final appears to be in progress ({lock_path} "
                         "exists) — remove it only if you are certain none is")
    try:
        # Ledger read happens INSIDE the lock (codex round 3, finding 3:
        # a pre-lock read could act on a stale no-final snapshot).
        entries = read_journal(fail_closed=True)
        if final_exists(entries) and not args.force:
            raise SystemExit("a final already exists — test may be scored only once. "
                             "(--force overrides; requires a non-empty --reason)")
        if args.force and not args.reason.strip():
            raise SystemExit("--force requires a non-empty --reason")
        target = next((e for e in entries if e.get("entry_id") == args.entry
                       and e.get("type") == "iteration"), None)
        if target is None:
            raise SystemExit(f"iteration entry {args.entry} not found in journal")
        if target.get("error") is not None:
            raise SystemExit(f"entry {args.entry} recorded an error — an errored "
                             "iteration cannot be designated final")
        seal = target.get("sealed_test_scores")
        if not seal:
            raise SystemExit(f"entry {args.entry} has no sealed test scores")
        state = convergence_state(entries)
        if target["entry_id"] != state.get("best_entry_id") and not args.not_best:
            raise SystemExit(
                f"entry {args.entry} is not the validation-best "
                f"({state.get('best_entry_id')}, primary {state.get('best_valid_primary')}). "
                "The organizers require the validation-best checkpoint. "
                "(--not-best overrides; requires a non-empty --reason)")
        if args.not_best and not args.reason.strip():
            raise SystemExit("--not-best requires a non-empty --reason")
        terminated = (state["converged"]
                      or state["iterations_used"] >= ITERATION_CAP
                      or state["elapsed_seconds"] > WALL_CEILING_S)
        if not terminated and not args.early_final:
            raise SystemExit("the run has not terminated (not converged, under the "
                             "cap and ceiling) — finalizing now requires "
                             "--early-final with a non-empty --reason")
        if args.early_final and not args.reason.strip():
            raise SystemExit("--early-final requires a non-empty --reason")

        # Marker BEFORE scoring: even a crash leaves evidence test was consumed.
        append_journal({
            **base_entry_fields("final_pending"),
            "designated_entry": args.entry,
            "forced": bool(args.force),
            "not_best": bool(args.not_best),
            "early_final": bool(args.early_final),
            "override_reason": args.reason or "",
        })

        trusted = Trusted()
        arr = trusted.load_sealed(seal["path"], seal["sha256"])
        csv_path = final_csv_path()
        trusted.write_and_check_submission(csv_path, arr)
        # CSV parity (codex round 2, finding 4): the journaled metric is
        # computed from the checker-PARSED CSV values — the exact bytes that
        # would be submitted — not from the raw sealed array.
        metrics = trusted.score_csv(csv_path)

        entry = {
            **base_entry_fields("final"),
            "designated_entry": args.entry,
            "designated_solution": {k: v for k, v in target.get("solution", {}).items()
                                    if k != "source"},
            **git_state(),
            "valid_metrics": target.get("valid_metrics"),
            "test_metrics_from_submitted_csv": metrics,
            "baseline_test_primary": BASELINE_TEST_PRIMARY,
            "delta_over_baseline": round(metrics["primary"] - BASELINE_TEST_PRIMARY, 4),
            "forced": bool(args.force),
            "not_best": bool(args.not_best),
            "early_final": bool(args.early_final),
            "override_reason": args.reason or "",
            "submission_csv": (str(csv_path.relative_to(ROOT))
                               if csv_path.is_relative_to(ROOT) else str(csv_path)),
            "submission_csv_sha256": sha256_file(csv_path),
        }
        append_journal(entry)
        print(json.dumps(entry, indent=2))
        return 0
    finally:
        lock_path.unlink(missing_ok=True)


SANITIZED_DIR = DATA_DIR.parent / "data_sanitized"
# Feedback columns zeroed on TEST-date rows (>= 20220429): every engagement
# signal, including play_time_ms (long_view is derivable from play time vs
# duration, so play time must go too). duration_ms/is_rand are item/context
# properties and stay. Train/validation rows are untouched.
FEEDBACK_COLUMNS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward",
                    "is_hate", "long_view", "play_time_ms", "profile_stay_time",
                    "comment_stay_time", "is_profile_enter"]
TEST_DATE_START = 20220429


def cmd_sanitize(args) -> int:
    """Build the sanitized dataset copy — the SANCTIONED path for solutions
    that need file-level access to auxiliary signals (sequences, multi-task
    labels): identical to the raw data except every feedback column is zeroed
    on test-date rows (codex round 3, finding 4: raw rereads are a plausible
    cooperative mistake, so a safe copy must exist)."""
    import csv as _csv
    import shutil
    SANITIZED_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    LOGS_TO_SANITIZE = {"log_standard_4_22_to_5_08_pure.csv",
                        "log_random_4_22_to_5_08_pure.csv"}
    for f in sorted(DATA_DIR.iterdir()):
        out = SANITIZED_DIR / f.name
        if f.name in LOGS_TO_SANITIZE:
            with open(f, newline="") as src, open(out, "w", newline="") as dst:
                reader = _csv.DictReader(src)
                writer = _csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                zeroed = 0
                for row in reader:
                    if int(row["date"]) >= TEST_DATE_START:
                        for col in FEEDBACK_COLUMNS:
                            if col in row:
                                row[col] = "0"
                        zeroed += 1
                    writer.writerow(row)
            report[f.name] = f"test-date rows zeroed: {zeroed}"
        else:
            shutil.copyfile(f, out)
            report[f.name] = "copied unchanged"
        report[f.name + " sha256"] = sha256_file(out)
    print(json.dumps({"sanitized_dir": str(SANITIZED_DIR.relative_to(ROOT)),
                      "files": report}, indent=2))
    return 0


def cmd_log(args) -> int:
    entries = read_journal()
    for e in entries:
        vm = e.get("valid_metrics") or {}
        print(f"#{e.get('iteration', '-')} {e.get('type')} | "
              f"primary={vm.get('primary', '-')} | err={bool(e.get('error'))} | "
              f"{e.get('hypothesis', '')[:70]}")
    print(json.dumps(convergence_state(entries), indent=2))
    return 0


def cmd_intervention(args) -> int:
    append_journal({**base_entry_fields("intervention"),
                    "description": args.describe})
    print("intervention recorded")
    return 0


def main() -> int:
    global JOURNAL_PATH
    parser = argparse.ArgumentParser(description="Track 2 iteration harness")
    parser.add_argument("--ledger", default=None,
                        help="alternate journal path for scratch/test runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify organizer file + dataset hashes")
    p_run = sub.add_parser("run", help="run one iteration (validation feedback only)")
    p_run.add_argument("--solution", required=True)
    p_run.add_argument("--tokens", type=int, default=0,
                       help="LLM tokens spent authoring this iteration (self-reported)")
    p_run.add_argument("--post-final", action="store_true",
                       help="journaled override: development run after a final exists")
    p_run.add_argument("--continue-past-convergence", action="store_true",
                       help="journaled override: run past the convergence rule")
    p_run.add_argument("--override-reason", default="",
                       help="required reason when using a run override flag (journaled)")
    p_run.add_argument("--timeout", type=int, default=1800,
                       help="per-iteration wall timeout in seconds (journaled)")
    p_fin = sub.add_parser("final",
                           help="score one designated iteration's sealed test scores, once")
    p_fin.add_argument("--entry", required=True, help="journal entry_id to designate")
    p_fin.add_argument("--force", action="store_true",
                       help="journaled override of the once-only rule (needs --reason)")
    p_fin.add_argument("--not-best", action="store_true",
                       help="journaled override: designate a non-validation-best entry")
    p_fin.add_argument("--early-final", action="store_true",
                       help="journaled override: finalize before termination")
    p_fin.add_argument("--reason", default="", help="reason for any override flag")
    sub.add_parser("start-run",
                   help="journal the official run_start marker (starts the 6h clock)")
    sub.add_parser("sanitize-data",
                   help="build the sanitized dataset copy (safe file-level access for solutions)")
    sub.add_parser("log", help="print journal summary + convergence state")
    p_int = sub.add_parser("intervention", help="record a manual human intervention")
    p_int.add_argument("--describe", required=True)

    args = parser.parse_args()
    if args.ledger:
        JOURNAL_PATH = Path(args.ledger).resolve()
    if args.cmd == "check":
        verify_hashes()
        print("hashes OK (organizer files + dataset)")
        return 0
    if args.cmd == "sanitize-data":
        verify_hashes(require_sanitized=False)
        return cmd_sanitize(args)
    if args.cmd == "start-run":
        existing = [e for e in read_journal(fail_closed=True)
                    if e.get("type") == "run_start"]
        if existing:
            raise SystemExit(f"a run_start marker already exists "
                             f"({existing[0]['entry_id']}) — the clock anchors to "
                             "the FIRST marker and cannot be reset")
        append_journal(base_entry_fields("run_start"))
        print("run_start journaled — the 6h ceiling clock starts now")
        return 0
    return {"run": cmd_run, "final": cmd_final, "log": cmd_log,
            "intervention": cmd_intervention}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
