#!/usr/bin/env python3
"""Iteration harness ("the lab bench"), v0.1.0-unfrozen.

Runs one candidate solution through the official scoring pipeline and appends a
machine-written journal entry — the journal is a required competition
deliverable (hypothesis, code identity, validation metrics, errors/recovery,
wall-clock) and the evidence for the Autonomy score.

Hard rules enforced here:
  - organizer files are hash-verified before every run (manifest.json)
  - the agent develops on train + validation ONLY: run mode never evaluates or
    prints test scores. Test is scored exactly once, via `final`, on the
    designated solution.
  - solution source is hashed and executed from those exact bytes
  - every entry records the convergence state (epsilon=0.002 over N=3, the
    organizers' own rule) and the iteration count against the 50-iteration cap

Solution contract (a .py file, usually in Project/solutions/):
  HYPOTHESIS = "one line: what this tries and why"        (required)
  def run(data_dir) -> dict with:
      'valid': array of scores aligned with the validation split rows
      'test':  array of scores aligned with the test split rows
  (Both arrays row-aligned with data.load()[split]. 'test' is ignored in run
   mode; it is used only by `final`.)

This file is part of the trusted lab bench. After its freeze it must not be
modified without user approval (see Project/PLAN.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

HARNESS_VERSION = "0.1.0-unfrozen"

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "kuairand-starter-kit"
DATA_DIR = KIT / "KuaiRand-Pure" / "data"
MANIFEST_PATH = ROOT / "Project" / "manifest.json"
RESULTS_DIR = ROOT / "Project" / "results"
JOURNAL_PATH = RESULTS_DIR / "JOURNAL.jsonl"

EPSILON = 0.002   # organizers' convergence rule
N_CONVERGE = 3
ITERATION_CAP = 50

sys.path.insert(0, str(KIT))  # organizers' modules: data, evaluate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hashes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    bad = []
    for name, expected in manifest["files"].items():
        actual = sha256_file(ROOT / name)
        if actual != expected:
            bad.append(name)
    if bad:
        raise SystemExit(f"INTEGRITY FAILURE: organizer files changed: {bad}")


def load_solution(path: Path):
    source_bytes = path.read_bytes()
    sha = hashlib.sha256(source_bytes).hexdigest()
    module = types.ModuleType(path.stem)
    module.__file__ = str(path)
    sys.modules[path.stem] = module
    exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    if not hasattr(module, "run") or not hasattr(module, "HYPOTHESIS"):
        raise SystemExit(f"{path} must define HYPOTHESIS and run(data_dir)")
    return module, sha


def read_journal() -> list:
    if not JOURNAL_PATH.exists():
        return []
    out = []
    for line in JOURNAL_PATH.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def append_journal(entry: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def convergence_state(entries: list) -> dict:
    """Organizers' rule: converged when validation primary has not improved by
    more than EPSILON over the last N_CONVERGE consecutive iterations."""
    primaries = [
        e["valid_metrics"]["primary"]
        for e in entries
        if e.get("type") == "iteration" and e.get("valid_metrics")
    ]
    iterations_used = len([e for e in entries if e.get("type") == "iteration"])
    best = max(primaries) if primaries else None
    converged = False
    if len(primaries) > N_CONVERGE:
        best_before = max(primaries[:-N_CONVERGE])
        converged = max(primaries[-N_CONVERGE:]) <= best_before + EPSILON
    return {
        "iterations_used": iterations_used,
        "iteration_cap": ITERATION_CAP,
        "best_valid_primary": best,
        "converged": converged,
    }


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def leak_guard_note() -> str:
    """The mechanical guard: data.load() splits by date; training data is
    whatever the solution takes from splits['train'] (dates <= 20220421).
    Solutions receive only data_dir; this harness never exposes test metrics in
    run mode. Recorded here so every journal entry carries the discipline."""
    return "validation-only feedback; test scored once via `final`; train dates <= 20220421"


def _score(rows, scores):
    from evaluate import evaluate  # organizers' module, hash-verified
    if len(scores) != len(rows):
        raise SystemExit(
            f"solution returned {len(scores)} scores for a split of {len(rows)} rows"
        )
    return evaluate([x[1] for x in rows], [x[6] for x in rows], list(scores))


def cmd_run(args) -> int:
    verify_hashes()
    from data import load  # organizers' module
    entries = read_journal()
    state_before = convergence_state(entries)
    if state_before["iterations_used"] >= ITERATION_CAP:
        raise SystemExit("iteration cap (50) reached — no further runs allowed")

    solution_path = Path(args.solution).resolve()
    module, sha = load_solution(solution_path)

    splits = load(str(DATA_DIR))
    entry = {
        "entry_id": time.strftime("%Y%m%d-%H%M%S"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": "iteration",
        "iteration": state_before["iterations_used"] + 1,
        "solution": {"path": str(solution_path.relative_to(ROOT)), "sha256": sha},
        "hypothesis": getattr(module, "HYPOTHESIS", ""),
        "harness_version": HARNESS_VERSION,
        "git_rev": git_rev(),
        "leak_guard": leak_guard_note(),
        "llm_tokens_reported": args.tokens,
    }

    t0 = time.time()
    try:
        result = module.run(str(DATA_DIR))
        entry["valid_metrics"] = _score(splits["valid"], result["valid"])
        entry["error"] = None
    except BaseException as exc:  # noqa: BLE001 — robustness IS the graded feature
        if isinstance(exc, KeyboardInterrupt):
            raise
        entry["valid_metrics"] = None
        entry["error"] = f"{type(exc).__name__}: {exc}"
    entry["wall_seconds"] = round(time.time() - t0, 1)
    entry["convergence"] = convergence_state(entries + [entry])

    append_journal(entry)
    print(json.dumps({
        "iteration": entry["iteration"],
        "hypothesis": entry["hypothesis"],
        "valid_primary": (entry["valid_metrics"] or {}).get("primary"),
        "error": entry["error"],
        "wall_seconds": entry["wall_seconds"],
        "convergence": entry["convergence"],
    }, indent=2))
    return 0 if entry["error"] is None else 2


def cmd_final(args) -> int:
    """Score the DESIGNATED solution on test, exactly once, and write the
    submission CSV via the organizers' own writer."""
    verify_hashes()
    from data import load
    solution_path = Path(args.solution).resolve()
    module, sha = load_solution(solution_path)
    splits = load(str(DATA_DIR))

    t0 = time.time()
    result = module.run(str(DATA_DIR))
    valid_metrics = _score(splits["valid"], result["valid"])
    test_metrics = _score(splits["test"], result["test"])

    sys.path.insert(0, str(KIT))
    from submit import write_submission  # organizers' writer
    out_csv = ROOT / "Project" / "results" / "final_submission_test.csv"
    write_submission(str(out_csv), splits["test"], list(result["test"]))

    entry = {
        "entry_id": time.strftime("%Y%m%d-%H%M%S"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": "final",
        "solution": {"path": str(solution_path.relative_to(ROOT)), "sha256": sha},
        "hypothesis": getattr(module, "HYPOTHESIS", ""),
        "harness_version": HARNESS_VERSION,
        "git_rev": git_rev(),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "baseline_test_primary": 0.5946,
        "delta_over_baseline": round(test_metrics["primary"] - 0.5946, 4),
        "submission_csv": str(out_csv.relative_to(ROOT)),
        "wall_seconds": round(time.time() - t0, 1),
    }
    append_journal(entry)
    print(json.dumps(entry, indent=2))
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
    """Record a manual human intervention — honesty requirement; the count is a
    graded metric (fewer = better)."""
    append_journal({
        "entry_id": time.strftime("%Y%m%d-%H%M%S"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "type": "intervention",
        "description": args.describe,
    })
    print("intervention recorded")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 2 iteration harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify organizer file hashes")
    p_run = sub.add_parser("run", help="run one iteration (validation feedback only)")
    p_run.add_argument("--solution", required=True)
    p_run.add_argument("--tokens", type=int, default=0,
                       help="LLM tokens spent authoring this iteration (self-reported)")
    p_fin = sub.add_parser("final", help="score designated solution on test, once")
    p_fin.add_argument("--solution", required=True)
    sub.add_parser("log", help="print journal summary + convergence state")
    p_int = sub.add_parser("intervention", help="record a manual human intervention")
    p_int.add_argument("--describe", required=True)

    args = parser.parse_args()
    if args.cmd == "check":
        verify_hashes()
        print("hashes OK")
        return 0
    return {"run": cmd_run, "final": cmd_final, "log": cmd_log,
            "intervention": cmd_intervention}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
