#!/usr/bin/env python3
"""Read-only journal digest — the agent's session-start memory view.

Renders one compact page from results/JOURNAL.jsonl: run state, best result,
every hypothesis tried with its one-line outcome, flags, and interventions.
The agent reads THIS each session instead of the raw journal. Never writes
anything; deliberately lives outside the harness (tools/ is not part of the
frozen evaluator).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "Project" / "results" / "JOURNAL.jsonl"


def main() -> int:
    if not JOURNAL.exists():
        print("no journal yet")
        return 0
    entries = []
    malformed = 0
    for line in JOURNAL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            malformed += 1

    all_iterations = [e for e in entries if e.get("type") == "iteration"]
    starts = [e for e in entries if e.get("type") == "run_start"]
    # Mirror the harness's official-run scoping exactly: budget/best count from
    # the FIRST start-run marker; earlier entries are setup phase.
    if starts:
        seen = False
        iterations = []
        for e in entries:
            if e.get("entry_id") == starts[0]["entry_id"]:
                seen = True
            if seen and e.get("type") == "iteration":
                iterations.append(e)
    else:
        iterations = []
    setup = [e for e in all_iterations if e not in iterations]
    finals = [e for e in entries if e.get("type") in ("final", "final_pending")]
    interventions = [e for e in entries if e.get("type") == "intervention"]
    scored = [e for e in iterations if e.get("valid_metrics")]
    # Finalizable-best mirrors the harness's final gating exactly: error-free,
    # sealed, LAST among tied maxima (codex round 3, finding 6).
    finalizable = [e for e in scored
                   if e.get("error") is None and e.get("sealed_test_scores")]
    best = None
    if finalizable:
        top = max(e["valid_metrics"]["primary"] for e in finalizable)
        best = [e for e in finalizable if e["valid_metrics"]["primary"] == top][-1]

    print("# JOURNAL DIGEST (read-only view; source of truth = JOURNAL.jsonl)\n")
    if malformed:
        print(f"!! {malformed} malformed journal line(s) — investigate before trusting\n")
    print(f"OFFICIAL iterations used: {len(iterations)}/50 (run "
          f"{'started' if starts else 'NOT started — all entries below are setup'}) | "
          f"setup iterations: {len(setup)} | interventions: {len(interventions)} | "
          f"finals: {len(finals)}")
    if best:
        print(f"BEST: {best['entry_id']} primary={best['valid_metrics']['primary']:.4f} "
              f"({best.get('solution', {}).get('path', '?')})")
    print()
    for e in (iterations or setup):
        vm = e.get("valid_metrics")
        outcome = (f"primary {vm['primary']:.4f}" if vm
                   else f"ERROR: {str(e.get('error'))[:60]}")
        star = " ★" if best and e["entry_id"] == best["entry_id"] else ""
        flags = f" [flags: {', '.join(e['source_flags'])}]" if e.get("source_flags") else ""
        print(f"#{e.get('iteration', '?'):>2} {e['entry_id']}{star} | {outcome}{flags}")
        print(f"    {e.get('hypothesis', '')[:100]}")
    for e in interventions:
        print(f"[intervention] {e.get('description', '')[:100]}")
    for e in finals:
        print(f"[{e['type']}] designated={e.get('designated_entry')} "
              f"delta={e.get('delta_over_baseline', '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
