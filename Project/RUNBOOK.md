# RUNBOOK — how to operate this project (one page)

## Environment
Fedora Linux, Python 3.14, numpy (organizers' requirement — no torch/pandas needed for the
baseline). Dataset: run the two commands in `.gitignore`'s header to re-download (47 MB).
Everything runs on CPU; the baseline iteration takes ~30 s.

## The lab bench (iteration harness) — all commands from repo root
```
python3 Project/harness/iterate.py check                                  # organizer files + dataset hashes
python3 Project/harness/iterate.py run --solution Project/solutions/sXXX.py [--tokens N]
python3 Project/harness/iterate.py log                                    # journal summary + convergence
python3 Project/harness/iterate.py final --entry ENTRYID                  # ONCE, at the very end
python3 Project/harness/iterate.py intervention --describe "what and why" # honesty: log any manual help
```
Prepend `--ledger /path/scratch.jsonl` to isolate test/wiring runs from the production
journal (a scratch final writes its CSV next to the scratch ledger).

**Serialization rule: exactly ONE harness process at a time** (append-only journal, no
locking by design — single-operator project).

## Honesty rules the harness enforces mechanically
- Solutions receive test rows with labels STRIPPED; test labels live only inside the
  harness. Development feedback is validation-only.
- `run` refuses after convergence (ε=0.002 / N=3 over successful scores), past 50
  iterations, past the 6h ceiling, or after a final exists — every override flag is
  itself journaled.
- `final` scores the designated iteration's SEALED prediction file (no retraining — the
  scored artifact IS the measured artifact), refuses a second invocation, journals a
  `final_pending` marker before scoring so even a crash leaves evidence, and validates
  the CSV with the organizers' own checker.
- The evaluator is probed before/after candidate code runs; drift aborts the run.
  Same-process residual documented in the harness docstring (cooperative trust model).

## Solution contract (files in Project/solutions/)
`HYPOTHESIS = "..."` and `run(splits) -> {'valid': scores, 'test': scores}`, where splits
is the organizers' row-tuple format with test labels zeroed. Scores: finite reals,
row-aligned; only relative order matters. Full source is journaled verbatim per run.

## Recovery / gotchas
- A failed iteration still journals (error recorded, counts against the 50-cap, no score).
  That's by design — recovery evidence is graded.
- The 6h clock runs from the first journaled iteration's timestamp.
- The elapsed/converged state is printed by `log` — check it before each run.
- Enforcement layers, honestly ranked: (1) deny rules in `.claude/settings.json` — the lock;
  (2) committed hashes + git history — tampering is visible; (3) the Bash guard hook — an
  accident seatbelt, never the load-bearing protection.
- Fresh session? `Project/memory/STATE.md` auto-injects on start; CLAUDE.md points everywhere.
