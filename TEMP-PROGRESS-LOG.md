# TEMP — read this when you're back (Track 2)

*Rewritten 28 Aug 13:27 (supersedes the overnight version; time from the system clock). Delete once done.
Live counts always come from `python3 Project/tools/digest.py` — currently:
0/50 OFFICIAL iterations used (the official run hasn't started; 5 setup runs
are journaled as history and consume nothing).*

## Where things stand (10-second version)

Setup is fully built AND battle-tested: the experiment bench went through TWELVE rounds of independent AI
review, all documented in Project/memory/DECISIONS.md, ending in a clean YES ('remaining
blockers: none' — verdict file in Project/audits/). Along the way real flaws got fixed: test-label leaks made mechanically impossible, score-the-test-once
enforced in code, budgets/clock enforced, sanitized dataset created for file-level feature
work). Reviews are COMPLETE — nothing is pending. Your freeze steps below are all that remains.

## YOUR TO-DO, in order

1. **Freeze the bench** (same ceremony as Track 3): open `.claude/settings.json` IN THIS
   REPO, add inside `"deny": [...]` (comma after the previous entry):

   ```
   "Edit(Project/harness/**)",
   "Write(Project/harness/**)"
   ```

   Then restart the Claude session, have it try to edit `Project/harness/iterate.py`
   (must be blocked), and say **"freeze approved track 2"**.

2. Say **"go track 2"** — Claude journals the official `start-run` marker (the 50-iteration
   budget and 6-hour clock begin THERE) and the autonomous experiment run starts: best
   ideas first from the organizers' own hint list, every attempt journaled, no babysitting.

3. That's it until packaging day (report + 3-min video + Devpost form).

## Known limitation you may care about

The pasted problem statement in README.md contains the organizers' own contradictory
metrics row ("Limits" says NDCG@10/Recall@50/click) — the shipped scoring code is
authoritative (GAUC + nDCG@5, long_view). README is your file; annotate it if you like.

## What the plan is after your steps

Autonomous run to convergence → designate the validation-best → score the hidden test
ONCE via the enforced `final` → package. Full plan: `Project/PLAN.md`. Status:
`Project/memory/STATE.md`. Operations: `Project/RUNBOOK.md`.
