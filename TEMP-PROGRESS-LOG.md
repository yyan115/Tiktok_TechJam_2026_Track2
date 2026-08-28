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


## WHAT WE ACTUALLY BUILT (plain words, one line each, with the file to open)

**The machine:**
- **The lab bench** — one script that runs each experiment, scores it with the ORGANIZERS' OWN scoring code, and writes the result to the logbook. I run it; I never edit it. → `Project/harness/iterate.py`
- **The honesty machinery** — experiments physically cannot see the test answers (labels stripped before their code runs); every experiment's test predictions get sealed unopened; the test set is scored ONCE, at the very end, on the exact file that would be submitted — and the bench refuses second attempts, non-best picks, and running past the budget. → enforced inside `iterate.py`
- **The sanitized dataset** — a copy of the data with every feedback signal blanked on test-period rows, so even file-level feature work can't accidentally peek. → `kuairand-starter-kit/KuaiRand-Pure/data_sanitized/` (rebuildable via `sanitize-data`)
- **The fingerprint pins** — the organizers' files AND the dataset (raw + sanitized) are fingerprint-recorded; the bench refuses to run if anything changed. → `Project/manifest.json`
- **The budget clock** — the competition's 50-experiment cap and 6-hour ceiling are enforced code, and their clock only starts at the explicit `start-run` marker; everything before it is setup that consumes nothing.
- **The locks** — settings that make my editing tools refuse to touch the organizers' code, the bench, or the results. YOU arm these (your 2-line paste + restart). → `.claude/settings.json`

**The memory (so no session ever starts blank):**
- **Status board** — where we are, what's next; auto-loaded into every new session. → `Project/memory/STATE.md`
- **Diary** — every decision and every review round, plain language, dated. → `Project/memory/DECISIONS.md`
- **Mistakes list** — including the organizers' own published dead-ends, never to be retried. → `Project/memory/LESSONS.md`
- **Logbook** — every experiment: hypothesis, full code, score, errors; machine-written. → `Project/results/JOURNAL.jsonl`
- **Digest** — the one-page view of the logbook the agent reads at session start. → `python3 Project/tools/digest.py`

**The oversight:**
- **Second-AI review trail** — codex reviewed this bench TWELVE times, rejecting it for real flaws until "remaining blockers: none." Final sign-off verbatim: → `Project/audits/track2_harness_verdict_round12.md`
- **Operating manual** — every command, what writes what, the honesty rules. → `Project/RUNBOOK.md`

## HOW TO CHECK IT YOURSELF (10 min, no code reading)

1. Read the reviewer's final verdict: `Project/audits/track2_harness_verdict_round12.md`
2. Skim the diary for the whole story: `Project/memory/DECISIONS.md`
3. Watch the machinery work — run these in this folder:
   `python3 Project/harness/iterate.py check`   (fingerprints: should print hashes OK)
   `python3 Project/tools/digest.py`   (the logbook digest: 0/50 official experiments used)
   `python3 Project/harness/iterate.py --ledger /tmp/t2rt.jsonl run --solution Project/harness/redteam/rt02_hang.py --timeout 3`   (a deliberately-hanging experiment gets killed by the timeout — you'll see TimeoutError journaled)
4. After your restart: tell Claude "try to edit the bench" — watch the lock block it.
5. Anytime, forever: any claim Claude makes → "show me the journal entry."

## What the plan is after your steps

Autonomous run to convergence → designate the validation-best → score the hidden test
ONCE via the enforced `final` → package. Full plan: `Project/PLAN.md`. Status:
`Project/memory/STATE.md`. Operations: `Project/RUNBOOK.md`.
