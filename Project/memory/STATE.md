# STATE — read this first in every session

Updated: 2026-08-28 ~08:45 (initial setup by the Track 3 session, user at work)

## Where we are
- Research + plan approved by user (28 Aug morning): see Project/PLAN.md. This repo mirrors Track 3's architecture (user's repo: ../Tiktok_TechJam_2026_Track3 — read its DECISIONS.md for the full origin story).
- Stage 0 DONE: starter kit unzipped + hash-pinned (Project/manifest.json), dataset downloaded (gitignored — re-download command in .gitignore header), all three official baselines reproduced within published seed noise. Guardrails + wiki in place.
- Stage 1 (iteration harness) BUILT: Project/harness/iterate.py v0.1.0-unfrozen (validation-only feedback, organizers' convergence rule, leak-guard note, journal append, intervention counter, single-shot `final` for test scoring). Proven end-to-end: iteration 1 (organizers' FM via our harness, Project/solutions/s000_fm_baseline.py) journaled at valid primary 0.6015 (official reference 0.6016) in 31 s.
- Harness NOT yet Sol-reviewed or frozen — that's its gate before the autonomous run (same ceremony as Track 3's referee).
- Optimization budget untouched: 1 of 50 iterations used (the baseline reproduction).

## Standing rules (never violate)
1. Never edit: kuairand-starter-kit/** (organizer ground truth — evaluate.py is the sole scoring authority), README.md, Project/manifest.json, Project/results/** (harness-written only), .claude/**. After freeze: Project/harness/.
2. The agent develops on train + validation ONLY. Test labels are on disk but off-limits until the one final scoring of the designated submission.
3. Every iteration goes through the harness and gets journaled: hypothesis, code diff, validation metrics, errors/recovery, tokens, wall-clock. The journal is a required competition deliverable.
4. Check LESSONS.md before working — it contains organizer-verified dead ends that must never be retried.
5. Plain language to the user; explicit user "go" before starting the autonomous run.

## Next actions (in order)
1. Sol (codex) checkpoint review of iterate.py → user approves → freeze (same ceremony as Track 3; bind the review to a committed sha, per Track 3's lesson 13).
2. User says go → autonomous run on the hypothesis queue (PLAN.md) → convergence → final submission + test-scored once.

## Blocked / needs user
- Harness freeze approval (after it's built + Sol-reviewed).
- The "start the run" go.
