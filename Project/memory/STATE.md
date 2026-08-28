# STATE — read this first in every session

Updated: 2026-08-28 ~10:45 (harness v0.2.0 rebuild after codex round 1)

## Where we are
- Research + plan approved by user (28 Aug morning): see Project/PLAN.md. This repo mirrors Track 3's architecture (user's repo: ../Tiktok_TechJam_2026_Track3 — read its DECISIONS.md for the full origin story).
- Stage 0 DONE: starter kit unzipped + hash-pinned (Project/manifest.json), dataset downloaded (gitignored — re-download command in .gitignore header), all three official baselines reproduced within published seed noise. Guardrails + wiki in place.
- Stage 1 (iteration harness) REBUILT as v0.2.0-unfrozen after codex audit round 1 (verdict NO, 8 findings — all addressed): mechanical test-label stripping, evaluator tamper probes, sealed test predictions, enforced once-only final (+ crash-evidence marker + official checker), enforced convergence/cap/6h-ceiling with journaled overrides, complete journal provenance (harness sha, git state, dataset hashes, verbatim solution source). Proven end-to-end: iteration 2 at valid primary 0.6015; full final wiring proven on a scratch ledger (delta +0.0007 = published baseline; once-only and post-final refusals verified).
- Harness awaiting codex re-review (round 2) of the v0.2.0 rebuild, then user freeze — the gate before the autonomous run.
- Optimization budget: 2 of 50 iterations used (both baseline reproductions — v0.1 contract and v0.2 restricted contract).

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
