# STATE — read this first in every session

Updated: 2026-08-28 ~13:20 (harness v0.5.0 after codex round 4)

## Where we are
- Research + plan approved by user (28 Aug morning): see Project/PLAN.md. This repo mirrors Track 3's architecture (user's repo: ../Tiktok_TechJam_2026_Track3 — read its DECISIONS.md for the full origin story).
- Stage 0 DONE: starter kit unzipped + hash-pinned (Project/manifest.json), dataset downloaded (gitignored — re-download command in .gitignore header), all three official baselines reproduced within published seed noise. Guardrails + wiki in place.
- Stage 1 (iteration harness) now v0.3.0-unfrozen after codex rounds 1-2. Round-2 adoptions: validation-best/error-free/termination gates on `final` (overrides need journaled reasons); test metric computed from the checker-PARSED submission CSV (exact artifact parity); fail-closed ledger reads for final/run gates + final lockfile; per-iteration timeout (SIGALRM, journaled); 6h clock anchored to a journaled `start-run` marker; probe-before-seal ordering; pre-exec source provenance (crash paths keep sha+source, verified); suspicious-source scanner flags journaled; manifest sha in entries. Round-2 items OVERRULED under the declared cooperative trust model (same residual codex accepted on Track 3): out-of-process isolation, frame-walking/conditional-mutation attacks, raw-CSV rereads by solutions — detection-and-audit, not prevention. Earlier v0.2.0 adoptions: mechanical test-label stripping, evaluator tamper probes, sealed test predictions, enforced once-only final (+ crash-evidence marker + official checker), enforced convergence/cap/6h-ceiling with journaled overrides, complete journal provenance (harness sha, git state, dataset hashes, verbatim solution source). Proven end-to-end: iteration 2 at valid primary 0.6015; full final wiring proven on a scratch ledger (delta +0.0007 = published baseline; once-only and post-final refusals verified).
- Harness v0.5.0 awaiting codex round 5, then user freeze — the gate before the autonomous run. Round-4 adoptions: randomized log sanitized too; sanitized hashes ENFORCED by verify_hashes; official-run scoping (budget/convergence/best count from the start-run marker; prior entries = setup phase, resolving the setup-converged ledger); ledger-identity namespaces for seals/CSV; base fields + random-suffix ids on every entry type; raw-seconds gating; scanner catches path-join/read_csv forms; crash entries recover HYPOTHESIS from source. Memory upgrades added per user approval: tools/digest.py (session-start journal view) + mandatory reflection ritual in PLAN.
- Optimization budget: 0 of 50 OFFICIAL iterations used; 5 setup-phase iterations journaled (baseline reproductions across harness versions). Budget, convergence and the 6h clock all start at `start-run`. Historical test-split status, verbatim: "not pristine — a bounded organizer-reference exception".

## Standing rules (never violate)
1. Never edit: kuairand-starter-kit/** (organizer ground truth — evaluate.py is the sole scoring authority), README.md, Project/manifest.json, Project/results/** (harness-written only), .claude/**. After freeze: Project/harness/.
2. The agent develops on train + validation ONLY. Test labels are on disk but off-limits until the one final scoring of the designated submission.
3. Every iteration goes through the harness and gets journaled: hypothesis, full verbatim solution source + hash (diffs derivable between consecutive entries), validation metrics, errors/recovery, tokens, wall-clock. The journal is a required competition deliverable.
4. Check LESSONS.md before working — it contains organizer-verified dead ends that must never be retried.
5. Plain language to the user; explicit user "go" before starting the autonomous run.

## Next actions (in order)
1. Sol (codex) checkpoint review of iterate.py → user approves → freeze (same ceremony as Track 3; bind the review to a committed sha, per Track 3's lesson 13).
2. User says go → autonomous run on the hypothesis queue (PLAN.md) → convergence → final submission + test-scored once.

## Blocked / needs user
- Harness freeze approval (after it's built + Sol-reviewed).
- The "start the run" go.
