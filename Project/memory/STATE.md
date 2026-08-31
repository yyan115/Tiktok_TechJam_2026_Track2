# STATE — read this first in every session

Updated: 2026-08-30 (harness hardening; no official rerun)

## Current state

- The first completed KuaiRand-Pure run is untouched. The user asked the
  organizers whether a clean rerun is allowed.
- All replacement work is isolated in
  \`/tmp/tiktok-track2-harness-hardening\` on branch
  \`track2-harness-hardening\`, based on the clean pre-run commit. The original
  dirty repository and the separate rerun worktree are untouched.
- No replacement \`run_start\`, official attempt, candidate score, hidden-test
  score, final artifact, submission, or real reviewer API call has occurred.
- The old general override path is gone. Earliest convergence, cap, deadline,
  and final-pending states are structurally irreversible.

## Implemented architecture

- External owner-private HMAC journal authority and immutable input snapshot.
- Exact state-machine replay with strict row shapes, adjacency, chronology,
  review-consensus replay, family registration, and terminal/final binding.
- Outer Unix-socket controller service exposing only \`log\` and \`run\` to the
  researcher; owner-only \`start-run\`, \`final\`, and \`intervention\`.
- Restricted researcher mount graph with no Git metadata, controller, results,
  manifest, dataset, raw labels, or normal home.
- Frozen primary-source research bank and four-family opening-order protocol.
- Direct no-tools GPT-5.6 Sol quorum: two calls, third only on disagreement,
  sticky exact verdicts, bounded corrections.
- Candidate bubblewrap with mandatory network namespace, immutable exact code
  and runtime files, resource/deadline limits, and prediction-only output.
- Candidate data contains the standard logs, user features, and basic video
  features with exact schemas. Validation/test feedback is zero. The unresolved
  random log and month-aggregate engagement statistics are withheld.

## Work still in progress

- Finish client request crash recovery and socket/response authentication.
- Finish documentation and owner-private installed-composition test.
- Run the complete unit suite, real synthetic bubblewrap test, Unix-socket
  integration, syntax/diff checks, and final independent hostile audit.
- Freeze only after the user has reviewed the final diff and remaining
  limitations.

## Honest limitations

- The outer Fable process currently shares host networking because its model
  client needs provider access. Local benchmark data and built-in web/MCP
  capabilities are removed, but arbitrary host egress is not a proof against a
  deliberately malicious download. Provider-only egress or a dedicated VM is
  required to make that boundary mechanical.
- Hashes prove exact identity and provenance, not that a cited research claim
  is true. Sol supplies semantic skepticism; deterministic policy owns rules.
- Same-UID root/ptrace/kernel compromise is outside this cooperative process
  boundary. The design does not claim hostile-machine isolation.

## Non-negotiable next gate

Complete harness verification, report the exact outcome, then stop. Await a
fresh explicit user **go** before real benchmark research, \`start-run\`, or any
official attempt.
