# Plan of record — Track 2 autonomous recommender research

## Current outcome

The v0.5 override-based harness is retired. It could report convergence while
still exposing a continuation option to the researcher. The replacement makes
the first convergence, iteration-cap, or wall-time trigger an irreversible
controller transition.

This branch contains harness work only. No replacement official run, candidate
evaluation, final score, or submission has begun. The original completed run
and the user's original worktree are not modified.

## Roles and authority

```text
owner: freeze/start/final
          |
external controller service: bound workspace + exact Git/run capability
          |
restricted Fable workspace: code/cards + five narrow tools only
          |
Sol semantic quorum: deny-only review of exact frozen bytes
          |
candidate bubblewrap: fixed inputs -> finite prediction arrays only
          |
trusted parent: validation scalar + immutable journal/terminal latch
```

Fable chooses research directions and writes code. Sol challenges semantic
quality and rule fit. Neither model owns competition state. Only deterministic
controller code can consume an attempt, latch a stop, or promote an artifact.

The researcher cannot call `start-run` or `final`, read Git history, edit the
controller, inspect the journal, or access local datasets. The owner retains
those lifecycle actions outside the researcher sandbox.

Every live request carries both a durable logical workspace ID and a physical
binding derived independently by the shell and service from canonical path,
directory identity, and manifest bytes. A bounded two-phase WAL is fsynced
before any consuming transition, reserves completion space in advance, and
forbids automatic replay of an indeterminate request.

## Organizer rules as state transitions

- Required benchmark: KuaiRand-Pure.
- Development feedback: trusted validation GAUC, nDCG@5, and their primary
  mean; never validation row labels inside candidate code.
- Convergence: after any three consecutive official outcomes, stop at the
  earliest prefix whose validation best did not improve by more than 0.002.
  Failed official attempts are non-improvements.
- Backstops: 50 official attempts or six hours from the immutable `run_start`.
- Selection: validation-best eligible checkpoint in the first terminal prefix,
  with deterministic tie handling.
- Hidden test: score the selected sealed predictions once, only after a durable
  `final_pending` row.

There is one production journal and no alternate ledger, force flag, early
final, not-best selection, or continuation override. Later improvement cannot
erase an earlier convergence.

## Research quality before evaluation

Before `start-run`, build and commit a primary-source research bank. Each
catalog claim resolves to an exact note hash and bounded line range. Hashes
bind identity, not truth; the note must preserve source, supported claim,
caveats, and a falsifying experiment.

The committed portfolio must contain at least four mechanistically distinct,
evidence-grounded families and a frozen opening order. The first four official
attempts must follow that order exactly, preventing early collapse onto one
attractive idea. Later families require an explicit extension describing their
nearest existing families, novel topics, and mechanism delta. Expected-score
ranges must be finite, ordered, bounded, and informative.

Every attempt card binds the run, exact next iteration, family, candidate path
and SHA-256, evidence claims, falsifier, and every prior official outcome ID in
chronological order.

## Independent semantic review

Portfolio, attempt, and final packets are reviewed through direct no-tools
GPT-5.6 Sol calls with strict structured output. Two independent calls are
always made; a third is made only when their accept/reject classes disagree.
The majority result is replay-validated from the recorded calls.

An exact request has a sticky verdict. A corrected artifact gets a new,
content-derived request and must cite the rejected review. At most three
conclusive semantic reviews are permitted for a portfolio or one attempt
stage. Failed reviewer calls are separately capped at three; transport or
schema failures never become approval and cannot loop forever.

Review is deliberately deny-only. Approval is consumed immediately by the
matching transition and cannot be reused or interpreted as an organizer-rule
exception.

## Data and execution boundaries

At `start-run`, the controller creates an external immutable copy of every
manifest-pinned organizer input and binds it to the current manifest bytes.
The candidate derivative is narrower:

- the two standard logs, static user features, and basic video features only;
- exact approved CSV schemas and content hashes;
- all feedback fields zero from 2022-04-22 onward, covering validation and test;
- no random-exposure log while its split use remains unresolved;
- no month-aggregated video engagement statistics, which can include future
  outcome information;
- no raw data, repository, journal, sealed output, or network.

Candidate execution uses immutable reviewed bytes, a new user/mount/PID/IPC/UTS
and network namespace, fixed runtime artifacts, resource limits, and an
absolute deadline. Its public output is fixed-size finite validation/test
prediction arrays plus row counts. Candidate stdout, stderr, exit details,
free-form metadata, and hypothesis text do not cross the boundary.

The trusted parent computes validation metrics and seals test predictions
without scoring them. Raw test labels are first opened in the once-only final
path.

## Honest threat model

The external HMAC journal authority, exact Git/object checks, controller
service, and bubblewrap boundaries defend against accidental edits, model
disobedience, stale state, and many same-user tampering paths. `.claude` hooks
are advisory only.

The outer Fable process still needs network access to reach its model provider.
The filesystem boundary removes local benchmark data and built-in web/MCP
tools should be disabled, but shared host networking alone cannot prove that a
deliberately malicious process did not download public KuaiRand files. For
mechanical egress enforcement, deploy the researcher in a dedicated VM or
provider-only egress proxy. This residual is disclosed; it is not described as
solved by prompts.

## Lifecycle

1. Finish implementation, primary-source review, hostile tests, and installed
   composition tests using synthetic state only.
2. Present the exact branch/diff and residual limitations to the user.
3. After user approval, populate the real research bank and portfolio on a
   standalone clean repository and freeze their exact commit.
4. Only after a separate explicit user **go**, the owner invokes `start-run`.
5. Fable iterates through the restricted service until the first terminal row.
6. Only after a separate final authorization does the owner invoke `final`.
7. Submission remains a separate human decision.

## Meter honesty

The journal records exact controller/attempt wall times and every reviewer
call's model identity and reported usage. Agent-session token totals must come
from provider records when the report is assembled and be labeled unavailable
when they cannot be independently recovered.
