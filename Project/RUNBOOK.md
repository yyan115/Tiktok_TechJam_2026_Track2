# Owner runbook — Track 2 irreversible controller

This document is for the owner. Run commands from the repository root. Harness
tests do not authorize benchmark research, an official run, hidden-test
scoring, or submission. Those phases still require the user's separate,
explicit approval.

Run the official controller from the repository's primary checkout: `.git`
must be the real Git common directory directly under the repository root.
Linked worktrees, partial/promisor clones, replacement refs, grafts, and
alternate object stores fail closed.

## 1. Verify the harness without consuming anything

```bash
python3 -m unittest discover -s Project/harness/tests -v
python3 Project/harness/iterate.py check
python3 Project/harness/iterate.py log
```

The optional bubblewrap and Unix-socket integration tests use synthetic data:

```bash
TRACK2_TEST_BWRAP=1 python3 -m unittest \
  Project.harness.tests.test_sandbox.SandboxExecutionTests -v
TRACK2_TEST_UNIX_SOCKET=1 python3 -m unittest \
  Project.harness.tests.test_outer_authority.RpcIntegrationTests -v
TRACK2_TEST_RESEARCHER_BWRAP=1 python3 -m unittest \
  Project.harness.tests.test_researcher_shell_hardening.ResearcherShellExecutionTests -v
```

No test above loads or scores KuaiRand.

## 2. Freeze the evidence before `start-run`

Before the official clock starts:

1. Build the real research bank from the committed templates. Every catalog
   claim must resolve to committed note bytes and a bounded line range.
2. Build `Project/research/portfolio.json` from the portfolio template. It
   must contain at least four distinct mechanism families and a frozen opening
   order.
3. Commit the bank, portfolio, harness, manifest, and organizer inputs. Verify
   that the protected files and repository state are exactly the intended
   freeze. Do not pre-create official candidate/card destinations.
4. Keep the research bank and portfolio compact enough for their policy caps.
   The hashes establish byte identity, not the truth of a claim.

Only after a fresh explicit authorization for this phase, the owner starts the
run directly:

```bash
python3 Project/harness/iterate.py start-run \
  --portfolio Project/research/portfolio.json
```

This performs the bounded portfolio review and creates the immutable run ID,
start revision, six-hour clock, and policy state. There is no continuation,
force, early-final, not-best, or prose-override option.

## 3. Create three private external locations

Use canonical absolute paths outside the repository and outside the
controller's authority directory. The examples below are placeholders; replace
them with real owner-controlled paths. Each parent and the dedicated agent
home must be a real, owner-owned mode-0700 directory. Do not place any private
run path beneath `/usr` or `/etc`, which are mounted as host runtime inputs.

```bash
install -d -m 700 /absolute/private/track2-runtime
install -d -m 700 /absolute/private/track2-workspaces
install -d -m 700 /absolute/private/track2-agent-home
python3 Project/tools/init_researcher_workspace.py \
  --workspace /absolute/private/track2-workspaces/run-01
```

The initializer copies the exact frozen portfolio and creates a durable
logical workspace ID. The service and shell independently derive a physical
binding from the canonical path, directory inode/device, and manifest bytes;
copying or replacing a workspace does not preserve that live capability.

Use a fresh agent home for this run. Never point `--agent-home` at a normal
home containing unrelated credentials or conversations. Preserve it during
the run because it contains the client's exactly-once request state.

## 4. Start the external controller service

In a dedicated owner terminal:

```bash
python3 Project/harness/controller_service.py \
  --socket /absolute/private/track2-runtime/controller.sock \
  --audit-log /absolute/private/track2-runtime/controller.audit.jsonl \
  --workspace /absolute/private/track2-workspaces/run-01
```

The socket and write-ahead log must share the same durable mode-0700 runtime
parent. The runtime, researcher workspace, repository, and HMAC authority
state must all be disjoint. Do not put the audit log in an ephemeral directory
that disappears on reboot.

The service exposes only `log` and `run`. It serializes all transitions with
the inner controller lock, commits exact staged bytes through a Git
compare-and-swap, reserves audit space for both halves of every consuming
request, and refuses automatic replay after an indeterminate transition.
Every fixed Git write disables replacement/lazy-fetch semantics and forces
objects and references through `fsync`, overriding repository or host defaults.

## 5. Launch the restricted researcher

In a second owner terminal, use the exact Claude executable recorded in
`Project/harness/claude_runtime.json`:

```bash
python3 Project/harness/researcher_shell.py \
  --socket /absolute/private/track2-runtime/controller.sock \
  --agent-executable /absolute/path/to/claude \
  --agent-home /absolute/private/track2-agent-home \
  --workspace /absolute/private/track2-workspaces/run-01
```

The launcher verifies the executable, socket, logical workspace ID, physical
binding, frozen portfolio, mount graph, and repository/service binding before
starting Claude. The researcher gets five narrow tools:

- `hash_solution`: hash one safely staged solution without consuming an
  attempt;
- `log`: read bounded state and aggregate outcomes;
- `run`: submit one exact solution/card pair;
- `retry`: resend only the already-persisted exact pending request;
- `recover`: return only a locally persisted completed response.

It cannot call `start-run`, `final`, the raw controller, Git, Bash, web tools,
the journal, evaluator, results, manifest, or datasets.

## 6. Researcher iteration protocol

For every proposed attempt, the researcher must:

1. Call `log`. Stop immediately on `TERMINAL_DUE` or `TERMINAL`.
2. Read the exact frozen portfolio from the `frozen_portfolio.path` returned by
   `log`. `portfolio_summary` is only a bounded convenience summary.
3. Write one new candidate under `Project/solutions/` in the restricted
   workspace.
4. Call `hash_solution` and put that exact SHA-256 into a new attempt card
   based on the frozen template.
5. Bind the card to the exact run ID, next iteration, family, prior outcome
   IDs, mechanism, falsifier, expected range, and research-bank claims.
6. Call `run` once. The service captures stable bytes, commits them itself,
   performs deny-only review, evaluates in isolation, records the outcome, and
   atomically latches the first terminal condition.
7. Compare the aggregate result with the declared prediction/falsifier, update
   run memory honestly, and call `log` again.

The first four accepted attempts follow the frozen opening order. Corrected
bytes receive a new content-derived semantic review request; exact rejected
bytes keep their sticky verdict. Review attempts and reviewer failures are
both capped. Failed official executions count as non-improvements.

## 7. Crash and retry rules

Do not delete or edit the controller WAL, HMAC journal, agent request state,
workspace manifest, or committed artifacts.

### Ambiguous client transport

If the `run` tool loses its response, do not change either staged file and do
not create another request. Call `retry`. It resends the exact persisted bytes
and request ID. `recover` is local-only and never replays a pending request.

If the service answers `recovery_required: true`, stop all new attempts. A
global pending latch blocks different request IDs as well.

### Owner reconciliation of a pending WAL request

Stop the researcher and controller service. Copy the exact request ID from the
error response or WAL record. Inspect the private inner state:

```bash
python3 Project/harness/iterate.py _admission-state
```

If it reports an open attempt, close that exact interrupted inner transaction
first:

```bash
python3 Project/harness/iterate.py recover
```

Then reconcile the outer WAL offline using the same paths and exact request
ID:

```bash
python3 Project/harness/controller_service.py \
  --socket /absolute/private/track2-runtime/controller.sock \
  --audit-log /absolute/private/track2-runtime/controller.audit.jsonl \
  --workspace /absolute/private/track2-workspaces/run-01 \
  --resolve-pending 0123456789abcdef0123456789abcdef
```

The resolver verifies the request, start revision, run/workspace identity,
portfolio, research bank, controller state, Git parent/child relation, and
exact artifact hashes. It never re-executes the candidate.

After reconciliation completes, restart the service against the same runtime
and workspace, relaunch the researcher with the same dedicated agent home, and
call `retry`. The service returns the now-durable WAL response and the client
caches it locally; it does not execute the candidate again.

### Torn final WAL line after power loss

With the service stopped, repair only a non-newline final suffix:

```bash
python3 Project/harness/controller_service.py \
  --socket /absolute/private/track2-runtime/controller.sock \
  --audit-log /absolute/private/track2-runtime/controller.audit.jsonl \
  --workspace /absolute/private/track2-workspaces/run-01 \
  --repair-torn-audit-tail
```

The repair validates the complete prefix and refuses newline-complete
corruption. Afterward, resolve any surviving pending request as above.

### Lost researcher workspace

Keep the service stopped. If needed, repair a torn WAL suffix and close an open
inner attempt first, in that order. If a pending WAL request survives, recover
its exact logical workspace ID from validated durable evidence, create a new
empty target, and rebuild:

```bash
python3 Project/tools/init_researcher_workspace.py \
  --workspace /absolute/private/track2-workspaces/rebuilt-run-01 \
  --recover-workspace-id 0123456789abcdef0123456789abcdef
```

While the service is still stopped, run `--resolve-pending` against the rebuilt
workspace. Only after offline reconciliation completes should the service be
restarted against that path. The rebuilt directory has a new physical binding,
so ordinary live replay from the old workspace namespace is intentionally
refused; the new binding begins in a separate client-state namespace.

If there is no pending WAL request, there is nothing to reconcile and normal
workspace initialization may mint a fresh logical ID.

## 8. Terminal state and final

At the first convergence prefix, 50-attempt cap, or six-hour deadline, the
controller writes an immutable terminal snapshot containing the selected
validation-best eligible checkpoint. Later attempts are unreachable and later
scores cannot change that target.

After the researcher has stopped, stop the service cleanly and inspect:

```bash
python3 Project/harness/iterate.py log
```

Only after a separate explicit authorization for hidden-test finalization, the
owner runs:

```bash
python3 Project/harness/iterate.py final
```

`final` accepts only the checkpoint frozen in the terminal snapshot. It writes
`final_pending` before raw test labels are opened, verifies the exact
submission with the organizer checker, and scores once. Re-invoking `final`
can return a matching durable cached result after a response-loss crash; it
cannot rescore or choose another checkpoint. A pending marker without a valid
durable result fails closed and requires inspection.

Record genuine behavior-changing human help with:

```bash
python3 Project/harness/iterate.py intervention \
  --describe "what changed and why"
```

An intervention is evidence only. It grants no authority and cannot reopen a
terminal run.

## 9. Honest deployment limits

- The researcher must reach its model provider, so this local launcher shares
  host networking. Disabled web/MCP tools and absent local datasets do not
  mechanically prevent a deliberately malicious process from downloading a
  public benchmark. For a stronger boundary, use a dedicated VM or a
  provider-only egress proxy.
- The dedicated agent home is writable for authentication and client state.
  Use a fresh isolated home and preserve it for recovery; do not mix it with
  unrelated sessions.
- The writable researcher workspace and agent home have bounded prelaunch file
  checks but no live filesystem-block or inode quota. Put them on a dedicated
  quota-limited filesystem or VM if host-storage exhaustion is in scope.
- File and process limits are per process. Without an owner-configured cgroup,
  aggregate fork/resource exhaustion remains a host-availability risk.
- A malicious same-UID process, ptrace-capable peer, root, or kernel compromise
  is outside this boundary. A dedicated OS account or VM is the stronger
  deployment choice.
- Exact hashes and independent review establish identity, provenance, and a
  bounded skeptical check. They do not prove that a scientific claim is true
  or that a model will improve the score.

These are disclosed residuals, not prompt-enforced guarantees.
