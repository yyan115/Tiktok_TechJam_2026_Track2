# Standing orders — Track 2

1. Read `Project/RESEARCHER_BRIEF.md`, `Project/PLAN.md`, and
   `Project/RESEARCH_PROTOCOL.md`, then call controller `log`. Read
   `Project/memory/STATE.md` and `Project/memory/LESSONS.md` only if you created
   them during this designated run; a fresh run intentionally starts without
   legacy memory.
2. Do not begin benchmark research grinding, `start-run`, an official attempt,
   `final`, or submission without the user's fresh explicit **go** for that
   phase. Harness work and synthetic tests are not permission to run KuaiRand.
3. After freeze, never edit organizer files, `README.md`,
   `Project/manifest.json`, `Project/results/**`, `.claude/**`, trusted harness
   or reviewer code, the review schema, or the frozen research bank.
4. During the official run, work only in the restricted researcher shell. Use
   only the `track2_controller` MCP tools (`hash_solution`, `log`, `run`,
   `retry`, `recover`); the researcher has no `start-run`, `final`, journal,
   evaluator, dataset, Git history, or rules capability.
5. Candidate code receives training feedback only. Validation and test feedback
   columns are zero inside the candidate sandbox. Raw hidden-test labels are
   loaded only by the once-only final path after `final_pending` is durable.
6. Use only the frozen research bank during the official run. Its hashes prove
   which bytes were cited, not that a claim is true. Do not fetch benchmark
   data, enable web/arbitrary MCP tools, or use a permission-bypass flag.
7. Independent Sol review is a deny-only semantic check. Two calls must agree;
   a third breaks disagreement. It cannot unlock a terminal state, and an exact
   rejected request has a sticky verdict.
8. If the controller MCP `log` tool reports `TERMINAL`, stop. There is no reasonable-
   circumstances, owner, reviewer, or model override.
9. `.claude` hooks are advisory ergonomics, not authority. The external
   controller service, authenticated journal, immutable snapshots, and OS
   sandbox are the enforcement layers.
10. Keep explanations plain. Record genuine behavior-changing human help with
    the owner-only `intervention` command.
