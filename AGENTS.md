# Project Working Agreement

## Mandatory user authorization gate

- Treat every user message as read-only unless it contains an intentional, affirmative, uppercase `GO` that authorizes the requested modification.
- Authorization applies only to the task and scope being discussed when that `GO` is given. It is not blanket permission for unrelated changes.
- A `GO` from an earlier user message does not override a later user message that changes the task or requests additional modifications without its own explicit authorization.
- `NO GO`, quoted examples, hypothetical or conditional mentions, substrings, and discussion about the word `GO` are not authorization.
- Without valid authorization, limit work to discussion, reasoning, read-only inspection, and read-only commands. Do not edit files, install dependencies, generate artifacts, run mutating builds or scripts, or otherwise change local or external state.
- If the user appears to request a modification but omitted `GO`, discuss the request or remind them of this gate instead of assuming permission.

## Harness engineering principles

- Prefer the smallest harness that is correct, understandable, reproducible, and auditable.
- Reduce and understand the existing system before adding improvements.
- Add abstractions, services, containers, sandboxing, caching, concurrency, or instrumentation only when they solve a demonstrated requirement or measured bottleneck.
- Keep every reported metric traceable to its dataset inputs, evaluation code path, run configuration, and saved raw result. Clearly label estimates, projections, mocks, and synthetic values; never present them as measured results.
- Validate changes incrementally with small, inspectable tests before committing time to a full run.
- Use the failed Pure attempt and the Track 3 implementation as evidence to learn from, not as designs to copy blindly. Preserve useful correctness improvements while rejecting unnecessary complexity and unverifiable behavior.
- For the current submission, prioritize producing a trustworthy result on the approximately 1k-example dataset. Treat the approximately 27k-example dataset as out of scope unless the user explicitly changes that decision.
