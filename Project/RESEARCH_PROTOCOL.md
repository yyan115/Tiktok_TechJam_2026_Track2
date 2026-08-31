# Track 2 research protocol

This protocol turns research quality into evidence and bounded choices before
an official iteration is spent. It does not give a language model authority
over competition rules.

## 1. Build the evidence bank before the clock

Use primary sources where available: official dataset documentation and code,
original papers, and official library documentation. A public solution may be
recorded as a public solution, but must not be presented as primary evidence.
Do not search for or ingest hidden-test labels.

For each useful claim, create one compact Markdown note containing:

- source title, authors/owner, URL or publication identifier, and access date;
- the narrow claim actually supported;
- why it may matter for KuaiRand-Pure's within-user GAUC/nDCG@5;
- limitations, competing explanations, and leakage/compute risks;
- the smallest result that would falsify its proposed use.

Catalog only the exact bounded line range that supports the claim. The
controller verifies committed bytes, note SHA-256, line bounds, topics, and
the whole bank snapshot. Those hashes establish identity and provenance, not
truth or relevance.

The real `catalog.json` is intentionally absent during harness construction.
It is created only when benchmark-research work is separately authorized, then
committed and frozen before `start-run`.

## 2. Preserve a diverse opening portfolio

Create at least four mechanism families from independent first-pass reasoning.
Group by causal mechanism, not names, hyperparameters, or wording. Examples of
different surface syntax implementing the same representation or loss remain
one family.

Each family must have:

- a metric-linked causal claim;
- a smallest decisive experiment;
- a concrete falsifier;
- a finite, narrow expected-primary delta range;
- known leakage, overfit, implementation, and compute risks;
- exact research-bank claims and topic coverage.

Freeze an `opening_order`. Official attempts 1–4 must follow its first four
distinct families exactly. This is the executable version of the diverse-
portfolio lesson from the OpenAI proof prompt: keep incompatible routes alive
long enough to reveal real evidence rather than letting every worker imitate
the first attractive idea.

## 3. Bind every attempt

Stage one candidate and one JSON card in the restricted workspace. The outer
controller captures stable bytes and creates the exact Git commit. The card
binds:

- immutable run ID and exact next iteration;
- family ID, candidate path, and candidate SHA-256;
- mechanism, change, hypothesis, falsifier, and why-now decision;
- expected-primary range and exact research claims;
- every previous official outcome ID in chronological order.

The restricted tool surface is exactly `hash_solution`, `log`, `run`, `retry`,
and `recover`. Hash the staged solution before authoring the card; no tool can
start, finalize, unlock, or rewrite controller state.

A family not in the opening portfolio needs a structured extension identifying
its nearest existing families, truly novel topics, and material mechanism
delta. The controller registers that definition at first use and refuses later
redefinition.

## 4. Adversarial preflight is a brake, not the driver

The trusted controller sends exact portfolio/card/code/evidence bytes directly
to GPT-5.6 Sol with no tools and strict JSON output. Two independent calls are
mandatory. If their accept/reject classes disagree, a third call decides the
majority.

The reviewer attacks:

- disguised duplicates and unsupported novelty;
- evidence that does not actually support the claimed mechanism;
- leakage, hidden-test use, memorized benchmark answers, and invalid splits;
- unfalsifiable or vacuous experiments;
- code/card mismatch and provenance gaps;
- any plan that assumes a stop-rule or finalization override.

It does not choose the next idea or approve exceptions. Exact verdicts are
sticky, approvals are single-use, and at most three conclusive semantic
reviews may be spent on the portfolio or any one attempt stage. Failed
reviewer calls are separately capped at three instead of being retried
forever. This prevents review-shopping and availability loops while allowing
bounded correction of a real rejection.

## 5. Learn from outcomes without collapsing the search

After each official outcome:

1. include its exact entry ID in the next card;
2. compare the observation with the declared falsifier and expected range;
3. mark unsupported mechanisms blocked;
4. update the family registry and re-rank remaining experiments;
5. cross-pollinate only when the evidence identifies a concrete transferable
   mechanism;
6. prefer informative ablations over cosmetic tuning.

Failed executions are official evidence and count as non-improvements. A
preflight rejection is journaled but consumes no model-evaluation iteration.
Transport failures are bounded and do not become semantic rerolls.

## 6. Stop means stop

The deterministic controller scans every official prefix. At the first
convergence, 50-attempt cap, or six-hour deadline it writes a terminal row and
refuses all further attempts. Fable, Sol, the owner, and any natural-language
instruction lack an override path. A separately authorized rerun would be a
new immutable run, never a reopened terminal journal.
