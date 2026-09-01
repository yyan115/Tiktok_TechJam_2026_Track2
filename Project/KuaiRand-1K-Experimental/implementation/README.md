# TikTok TechJam Track 2 — KuaiRand-1K

> [!NOTE]
> **Post-run archive status:** this is the implementation snapshot frozen before
> the campaign. The autonomous research loop subsequently ran for 13 attempts / 11
> scored iterations and was externally aborted when the submission deadline
> arrived, before its frozen 15-scored-iteration floor. It never opened hidden
> test. See the [campaign evidence report](../README.md). References below to a
> campaign that “has not started” describe the moment this code was frozen.

This branch is a small autonomous ML harness for the 1K benchmark. It is designed
to make the researcher capable without giving it control of scores, stopping, or
hidden labels. The real campaign has not started and this repository contains no
claimed 1K score yet.

## Architecture

```text
one launch
   |
   v
Supervisor -> Researcher AI -> EDA + web research -> proposal/code
     |                                  ^                 |
     |                                  |                 v
     +------------ Critic AI (advice; hard veto only for objective faults)
                                        |
                                        v
Controller -> isolated counted execution -> exact validation evaluator -> ledger
     |                                                              |
     +-- fixed caps/convergence -------------------------------------+
     |
     +-- terminal -> frozen validation-best checkpoint -> one hidden final
```

The researcher can read trusted evidence but cannot change it. Candidate code has
the full training data, label-free validation impressions, the locked ML runtime,
and CUDA. It has no network, validation labels, hidden-test data, ledger, or
controller access. Final inference receives test features and the frozen checkpoint
but not training data, which prevents post-terminal retraining.

## What happens autonomously

After one launch, the supervisor starts the six-hour clock, makes fresh researcher
sessions read the complete task/reference material, executes bounded training-only
EDA chosen by the researcher, requires real web tool use and a source-grounded
plan, obtains an independent critic review, and enters the experiment loop. Every
real candidate execution counts, including crashes. The researcher receives
sanitized errors and aggregate metrics, reflects, and repairs or changes direction
without asking the owner. The controller stops mechanically and finalizes the
earliest validation-best checkpoint.

The researcher is OpenAI GPT-5.6 Sol at maximum reasoning effort. The independent
critic is Claude Opus 5 at maximum effort. Both identities are recorded in their
agent traces.

There is no continuation flag, owner signature ceremony, AI-owned attempt counter,
uncounted shadow training, mutable dashboard, or general-purpose candidate shell.

## Frozen benchmark boundary

- Train: 2022-04-08 through 2022-04-21.
- Public validation: 2022-04-22 through 2022-04-28.
- Hidden test: 2022-04-29 through 2022-05-08.
- Label: `long_view`; metrics: GAUC and nDCG@5; primary is their mean.
- Hard caps: 50 attempts and 21,600 seconds.
- Failures count against caps but do not advance convergence.
- Random-exposure and month-aggregated statistic files are excluded until their
  competition legality and temporal provenance are explicitly resolved.

The downloaded data and generated safe view are ignored by Git. Their exact hashes,
row counts, and visible schemas live in `datasets/derived_1k/manifest.json`.

## Readiness

Run the non-scoring readiness check:

```bash
.venv/bin/python -m tools.doctor
```

It verifies raw and derived hashes, the absence of cached test labels, package
versions, CUDA arithmetic, real Bubblewrap isolation in both attempt and final
modes, and isolated Codex and Claude authentication. It neither invokes a model nor
scores a benchmark split.

Development regression tests:

```bash
.venv/bin/pytest
```

## Starting the official campaign

Do not launch until `epsilon`, `window_scored_iterations`, and
`minimum_scored_iterations` have been deliberately chosen and frozen. Copy
`config/campaign.template.json` to a local policy, fill those three values, then use
one command:

```bash
.venv/bin/python -m harness.supervisor \
  --run-dir runs/kuairand-1k \
  --workspace workspace/kuairand-1k \
  --policy config/campaign.local.json
```

That command begins official wall-clock accounting before research and EDA. Running
it again after an infrastructure interruption reconstructs state from the event
ledger and resumes; it does not reopen a terminal run.

Authoritative outputs are under the ignored run directory: atomic events, exact
candidate snapshots and diffs, validation metrics, frozen checkpoints, agent tool
traces, read-only research evidence, and the once-only final submission.
