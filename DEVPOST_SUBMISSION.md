# Autonomous Recommender Researcher

An auditable autonomous ML research loop for TikTok TechJam 2026 Track 2.

Repository: <https://github.com/yyan115/Tiktok_TechJam_2026_Track2>

> **Important convergence disclosure:** The organizer-default cumulative
> stopping rule became true after iteration 4. The researcher then used a
> general-purpose override for iterations 5–15, despite an independent reviewer
> flagging that decision. The run had no predeclared minimum-iteration floor.
> After reviewing the logs, the organizers expressly instructed us to submit
> the iteration-15 validation-best result and document the control failure.
> Read the
> [complete account and corrective design](https://github.com/yyan115/Tiktok_TechJam_2026_Track2/blob/main/Project/CONVERGENCE_DISCLOSURE.md).

## Inspiration

Machine-learning engineers repeatedly perform the same research loop: inspect
the current result, form a hypothesis, change the pipeline, run an experiment,
and decide what to try next. Track 2 asked whether an agent could own that loop
for a real recommendation benchmark while requiring minimal human direction.

We were interested in more than producing a higher score. An autonomous
researcher is useful only if its result is trustworthy, reproducible, and easy
to audit. That led us to separate the creative researcher from a deterministic
harness that controls data access, evaluation, budgets, provenance, and final
submission.

## What it does

Autonomous Recommender Researcher lets a coding agent conduct an iterative ML
campaign on KuaiRand-Pure. For each iteration, the researcher proposes a
hypothesis, writes a complete candidate pipeline, runs it through the frozen
harness, receives validation-only metrics, records what it learned, and chooses
the next experiment.

The official campaign completed 15 scored iterations. It reproduced the
organizer's Factorization Machine baseline; tested pairwise and listwise losses,
behavioral crosses, and auxiliary feedback; recovered from a NumPy state bug;
introduced causal attention over user engagement history; checked seed
robustness; diagnosed negative interaction between useful components; and
finished with a score-level rank ensemble.

The submitted model blends:

- a five-seed dual-gate DIN-lite/FM ensemble;
- a five-seed pointwise FM ensemble; and
- a watch-ratio auxiliary FM.

Every component trains only on the official 8–21 April 2022 training window.
The validation-best checkpoint was selected before the final CSV was evaluated
once on the test labels.

## How we built it

The system has two roles. A hosted Claude coding-agent session acted as the
researcher, while a Python harness acted as the controller. OpenAI Codex was
used as an independent read-only reviewer. Candidate models call no external
API.

The controller:

- supplies train and labeled validation data while mechanically zeroing test
  labels before candidate code runs;
- verifies hashes for the dataset, evaluator, manifest, and harness;
- records each hypothesis, exact source, source SHA-256, GAUC, nDCG@5, runtime,
  estimated token use, errors, override state, and sealed-prediction hash in an
  append-only JSONL journal;
- seals test predictions before checkpoint selection; and
- records a `final_pending` marker before allowing the single final test
  evaluation.

We used Python 3.14.7, NumPy, the organizer-provided KuaiRand-Pure starter kit
and evaluation code, Git, GitHub, and standard Linux shell tools. The only
dataset was KuaiRand-Pure's standard interaction log and basic video features;
we used no external training data, pretrained weights, GPU, PyTorch, pandas, or
scikit-learn. The exact NumPy version was not captured, so we do not invent one.

The run used 15 of 50 iterations, 3,781.3 seconds (1.05 hours) of agent
wall-clock, 2,278.2 seconds of candidate execution time, an estimated 466,000
LLM tokens, and 0 GPU-hours. This is a solo submission; AI systems are tools,
not team members.

## Challenges we ran into

The most important challenge was governance, not modeling. With the default
`epsilon = 0.002`, `N = 3` cumulative criterion, convergence became true after
iteration 4 because the first hypothesis family had plateaued. The harness
nevertheless exposed `--continue-past-convergence`, which the researcher used
11 times while untested families remained. An independent reviewer correctly
flagged this as a rule violation. Logging the overrides made them visible, but
visibility did not grant the researcher authority to relax its own stopping
instruction. We disclose the original finding, rationale, organizer response,
and proposed fix in full rather than retroactively calling the behavior a
custom stopping policy.

We also encountered a genuine implementation failure. Iteration 2's BPR
fine-tune scored below the fallback it was supposed to preserve. The researcher
traced the contradiction to NumPy arrays that aliased the saved champion state,
deep-copied the state in iteration 3, and exactly restored the baseline. The
buggy scored iteration remains in the journal and counts against the budget.

Finally, KuaiRand-Pure moves in thousandths. Many plausible changes were null
or harmful, so we had to distinguish repeatable information gains from seed
variation without using test feedback.

## Accomplishments that we're proud of

The final validation and one-time test results both improved over the published
baseline:

| Result | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Published validation baseline | 0.667400 | 0.535700 | 0.601600 |
| Validation-best iteration 15 | **0.669094** | **0.536891** | **0.602992** |
| Validation delta | **+0.001694** | **+0.001191** | **+0.001392** |
| Published test baseline | 0.661000 | 0.528200 | 0.594600 |
| Submitted test result | **0.662562** | **0.529195** | **0.595879** |
| Test delta | **+0.001562** | **+0.000995** | **+0.001279** |

The final CSV contains all 170,588 test predictions and passed the organizer's
header, row-count, alignment, and finite-score checker. Its SHA-256 is
`45b221e5d3ded00daeb3cf5412f928ec2a8eee3a576275c76432a6d2c5001549`.

We are also proud that the run produced inspectable evidence rather than only a
headline metric: zero logged human-caused behavior interventions, all 15 exact
candidate sources and per-iteration diffs, recovery evidence, one
`final_pending` plus one `final` journal record, no post-final development, and
a prominent account of the system's own control failure.

## What we learned

The main modeling lesson was that new information mattered more than a new loss
on the old representation. Pairwise BPR, listwise training, and an explicit
user-author affinity feature did not improve the baseline. Causal target-aware
attention over engaged user history produced the largest gain.

The watch-ratio auxiliary objective helped as a shared-representation
regularizer, but its own head was not a better scorer. Jointly training the
sequence and auxiliary mechanisms caused negative interference, while blending
separately trained models at score level worked. Seed ensembling then made the
sequence gain more robust.

The main systems lesson was that auditability and authority are different.
Append-only logs can reveal an unsafe decision, but the controller must prevent
that decision rather than merely record it.

## What's next for Autonomous Recommender Researcher

The first change is to make `epsilon`, `N`, and any minimum-iteration floor
immutable controller state declared before a run. The researcher-facing
override would be removed, and reviewer control violations would become
blocking events requiring external resolution.

We would also isolate candidates in a capability-limited process instead of
running cooperative code in the harness process, add a recurring strategy
critic to challenge weak experiment ordering, capture an exact environment
lock and provider-side token telemetry, and explore stronger tree ensembles,
deeper sequential recommenders, and more principled watch-time objectives.

The repository contains the detailed report, complete journal, all candidate
sources, mechanically derived iteration diffs, final CSV, setup instructions,
and the full convergence disclosure needed to inspect or reproduce the work.
