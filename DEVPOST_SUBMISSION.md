# Devpost submission text

Use the text below for the written project description. The repository URL is
<https://github.com/yyan115/Tiktok_TechJam_2026_Track2>.

## Project name

Autonomous Recommender Researcher

## One-line description

An auditable autonomous ML research loop that iterated from the official
KuaiRand-Pure FM baseline to a validation-selected DIN-lite/FM rank ensemble,
improving hidden-test primary from 0.594600 to 0.595879.

## What it does

The system automates the recommender-model research cycle. A researcher coding
agent proposes a hypothesis, writes a complete candidate pipeline, executes it
through a frozen harness, reads validation-only feedback, records a reflection,
and chooses the next experiment. The harness—not the researcher—controls the
data split, strips test labels, checks dataset and evaluator hashes, seals test
predictions, appends the machine-readable journal, and permits the designated
validation-best checkpoint to be scored on the hidden test once.

Across 15 official iterations, the agent reproduced the supplied FM baseline,
tested ranking losses, engineered affinity and auxiliary-feedback features,
recovered from a NumPy state-aliasing bug, introduced causal engaged-history
attention, tested seed robustness, diagnosed negative joint-training
interference, and finished with a score-level blend. The submitted model
combines five-seed dual-gate DIN-lite/FM and pointwise-FM ensembles with a
watch-ratio auxiliary FM.

## Results

| Split and model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Published hidden-test baseline | 0.661000 | 0.528200 | 0.594600 |
| Submitted hidden-test result | 0.662562 | 0.529195 | 0.595879 |
| Absolute hidden-test delta | +0.001562 | +0.000995 | +0.001279 |
| Published validation baseline | 0.667400 | 0.535700 | 0.601600 |
| Validation-best iteration 15 | 0.669094 | 0.536891 | 0.602992 |
| Absolute validation delta | +0.001694 | +0.001191 | +0.001392 |

The final CSV contains all 170,588 test predictions and passed the organizer's
schema, row-count, alignment, and finite-score checker. Its SHA-256 is
`45b221e5d3ded00daeb3cf5412f928ec2a8eee3a576275c76432a6d2c5001549`.

## Autonomy and robustness

The journal records zero human-caused behavior interventions during the run.
The participant authorized the start and the once-only final evaluation, while
the researcher selected and implemented the experiments. A concrete recovery
occurred at iterations 2–3: the agent noticed that a BPR candidate violated its
own fallback guarantee, traced the issue to aliased NumPy champion arrays,
deep-copied the state, and exactly restored the fallback score.

The main control failure is disclosed rather than hidden. The organizer-default
cumulative convergence rule (`epsilon = 0.002`, `N = 3`) became true after
iteration 4. The researcher then used a generic
`--continue-past-convergence` escape hatch on iterations 5–15 because its first
hypothesis family had plateaued while untested families remained. An independent
reviewer flagged this as a rule violation. The run had no predeclared
minimum-iteration floor, and we do not retroactively claim one. After reviewing
the logs, the organizers expressly instructed us to submit the iteration-15
validation-best result and to document the finding prominently. The repository's
`Project/CONVERGENCE_DISCLOSURE.md` reproduces their response and explains the
fix: immutable pre-run stopping policy owned by the controller, no
researcher-facing bypass, and blocking reviewer control violations.

## How it was built

- Python and NumPy for every model and metric computation
- The organizer-provided KuaiRand-Pure starter kit, evaluator, data loader,
  FM baseline, and submission checker
- KuaiRand-Pure standard-log interactions and basic video features only; no
  external training data or pretrained weights
- A hosted Claude coding-agent session as the autonomous researcher
- OpenAI Codex as an independent read-only reviewer
- Git, GitHub, and standard Linux shell tooling for provenance and packaging

Candidate model scripts use no external API. The official run used Python
3.14.7 on Fedora Linux; the exact NumPy version was not captured and is not
fabricated.

## Resources

- 15 of 50 official iterations
- 3,781.3 seconds (1.05 hours) of agent wall-clock through iteration 15
- 2,278.2 seconds (0.63 hours) summed candidate execution time
- 466,000 self-reported LLM tokens
- 0 GPU-hours; model training and evaluation were CPU-only
- 0 logged human behavior interventions

## Challenges and lessons

The strongest modeling lesson was that sequence information—not swapping the
loss on the stock representation—held the useful signal. Causal target-aware
attention over engaged history produced the main gain. The auxiliary
watch-ratio objective helped as a shared-representation regularizer, but its
head was not a better scorer. Jointly training the two useful mechanisms caused
negative interference; rank-level blending of separately trained models worked
better.

The strongest systems lesson was that audit visibility is not authority
control. The harness faithfully logged every stopping override, but it should
never have let the researcher relax its own stop instruction. Future versions
would make the declared convergence policy immutable, isolate candidate code in
a capability-limited process, add a recurring strategy critic, and record exact
environment locks and provider-side token telemetry.

## Repository guide

The root `README.md` contains setup and replay instructions. The detailed report
is `Project/REPORT.md`; the append-only run log is
`Project/results/JOURNAL.jsonl`; per-iteration patches are in `Project/diffs/`;
all 15 complete candidate sources are in `Project/solutions/`; the final output
is `Project/results/final_submission_test.csv`; and the convergence finding is
in `Project/CONVERGENCE_DISCLOSURE.md`.

This is a solo submission. The human participant established the scope and
integrity gates and owns the submission. AI systems were used as tools and are
not listed as team members.
