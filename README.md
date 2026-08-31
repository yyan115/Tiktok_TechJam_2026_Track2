# Autonomous Recommender Researcher

TikTok TechJam 2026 · Track 2 · Required benchmark: KuaiRand-Pure

This project is an autonomous machine-learning research loop for recommender
systems. A coding agent chose hypotheses, wrote each candidate pipeline, ran the
organizer's evaluator, reflected on the validation result, and revised its next
experiment. A separate harness controlled data access, stopping state, test
prediction sealing, and the append-only run record.

The official run improved the published KuaiRand-Pure hidden-test primary score
from **0.594600 to 0.595879 (+0.001279)**. It used 15 scored iterations, 1.05
hours of agent wall-clock, CPU-only model training, and no logged human behavior
interventions.

> [!IMPORTANT]
> **Convergence-control disclosure:** the organizer-default cumulative
> criterion (`epsilon = 0.002`, `N = 3`) became true after iteration 4. The
> researcher then invoked a general-purpose `--continue-past-convergence`
> escape hatch for iterations 5–15 because only one hypothesis family had been
> exhausted. Independent audits flagged that control decision as a rule
> violation. This run did **not** predeclare a minimum-iteration floor, and we
> do not retroactively describe the escape hatch as one. After reviewing the
> logs, the organizers explicitly instructed us to submit the iteration-15
> validation-best result. The complete finding, rationale, organizer response,
> and corrective design are documented in
> [CONVERGENCE_DISCLOSURE.md](Project/CONVERGENCE_DISCLOSURE.md).

## Results

### Hidden test — evaluated once

| Model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline (published) | 0.661000 | 0.528200 | 0.594600 |
| **Iteration 15 (submitted)** | **0.662562** | **0.529195** | **0.595879** |
| Absolute delta | **+0.001562** | **+0.000995** | **+0.001279** |

### Public validation

| Checkpoint | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline (published) | 0.667400 | 0.535700 | 0.601600 |
| **Validation-best, iteration 15** | **0.669094** | **0.536891** | **0.602992** |
| Absolute delta vs published baseline | **+0.001694** | **+0.001191** | **+0.001392** |

The exact reproduction in official-run iteration 1 scored `0.667133` GAUC,
`0.535805` nDCG@5, and `0.601469` primary. Against that same-run checkpoint,
iteration 15 improved by `+0.001961`, `+0.001086`, and `+0.001524`,
respectively.

The final model is a score-level rank blend:

- five-seed dual-gate DIN-lite/FM ensemble;
- five-seed pointwise FM ensemble; and
- one watch-ratio auxiliary FM.

Every component trained only on the official training window, 8–21 April 2022.
Validation selected the checkpoint and blend; the final CSV was then evaluated
once against the test labels.

## Submission artifacts

- [Detailed run report](Project/REPORT.md)
- [Ready-to-paste Devpost description](DEVPOST_SUBMISSION.md)
- [Convergence and override disclosure](Project/CONVERGENCE_DISCLOSURE.md)
- [Machine-written JSONL journal](Project/results/JOURNAL.jsonl)
- [Final checker-validated submission CSV](Project/results/final_submission_test.csv)
- [Final iteration source](Project/solutions/s014_blend_polish.py)
- [All 15 candidate sources](Project/solutions)
- [Per-iteration code diffs](Project/diffs)
- [Organizer starter kit](kuairand-starter-kit)

Final artifact identity:

| Field | Value |
|---|---|
| Journal entry | `20260828-204207-8fb9a6` |
| Solution SHA-256 | `2323fcb0fc05ff59a3894f407aaa69fd6ebbb8aac3f91647e1df66714326f0f9` |
| CSV rows, excluding header | `170,588` |
| CSV SHA-256 | `45b221e5d3ded00daeb3cf5412f928ec2a8eee3a576275c76432a6d2c5001549` |

## How the system works

```text
Human start authorization
          |
          v
Researcher coding agent -----> hypothesis + candidate source
          ^                                |
          |                                v
reflection memory <----- validation metrics <----- frozen harness
                                                   |      |
                                                   |      +-- append journal
                                                   +--------- seal test scores
                                                               |
                                                               v
Human final authorization -> validation-best seal -> one test evaluation -> CSV
```

The harness, rather than the researcher, owns the sensitive boundaries:

- candidate code receives train, labeled validation, and test rows with test
  labels mechanically replaced by zero;
- the organizer's evaluator and dataset files are hash-checked;
- every attempt records its hypothesis, exact source and source hash, metrics,
  runtime, reported token use, audit flags, error state, and sealed-prediction
  hash;
- test predictions are sealed before final selection;
- the actual finalization selected the validation-best entry and wrote the
  organizer's CSV schema; and
- a `final_pending` record is written before scoring so a crash cannot silently
  permit a second test evaluation.

The trust model is cooperative rather than adversarial: candidate code ran in
the harness process, with detection and audit controls designed to prevent
accidental leakage. That limitation is stated explicitly in the report.

## Iteration trail

| It. | Validation primary | Outcome |
|---:|---:|---|
| 1 | 0.601469 | Reproduced the FM baseline and established a safe fallback |
| 2 | 0.600805 | BPR attempt exposed a NumPy champion-state aliasing bug |
| 3 | 0.601469 | Agent diagnosed and fixed the aliasing bug; BPR remained null |
| 4 | 0.601469 | Listwise loss remained null; default convergence became true |
| 5 | 0.601309 | User-author affinity feature was redundant |
| 6 | 0.601759 | Watch-ratio auxiliary target produced the first gain |
| 7 | 0.600038 | Stronger auxiliary weight hurt |
| 8 | 0.601759 | Auxiliary head worked as a regularizer, not a scorer |
| 9 | 0.602185 | DIN-lite engaged-history attention produced the main gain |
| 10 | 0.602058 | Longer history and positional bias did not compound |
| 11 | 0.602248 | Author-level attention added a marginal gain |
| 12 | 0.602445 | Three-seed sequence ensemble reduced variance |
| 13 | 0.601372 | Joint sequence/auxiliary training interfered negatively |
| 14 | 0.602636 | Score-level blending combined the useful mechanisms |
| 15 | **0.602992** | Five-seed blend polish; validation-best final checkpoint |

The full hypotheses and exact source bytes are in the journal. The convenience
patches in `Project/diffs/` were mechanically derived from adjacent committed
solution files after the run; they do not replace the journal as the source of
truth.

Files under `Project/memory/` are contemporaneous researcher notes and were
preserved rather than rewritten after the organizer ruling. Their earlier
shorthand estimates and defenses—including the stale `~471k` token estimate—
are superseded by the journal-derived figures and convergence disclosure in
this README.

## Autonomy and robustness

- **Human behavior interventions: 0.** The participant authorized the start and
  the once-only final, but did not redirect candidate behavior during the run.
  This follows the Track 2 webinar definition that only a human-caused behavior
  change counts as an intervention.
- **Recovery evidence:** iteration 2 scored below its own intended fallback.
  The researcher traced this to aliased NumPy arrays, deep-copied the champion
  state in iteration 3, and reproduced the fallback exactly.
- **Control failure disclosed:** the researcher could also invoke the generic
  convergence escape hatch. Reasons were recorded on all 11 uses, but logging
  an override did not make it a sound authority boundary. The independent
  reviewer objected, and the future design removes that capability from the
  researcher.
- **Final discipline:** no development runs occurred after finalization, and
  the journal contains exactly one `final_pending` and one `final` record.

## Resource use

| Resource | Recorded use |
|---|---:|
| Official iterations | 15 / 50 |
| Agent wall-clock to seal iteration 15 | 3,781.3 s (1.05 h) |
| Sum of candidate execution times | 2,278.2 s (0.63 h) |
| LLM tokens | 466,000, self-reported estimate |
| GPU use | 0 GPU-hours |
| Human behavior interventions | 0 |

Token totals are estimates entered per iteration because provider-side billing
telemetry was not available inside the agent session. Model training and
evaluation were local and CPU-only; the researcher and auditor were hosted
coding-agent services.

## Tools, libraries, APIs, and data

- Python 3.14.7 on Fedora Linux for the recorded run;
- NumPy for every benchmark model and metric computation;
- the organizer-provided KuaiRand-Pure starter kit, evaluator, baseline, data
  loader, and submission checker;
- KuaiRand-Pure standard-log interactions and basic video features only;
- a hosted Claude coding-agent session as the researcher;
- OpenAI Codex as an independent read-only auditor; and
- Git/GitHub and standard shell tooling for provenance and packaging.

Candidate model scripts call no external API and use no external training data
or pretrained weights. The exact NumPy package version was not captured in the
official journal, so it is intentionally not fabricated here.

## Setup

Python 3.9+ and NumPy are sufficient. The recorded run used Python 3.14.7.

```bash
git clone https://github.com/yyan115/Tiktok_TechJam_2026_Track2.git
cd Tiktok_TechJam_2026_Track2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cd kuairand-starter-kit
curl -L --fail \
  -o KuaiRand-Pure.tar.gz \
  https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
cd ..

python3 Project/harness/iterate.py sanitize-data
python3 Project/harness/iterate.py check
```

`sanitize-data` creates the candidate-safe file view: feedback columns on test
dates are zeroed while training and validation remain unchanged. `check`
verifies the organizer files plus raw and sanitized dataset hashes against
`Project/manifest.json`.

## Inspect and reproduce

Render a compact view of the preserved official journal:

```bash
python3 Project/tools/digest.py
```

Replay iteration 15 into an isolated scratch ledger without touching the
official record:

```bash
python3 Project/harness/iterate.py \
  --ledger /tmp/track2-replay.jsonl \
  run \
  --solution Project/solutions/s014_blend_polish.py \
  --tokens 0
```

Expected validation primary: approximately `0.602992`. On the recorded CPU
environment this candidate took about 10.6 minutes.

Validate the submitted CSV's schema, row count, alignment, and finite scores
without re-scoring the test labels:

```bash
python3 kuairand-starter-kit/submit.py \
  --data_dir kuairand-starter-kit/KuaiRand-Pure/data \
  --split test \
  --check \
  Project/results/final_submission_test.csv

sha256sum Project/results/final_submission_test.csv
```

The official production harness will refuse another final evaluation because
the once-only final has already been consumed.

## Limitations and what we would improve

1. The hidden-test gain is real but modest: +0.001279 primary.
2. The researcher-accessible stopping override was a governance defect even
   though every use was visible. The corrective design is detailed in the
   convergence disclosure.
3. Research breadth was shallow: searches were brief, the organizer-referenced
   CWM work was not studied deeply, and tree ensembles plus heavier sequential
   models were not attempted.
4. The harness protects against cooperative mistakes, not malicious same-process
   candidate code. A stronger design would execute candidates in a separate,
   capability-limited process.
5. LLM token use is estimated, and the exact NumPy version was not recorded.

## Contributions

This is a solo submission. The human participant chose the track, established
the authorization and integrity requirements, approved the official-run and
final-evaluation gates, and owns the submission. The researcher coding agent
proposed and implemented candidate experiments; Codex supplied independent
reviews. AI systems are listed as tools, not team members.

The copied Track 2 brief is preserved in
[docs/TRACK_2_PROBLEM_STATEMENT.md](docs/TRACK_2_PROBLEM_STATEMENT.md). The
workshop transcript and participant notes are in [MEETING-NOTES.md](MEETING-NOTES.md);
the transcript records the track owner's statement that a video was optional
provided the written report was sufficiently detailed.
