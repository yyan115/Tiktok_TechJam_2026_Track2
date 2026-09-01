# KuaiRand-1K Experimental Campaign — Deadline-Aborted

This directory preserves a secondary Track 2 campaign built after the completed
KuaiRand-Pure submission. It used a substantially redesigned, deliberately
compact harness and produced a promising validation trajectory, but it was
externally aborted when the submission deadline arrived, before its frozen
15-scored-iteration floor. It is published as an honest engineering and autonomy
record, **not** as a completed benchmark submission.

## Status

| Field | Recorded value |
|---|---:|
| Attempts | 13 |
| Scored iterations | 11 |
| Candidate errors | 2 |
| Best attempt | 13 |
| Best validation GAUC | 0.6909917126 |
| Best validation nDCG@5 | 0.6569251681 |
| **Best validation primary** | **0.6739584404** |
| Validation rows / users | 2,524,980 / 978 |
| Hidden-test evaluations | **0** |
| Completed submission artifact | **No** |

The supplied starter kit did not contain an official KuaiRand-1K baseline, so
no baseline delta is claimed. The campaign ledger has no `run_terminal` or
`final_scored` event. Hidden labels were never opened.

## Autonomous operation

Within the active campaign, this was a fully autonomous ML research loop. After
launch, the GPT-5.6 Sol researcher read the task evidence, chose and interpreted
controlled EDA, performed web/source research, wrote an evidence-grounded plan,
answered a Claude Opus 5 critic, designed and coded every candidate, interpreted
each measured outcome, and independently chose the next repair or experiment.

The participant did not choose models, features, fixes, ensemble weights, or
individual attempts; did not edit candidate code; and did not redirect the
scientific search. Status questions did not alter the loop. Codex supervised
process health and trusted-state recovery without making the ML decisions. The
only participant action that changed campaign state was the final external abort
when there was no longer enough submission time to complete four more scored
iterations and a hidden final.

## Validation trajectory

| Attempt | Status | Validation primary | Main outcome |
|---:|---|---:|---|
| 1 | error | — | Exact target reconstruction was impossible because `play_time_ms` was absent at inference |
| 2 | error | — | LightGBM LambdaRank exceeded its query-size limit |
| 3 | scored | 0.451936 | Repaired query chunking; sparse metadata ranker remained weak |
| 4 | scored | 0.513620 | Uniform pointwise binary learning beat LambdaRank |
| 5 | scored | 0.401536 | User-balanced weighting was strongly negative |
| 6 | scored | 0.619466 | Removing sparse metadata and target-rate features produced a large gain |
| 7 | scored | 0.656265 | Strictly backward session features produced another large gain |
| 8 | scored | 0.657632 | Added a causal predecessor-engagement proxy |
| 9 | scored | 0.659720 | Within-user rank ensemble improved both-metric balance |
| 10 | scored | 0.660045 | Selected rounds and blend weights on chronological internal validation |
| 11 | scored | 0.673376 | Target-free user-by-context personalization was the major breakthrough |
| 12 | scored | 0.673376 | Internal gate rejected unsupported higher-order crosses and exactly retained attempt 11 |
| 13 | scored | **0.673958** | Ordered causal session history raised GAUC and produced the best primary |

All displayed values come from the controller-written event ledger. Attempt 13
also passed checkpoint-only replay: the attempt and replay predictions share
SHA-256
`78a120d73da57399886cfbbf27933e9bf0ae4d1dbc327d506755f79f923b3b0e`.

## Architecture and why it was better

```text
frozen policy + one authorized launch
                 |
                 v
        AI campaign supervisor
          |             |
          v             v
 researcher AI <---- critic AI
          |
          +--> controlled EDA + source-grounded plan
          +--> proposal + exact candidate snapshot
                              |
                              v
 deterministic controller --> isolated candidate --> validation metrics
          |                         |                       |
          |                         +-- no network          |
          |                         +-- no hidden labels     |
          |                                                 v
          +-- append-only ledger <-- checkpoint replay <-- reflection
```

The redesign focused on enabling good ML work while retaining narrow,
deterministic authority:

- one frozen benchmark and stopping policy, with no researcher-owned override;
- controlled EDA and source-grounded planning before experiments;
- separate GPT-5.6 Sol researcher and Claude Opus 5 critic roles;
- candidate execution in a network-isolated Bubblewrap process;
- hidden labels, evaluator state, ledger, and stopping state kept outside the
  candidate sandbox;
- a full CPU ML runtime including LightGBM, Polars, SciPy, and scikit-learn;
- controller-only validation scoring and exact checkpoint replay;
- counted failures that do not advance the scored convergence window; and
- immutable candidate snapshots, append-only events, and earliest-best final
  selection.

Unlike the abandoned hardening fork, vNext did not make security synonymous
with an incapable researcher. Candidate code received a frozen threaded CPU ML
stack, persistent caches and checkpoints, aggregate validation feedback, and
bounded error diagnostics. The authority boundary withheld only what the
researcher did not need to own: hidden labels, raw evaluator state, the ledger,
stopping policy, and final selection.

### Design claims exercised by the campaign

| Mechanism | Observed evidence |
|---|---|
| Mandatory research before experimentation | The ledger binds a completed EDA result, evidence-linked research portfolio, and critic verdict before the first candidate |
| Autonomous failure recovery | Attempts 1 and 2 failed for unrelated schema and library-contract reasons; the researcher changed mechanism, repaired query construction, and reached a scored attempt 3 without participant ML guidance |
| Evidence-based pruning | The researcher abandoned harmful user weighting, removed sparse target-rate features, and followed the session/context signals that measured positively |
| Honest incumbent retention | Attempt 12's internal gate gave unsupported higher-order crosses zero weight and reproduced attempt 11 exactly instead of claiming progress |
| Artifact-to-score binding | All 11 scored candidates passed checkpoint replay; attempt 13 reproduced prediction SHA-256 `78a120d73da57399886cfbbf27933e9bf0ae4d1dbc327d506755f79f923b3b0e` |
| Recovery without authority drift | A branch interruption and temporary critic outage were recorded and recovered without changing frozen policy, candidate identity, or controller-owned scoring |

The strongest evidence is therefore not merely the final validation number. It
is that the system navigated failures, negative results, null additions, model
selection, and infrastructure trouble while preserving its authority boundary.

## Honest limitations and interruptions

- The submission deadline arrived after attempt 13, at 11 scored iterations.
  The campaign was externally aborted because there was not enough time to
  reach the predeclared minimum of 15 and complete the once-only final. It
  therefore never reached a normal controller terminal state.
- Hidden test was not evaluated, so `0.6739584404` is validation-only and must
  not be described as an official or final score.
- An unrelated shared-worktree switch briefly interrupted orchestration between
  attempts 6 and 7. The supervising Codex session restored the frozen branch,
  verified the authority hashes and ledger, and continued with the already
  agent-written candidate. No attempt, score, prompt, or scientific decision
  changed; this was AI-supervised infrastructure recovery, not participant
  steering of the ML loop.
- The Opus critic hit a temporary provider session limit for attempts 11 and 12.
  The controller recorded the outage, retried twice, applied deterministic
  integrity checks, and continued rather than deadlocking. Critic review resumed
  normally for attempt 13.
- Repeated public-validation use can overfit even when internal chronological
  screening is used. Only a hidden-test result could establish transfer.
- Search remained concentrated on scalable LightGBM context, personalization,
  causal-history, and ensemble models; deeper GPU sequence families were not
  reached.

These are why this package says **deadline-aborted and promising**, not complete
or winning.

## Evidence map

- [`SUMMARY.json`](SUMMARY.json) — compact machine-readable status and result.
- [`AUTONOMY_EVIDENCE.json`](AUTONOMY_EVIDENCE.json) — sanitized aggregate of
  all 15 successful researcher sessions and nine critic invocations; raw traces
  are excluded because they contain provider/session metadata.
- [`MANIFEST.sha256`](MANIFEST.sha256) — SHA-256 inventory of the complete
  compact package.
- [`campaign/events/`](campaign/events) — complete controller-written event
  ledger for all 13 attempts.
- [`campaign/evidence/`](campaign/evidence) — EDA results, accepted research
  plan, controller state, critic feedback, and recorded outages.
- [`campaign/policy.json`](campaign/policy.json) — stopping policy frozen before
  the run.
- [`campaign/benchmark.json`](campaign/benchmark.json) — frozen dataset and
  metric contract.
- [`best-attempt/candidate.py`](best-attempt/candidate.py) and
  [`best-attempt/proposal.json`](best-attempt/proposal.json) — exact attempt-13
  candidate and hypothesis.
- [`best-attempt/checkpoint/`](best-attempt/checkpoint) — manifest, training
  diagnostics, inference schema, and prediction audit. Large model weights and
  prediction arrays are intentionally omitted from Git; their identities and
  replay result remain in the ledger.
- [`implementation/`](implementation) — the complete minimal harness, guidance,
  tools, and tests from commit `34048e5`. Its source remains the frozen campaign
  implementation; its README now begins with a clearly marked post-run archive
  note before the original operating document.

The complete implementation history remains on the
[`track2-1k-vnext`](https://github.com/yyan115/Tiktok_TechJam_2026_Track2/tree/track2-1k-vnext)
branch. Dataset files are excluded from Git. Full model-agent CLI transcripts
are also excluded because they contain provider/session metadata and repeated
embedded source; controller events and structured critic findings are retained.

Return to the repository [overview](../../README.md) or the detailed
[Track 2 report](../REPORT.md).

## Quick inspection

```bash
jq '.controller | {attempts, scored_iterations, best_attempt, best_primary, final_scored, terminal}' \
  campaign/evidence/STATE.json

jq '{attempt, status, metrics, checkpoint_check_status}' \
  campaign/events/000029.json
```
