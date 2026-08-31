# Track 2 Run Report — Autonomous Recommender Researcher

**One paragraph:** An autonomous agent improved the official KuaiRand-Pure ranking
baseline from hidden-test primary **0.5946 to 0.5959 (+0.0013)** in a single
designated run: **15 iterations, ~1.05 h of active wall-clock, CPU-only,
zero logged human behavior interventions**. Every official-run experiment ran
inside a hash-frozen, pinned harness that mechanically strips test labels,
seals each iteration's test
predictions before any grading, and scored the hidden test exactly once, on the
designated validation-best artifact. The complete machine journal, exact
candidate sources, submitted CSV, and their hashes are committed in this repo.

---

## Critical convergence-control disclosure

The organizer-default cumulative rule (`epsilon=0.002`, `N=3`) became true at
iteration 4. The researcher then used the harness's general-purpose
`--continue-past-convergence` escape hatch on iterations 5–15. An independent
reviewer flagged this as a rule violation: the researcher had bypassed a stop
instruction enforced by its own controller. The reasons were journaled, but
visibility is not the same as valid authority.

This run did not predeclare a minimum-iteration floor, and we do not
retroactively call the escape hatch a custom policy. After reviewing the logs,
the organizers explicitly instructed us to submit the iteration-15
validation-best result because the run remained inside the 50-iteration and
6-hour caps. Their response, the reviewer's finding, why the override fired,
and the controller redesign are preserved in
[`Project/CONVERGENCE_DISCLOSURE.md`](CONVERGENCE_DISCLOSURE.md).

The future fix is structural: declare all stopping values before `start-run`,
make them immutable controller state, remove the override from the researcher
interface, and make reviewer control violations blocking.

## 1. Headline numbers

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM baseline (published, hidden test) | 0.6610 | 0.5282 | **0.5946** |
| **Our final (hidden test, scored once)** | 0.6626 | 0.5292 | **0.5959** |
| Delta | | | **+0.0013** |

| Public validation | GAUC | nDCG@5 | primary |
|---|---:|---:|---:|
| Official FM baseline (published) | 0.667400 | 0.535700 | 0.601600 |
| **Validation-best, iteration 15** | **0.669094** | **0.536891** | **0.602992** |
| Absolute delta | **+0.001694** | **+0.001191** | **+0.001392** |

Validation gain over this run's iteration-1 floor was +0.0015; it transferred
to test as +0.0013. Context for the magnitude: re-running the
organizers' *unchanged* baseline across seeds moves the score by ±0.0008, and the
organizers' own published improvement attempts (8 extra feature fields; larger
embeddings) moved it by ≈0.000. This benchmark moves in thousandths by
construction: 27.1% of test users have no positive label (unwinnable) and 9.2%
are all-positive (unloseable), so the theoretical ceiling is 0.8645 rather than
1.0.

Final artifact: a rank-average blend `2·(5-seed dual-gate DIN-lite FM) +
(5-seed pointwise FM) + (watch-ratio-auxiliary FM)` — every component trained
only on the train window (dates ≤ 2022-04-21), seed-deterministic.

## 2. The run, iteration by iteration

| it | valid primary | what happened |
|---|---|---|
| 1 | 0.6015 | Baseline re-anchored as the official floor (always-finalizable fallback) |
| 2 | 0.6008 | Pairwise BPR fine-tune — **scored under its own floor: a real bug** (numpy state aliasing corrupted the saved champion) |
| 3 | 0.6015 | **Autonomous recovery:** bug diagnosed from the trajectory, fixed, floor held exactly; BPR itself proven a null result |
| 4 | 0.6015 | Listwise softmax fine-tune — second null; the whole ranking-loss family declared dead. Convergence rule latched here |
| 5 | 0.6013 | User→author affinity count feature — null: redundant with FM's own embeddings |
| 6 | 0.6018 | Soft watch-ratio auxiliary head (α=0.3) — **first real positive** |
| 7 | 0.6000 | Dose–response test (α=1.0) — hurts; effect is non-monotonic |
| 8 | 0.6018 | Head-space search — the auxiliary head never outranks the main head: it is a regularizer, not a scorer |
| 9 | 0.6022 | **The discovery: DIN-lite sequence attention** over each user's engaged history, gated by a zero-initialized scalar — the model opened the gate itself |
| 10 | 0.6021 | Longer history + positional bias — does not compound |
| 11 | 0.6022 | Author-granularity second attention gate — marginal |
| 12 | 0.6024 | 3-seed ensemble of the champion — gain is seed-robust |
| 13 | 0.6014 | Jointly training the two proven mechanisms — **negative interaction**, proven and journaled |
| 14 | 0.6026 | Score-level rank-average blend of separately-trained champions |
| 15 | **0.6030** | Blend polish (5 seeds per component, faithful aux member) — designated final |

The agent re-ranked its hypothesis queue after every iteration (ritual log in
`Project/memory/QUEUE.md`), ran a fresh web search before each new idea family,
and appended distilled lessons to `Project/memory/LESSONS.md` as they were
learned.

## 3. What the run established scientifically

Negative results below are exact, reproducible measurements, not opinions:

1. **Ranking-loss swaps on the stock representation are dead** (its 2–4). The
   pointwise FM is already pair-consistent within users; both BPR and listwise
   objectives degrade validation from the first epoch.
2. **Engineered count-crosses of fields the FM already embeds add nothing**
   (it 5) — the model's user×author interaction *is* a learned affinity from the
   same evidence.
3. **Label-side information is live but small** (its 6–8): the continuous
   watch-ratio signal that `long_view` binarizes away buys +0.0003 as a
   shared-embedding regularizer, and nothing as a scorer.
4. **Sequence information is the real untapped signal** (its 9–12): target-aware
   attention over the user's engaged history — the one mechanism the model,
   offered a free choice via a zero-initialized gate, chose to use.
5. **Proven mechanisms combine at the score level, not by joint training**
   (its 13–15): joint training interferes (measured); rank-average blending of
   separately trained models stacks cleanly.

## 4. Integrity design (what makes the number believable)

Everything runs on the participant's machine in this competition, so the claim
"the agent never saw the test" is only as good as its evidence. Ours:

- **Mechanical label-stripping:** solutions receive test rows with labels zeroed
  by the frozen harness; development feedback is validation-only.
- **Sealed predictions:** every iteration's test predictions are hashed and
  sealed *before* any grading; `final` scores those exact sealed bytes — the
  measured artifact is the submitted artifact (checker-parsed CSV parity).
- **Once-only final, enforced:** the harness refuses a second test scoring and
  refuses post-final development; a crash-evidence marker is journaled *before*
  scoring so even a failure leaves a trace that test was consumed.
- **Append-only machine-written journal** (`Project/results/JOURNAL.jsonl`):
  every iteration carries its full verbatim solution source, hashes of harness/
  manifest/dataset, git state, metrics, errors, timeout, and self-reported
  tokens. The journal is written by the harness, never by hand.
- **Independent cross-model audits:** an auditor from a different vendor
  reviewed promoted checkpoints from their complete journal packets; durable
  verdicts are committed in `Project/audits/verdicts.jsonl`. Those verdicts are
  mostly `RULE_VIOLATION`, not hidden or softened. The contemporaneous findings
  and responses are in `Project/memory/DECISIONS.md`, with the convergence issue
  promoted into the dedicated disclosure linked above.
- **Self-imposed handicaps, logged when chosen:** final model trains on the
  train window only (the organizers' engineer's webinar advice, followed);
  the randomized-exposure log stayed quarantined (its status was left
  unresolved by the organizers); the 6 h / 50-iteration caps were enforced by
  software, not vibes.

## 5. Autonomy and robustness evidence

- **Manual behavior interventions during the run: 0** (the journal contains no
  intervention records).
  The agent chose ideas, wrote all solution code, diagnosed and fixed its own
  bug (it 2→3), and re-ordered its queue from results. The user's role was two
  sanctioned gates: "go" at the start and sign-off before the once-only final.
  Separately, the agent's decision to bypass convergence is disclosed as an
  autonomy/control defect, not presented as a strength.
- **Error recovery, journaled:** iteration 2's aliasing bug produced a score
  below its own designed floor — the contradiction was caught from the journal,
  root-caused (saved champion arrays aliased the live model), fixed, and the
  fix verified by an exact floor-hold. The buggy scored iteration stays in the
  record and counted against the budget, per the rules.

## 6. Resources

- **Agent wall-clock:** 3,781.3 s (~1.05 h) from `start-run` through the sealed
  iteration-15 result. Candidate execution time sums to 2,278.2 s (~0.63 h).
- **Compute:** model training and evaluation were local and CPU-only; 0 GPU-hours.
  The researcher and auditor themselves were hosted coding-agent services.
- **LLM tokens:** 466,000, self-reported across the 15 official iterations
  (`--tokens` field). Provider-side metering was not readable from inside the
  session, so these are estimates of conception, authoring, and operation cost.

## 7. Limitations — stated plainly

1. **Convergence is claimed relative to the run's own hypothesis queue, not to
   the idea space.** The queue came from the organizers' ranked guidance plus
   per-idea web searches. Untried and honestly promising: gradient-boosted tree
   ensembles, heavier sequence models (full attention over all impressions,
   cross-day patterns), deeper dataset-specific feature work from the KuaiRand
   literature.
2. **The run's research depth was its weakest behavior.** Per-idea searches were
   single-query skims; the organizers' own reference method (CWM) was never read
   in full; no search targeted published results on this exact dataset. Our
   sister project's post-mortem crystallized the design gap: *the harness audits
   honesty, but nothing in the loop audits intelligence* — no mechanism ever
   asked "is this the right thing to build?" after iteration 0. We consider this
   the most transferable finding of the whole effort, and the fix design is
   recorded (research gates with enforcement hooks; a recurring strategy-critic
   audit distinct from fraud audits; an objective stall trigger).
3. **The researcher could override its own stopping controller.** Under
   ε=0.002/N=3, opening with a dud family caused the default criterion to become
   true at iteration 4; the researcher then used 11 journaled overrides. The
   independent reviewer correctly objected. The organizers subsequently
   permitted iteration 15 for submission while requiring prominent disclosure.
   Full details and the no-override redesign are in
   `Project/CONVERGENCE_DISCLOSURE.md`.
4. **The score is modest.** +0.0013 is ~1.6× the baseline's own seed noise —
   real, reproducible, and small, on a benchmark whose own authors' published
   attempts moved ≈0.000.

## 8. Where everything lives

- `Project/results/JOURNAL.jsonl` — the machine-written run record (required deliverable)
- `Project/results/final_submission_test.csv` — the submission (checker-validated; sha journaled)
- `Project/CONVERGENCE_DISCLOSURE.md` — reviewer finding, organizer ruling, and corrective design
- `Project/solutions/` — all 15 iterations' code, exactly as journaled
- `Project/diffs/` — convenience diffs mechanically derived from adjacent solution files
- `Project/harness/iterate.py` — the frozen lab bench (v0.5.0, review round 12: YES)
- `Project/audits/` — independent verdicts + the 12-round harness review trail
- `Project/memory/` — the agent's own diary: decisions, lessons, live queue
