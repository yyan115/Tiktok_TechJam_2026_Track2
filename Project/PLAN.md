# Plan of Record — Track 2: Autonomous ML Research Agent (approved 28 Aug 2026)

## The system in one line

The same hardened cross-reviewed loop as Track 3, pointed at a recommender-system pipeline: Fable (Claude) proposes and codes improvements → the organizers' own scoring script judges them → every iteration is auto-journaled (hypothesis, code diff, metrics, errors/recovery — the journal IS a required competition deliverable) → Sol (codex) reviews at checkpoints → the user retains final authority. Target: a fully autonomous run, zero manual interventions.

## The task (plain words)

Rank each user's videos so the ones they'll watch long sit on top. Dataset: KuaiRand-Pure (Kuaishou short-video logs; label = `long_view`). Beat the official Factorization Machine baseline: **hidden-test primary 0.5946** (validation 0.6016). Score = your absolute improvement over that number on the hidden test. Real ceiling is **0.8645**, not 1.0 (27.1% of test users have no positive labels — unwinnable rows). All CPU; the baseline trains in ~40 s.

## What is graded (and how we serve each dial)

- **Primary metric (in Technical Execution, 35%):** validation-best submission, test-scored once. Convergence rule: stop when validation primary improves ≤0.002 over 3 straight iterations; hard caps 50 iterations / 6 h.
- **Robustness (same bucket):** errors must be recovered from, not avoided — every failure + recovery auto-logged.
- **Autonomy (Impact, 20%):** graded by counting manual interventions. Target: zero. The journal carries an intervention counter.
- **Feasibility (15%):** LLM tokens + wall-clock, coarse tiers, only among baseline-beaters — journal meters both.
- **Innovation (20%):** judged on WHAT the agent chose to try and why — the journal's hypothesis field, grounded in fresh web research per idea (MLE-STAR style).

## Integrity rules

- `kuairand-starter-kit/` code (evaluate.py, data.py, baseline.py, submit.py) is organizer ground truth: hash-pinned, never edited. Same deny-rules + guard-hook setup as Track 3.
- **Hidden-test discipline (mechanically enforced by the harness):** solutions receive test rows with labels stripped — they cannot see a test label. Development feedback is validation-only; each run's test predictions are SEALED unscored, and `final` scores one designated sealed artifact exactly once (once-only + post-final refusals enforced, overrides journaled). The three organizer reference test scores reproduced at setup are their own published numbers (their explicit reproduce-the-baseline instruction) — no agent-designed solution's test metrics are ever revealed before the final.
- Promotion: an iteration becomes current-best on validation improvement; Sol audits at checkpoints (harness before freeze; implausible jumps; final submission), never per iteration.

## The hypothesis queue (seeded from the organizers' own tested guidance)

Their published dead-ends (pre-loaded into LESSONS — never retry): more feature fields (no gain), bigger embeddings (no gain), pure user-side features (mathematically zero effect under within-user ranking).
Their ranked untried directions, our starting order:
1. **Ranking-aligned loss** (in-user listwise softmax or pairwise BPR) — their top bet and ours: the metric is a ranking metric, the baseline trains a pointwise classifier.
2. **User behavior sequences** (DIN/SIM-style interest modeling) — timestamps exist per interaction; completely unused today.
3. **Multi-task heads** over the other 11 feedback signals (click, like, play_time_ms, …).
4. **Watch-time modeling** (censored regression à la CWM, KDD'24).
5. Model swaps (DeepFM/DCN/xDeepFM) — deprioritized; capacity is proven not the bottleneck.
6. Time features / train→test drift.
7. The randomized-exposure log as an unbiased extra validation set (also an innovation flourish).
The agent re-orders this queue from its own results, does a fresh web search before each new idea, ends with an agent-designed ensemble of top diverse candidates (rank averaging), and never repeats a journaled failure.

## Stages

- **Stage 0 — Rails** (done at setup): starter kit unzipped + hash-pinned; dataset downloaded (gitignored); all three official baselines reproduced within published seed-noise (random 0.4757 / pop 0.5715 exact / FM 0.5953 vs 0.5946±0.0008); wiki + guardrails in place.
- **Stage 1 — Iteration harness:** one command runs a candidate pipeline, scores validation via the organizers' evaluate.py, appends the journal (hypothesis, diff, metrics, errors, tokens, wall-clock), tracks current-best, enforces the convergence rule and leak-guard. Sol reviews it, user approves, freeze — same ceremony as Track 3.
- **Stage 2 — The run:** the agent iterates the hypothesis queue autonomously to convergence or caps.
- **Stage 3 — Final:** designate validation-best, score test once, generate + `--check` the submission CSV, package (report from the journal, resource totals, intervention count).

## Authority

User holds: harness freeze approval, the "start the run" go, and sign-off on the final submission. Everything else is autonomous by design — that's the graded feature.
