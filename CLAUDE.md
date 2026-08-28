# Standing orders (every session, read before doing anything)

1. Read `Project/memory/STATE.md` first (a SessionStart hook also injects it). Then `Project/memory/LESSONS.md` (contains organizer-verified dead ends — never retry those) and `Project/PLAN.md`. Log decisions in `Project/memory/DECISIONS.md`, lessons as they happen.
2. NEVER edit: `kuairand-starter-kit/**` (organizer ground truth; `evaluate.py` is the sole scoring authority), `README.md`, `Project/manifest.json`, anything in `Project/results/` (harness-written only), `.claude/**`. After the Stage-1 freeze: `Project/harness/`.
3. The agent develops on train + validation ONLY. Test labels exist on disk but are off-limits until the single final scoring of the designated submission. Training data must never cross date 20220421.
4. Every optimization iteration goes through the harness and is journaled (hypothesis, diff, validation metrics, errors/recovery, tokens, wall-clock) — the journal is a required competition deliverable and the autonomy evidence.
5. The user requires plain language (no jargon walls) and an explicit "go" before starting runs. Sol (codex, fresh `codex exec`, read-only) reviews at checkpoints only; its failures never block work.
6. Sister project: `../Tiktok_TechJam_2026_Track3` — same architecture, shared history in its `Project/memory/DECISIONS.md`.
