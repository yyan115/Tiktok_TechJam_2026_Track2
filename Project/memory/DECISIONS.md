# DECISIONS — plain-language diary of what we discussed and agreed

## 30 Aug 2026 — replace prompt authority with an irreversible controller

- The first official run demonstrated that the v0.5 design was unsafe: it
  correctly detected convergence but exposed a general continuation override,
  so Fable continued even after Sol identified the rules problem. A logged
  override is still a rule violation; transparency does not make it legal.
- The prior "cooperative trust model" ruling is superseded for competition
  authority and hidden-data boundaries. These are now enforced out of process.
- The latest Track 3 session and unfinished hostile authority-v4 review were
  inspected directly. Adopted: independent mechanism portfolios, frozen exact
  artifacts, concrete falsifiers, named adversarial checks, and a deny-only
  critic. Rejected: prose owner quotes, caller-selected modes, regex hooks as
  authority, asynchronous audits, reusable receipts, and semantic family IDs
  presented as machine proof.
- The OpenAI Cycle Double Cover prompt contributes search-management ideas, not
  a permission to persist: keep independent approaches alive, track mechanism
  families, block routes with theorem-strength gaps, require concrete evidence,
  and adversarially test named failure modes. Its "do not stop" instruction is
  explicitly scoped out whenever the deterministic controller is terminal.
- The new Track 2 architecture puts the rules inside the actual runner. There
  is one fixed journal and lock, no production overrides, an earliest-prefix
  terminal latch, exact committed card/code binding, a synchronous independent
  review invoked by the runner itself, crash recovery, and a final selected
  only from the frozen prefix.
- Candidate code now runs from immutable reviewed bytes in bubblewrap. Raw data,
  the repository, results, parent processes, and networking are absent. The
  trusted development process itself loads only sanitized data; raw labels are
  first loaded after final_pending. Candidate validation and test feedback are
  both zero. Primary-source KuaiRand documentation shows that the video
  statistics aggregate engagement across a month, so those outcome-derived
  features are withheld; the overlapping random log remains withheld pending
  an organizer answer.
- Track 3's later authority work was re-read rather than copied wholesale.
  Track 2 adopts bounded semantic-review corrections, informative expected
  ranges, installed-composition testing, and explicit authority roles. It does
  not adopt Track 3-specific lifecycle states or claim that a hook is a live
  gate.
- A frozen primary-source research bank now resolves exact cited note ranges.
  A diverse portfolio freezes the first four mechanism families and their
  opening order. Sol review uses two direct no-tools calls and a third only on
  disagreement; the controller recomputes the consensus and limits conclusive
  rerolls.
- The researcher is placed behind an outer service exposing only log/run RPCs.
  Request/reply identity, crash recovery, socket ownership, exact Git commits,
  and protected component bytes are checked outside the model process.
- Honest residual: the outer model client currently needs shared networking.
  Filesystem isolation and disabled web tooling do not prove resistance to a
  deliberately malicious public-dataset download. Provider-only egress or a
  dedicated VM is the mechanical deployment answer; prompts are not.
- This work is harness-only. No replacement portfolio research, candidate
  optimization, official iteration, or hidden-test evaluation is authorized by
  this decision.

## 28 Aug 2026 — research, plan approval, and setup

- **What this track is:** build an AI agent that autonomously improves a video-recommendation model — the agent writes the code, runs it, reads the scores, and decides what to try next, with per-iteration logs. Judged on: score improvement over the official baseline (hidden test set), how few human interventions the run needs, how failures are recovered, and how cheap the run is (LLM tokens + wall-clock).
- **Key realization:** this track's required deliverable is the same architecture the team already built and battle-tested for Track 3 (agent + trusted evaluator + machine-written journal + cross-model audits). Here it's even simpler: the evaluator (evaluate.py) is provided by the organizers, and everything runs on CPU — no GPU rental.
- **Research done before any code** (28 Aug, logged in the Track 3 session): the field's leading systems — OpenAI's MLE-bench, AIDE, Google's MLE-STAR (NeurIPS'25), Microsoft's R&D-Agent — reviewed; MLE-STAR's concrete tricks adopted (ablation-guided targeting, fresh web search per idea, final ensembling, mechanical leak-guard). Starter kit read file-by-file; dataset schema verified on kuairand.com; download link health-checked; an error in the problem statement's metrics row caught (shipped code is authoritative).
- **User decisions:** doing BOTH tracks (this one and Track 3). Same working rules as Track 3: plain language, explicit go before actions, user approves the harness freeze and the final submission, cross-review by codex ("Sol") at checkpoints only.
- **Setup executed:** starter kit unzipped and hash-pinned; 47 MB dataset downloaded (excluded from git, re-download command documented); all three official reference baselines reproduced on this machine within their published seed noise — the environment is proven trustworthy; guardrails (edit-locks + bash guard + auto-state-injection hook) and this wiki installed, mirroring Track 3.
- **Deliberate deferral:** per-idea literature deep-dives happen at run time (fresh web search before each hypothesis), not up front — with a 50-iteration budget, researching idea #6 before ideas #1–2 have run would be waste.

## 28 Aug 2026 late morning — codex audit round 1 and the v0.2.0 rebuild

- Codex reviewed the committed setup (c66b060): **NO — 8 findings, 3 critical.** All fair. The harness was rebuilt (v0.2.0): test labels now mechanically stripped from what solutions receive; evaluator tamper-probed around candidate execution; every run seals its test predictions (scored artifact = measured artifact); `final` is once-only with a crash-evidence marker and the organizers' own CSV checker; convergence/cap/6h ceiling are refusals, not prose; journal entries carry harness sha, git state (incl. dirty flag), dataset hashes, and the solution's verbatim source. Malformed journal lines warn.
- Partial pushback recorded on finding 1's "test set already not pristine": the three reference test scores reproduced at setup are the organizers' OWN published numbers, obtained by running their unmodified script per the competition's explicit reproduce-the-baseline instruction — no information beyond their published table was gained. The real point stands and is now mechanical: no agent-designed solution's test metrics before the single final.
- The full final path was wired-tested on a SCRATCH ledger by designating the baseline iteration: test primary 0.5953, delta +0.0007 — i.e., exactly the already-published baseline number, so nothing new was revealed. The once-only guard proved itself during this test (a crashed first attempt left its pending marker; the retry was refused until explicitly --force'd with a journaled reason). Production journal holds no final.
- Known user-side item: the pasted README's "Limits" row still carries the organizers' contradictory metric text (README is user-owned; noted in TEMP log).

## 28 Aug 2026 midday — codex round 2 and the v0.3.0 hardening

- Round-2 verdict on the v2 commit: **NO — 8 claims examined, most NOT-VERIFIED** under an adversarial lens. Split triage:
- **ADOPTED (real defects regardless of trust model):** `final` now requires the designated entry to be error-free AND validation-best AND the run terminated (each override demands a non-empty journaled reason); the journaled final metric is computed from the checker-PARSED submission CSV (it proved a real ~5e-6 discrepancy vs the raw array); fail-closed ledger reads (any malformed line blocks final/run gates) + an exclusive final lockfile; per-iteration SIGALRM timeout; 6h clock anchored to a `start-run` marker so setup/review idle time stops consuming the run allowance; tamper-probe BEFORE sealing; source read+hashed before execution so crash paths keep provenance (verified with a deliberate import-crash); a suspicious-source scanner journals audit flags; manifest sha journaled per entry.
- **OVERRULED with precedent:** demands for out-of-process scoring and defenses against frame-walking / conditionally-mutated evaluators / solutions rereading raw CSVs — the same same-process residual codex itself accepted on Track 3 under the declared cooperative trust model ("mistakes, not malice", user ruling). Recorded here for the round-3 reviewer to judge the consistency argument.
- **ADOPTED its framing** of the setup-time reference scores: "not pristine — a bounded organizer-reference exception" (now in RUNBOOK verbatim).
- **User-approved memory upgrades** landed: tools/digest.py (read-only session-start journal view) and a mandatory reflection ritual in PLAN (distill a lesson + re-rank the queue after every iteration).

## 28 Aug 2026 early afternoon — codex round 3 and v0.4.0

- Round-3 verdict: NO, but the trust-model consistency argument was ACCEPTED for out-of-process isolation and frame-walking/mutated-evaluator attacks (Track 3 precedent honored). Its one principled exception was adopted in full: raw dataset rereads are a plausible COOPERATIVE mistake here (file-level feature engineering is encouraged by our own plan), so v0.4.0 ships `sanitize-data` — a deterministic dataset copy with all feedback signals zeroed on test-date rows, hash-pinned in the manifest, as the sanctioned file-level path; the raw dir is scanner-flagged.
- Also adopted: hidden-test labels now consulted EXACTLY once (sealed files load-verified without evaluation; the single evaluation runs on the checker-parsed CSV); the iteration timeout brackets everything from data load to sealing; the final lock is acquired before the ledger read (stale-snapshot race closed); run overrides require journaled reasons; the 6h clock anchors to the FIRST unresettable start-run marker and gates on unrounded seconds; scratch ledgers now carry their own sealed/ dir and CSV (true isolated universes); manifest sha on every entry type; entry ids collision-proofed; digest and harness agree on "finalizable-best" (error-free + sealed, last tied max); crash/hang red-team fixtures tracked in Project/harness/redteam/; RUNBOOK/PLAN/STATE wording corrected to match implementation exactly, including its verbatim "not pristine — a bounded organizer-reference exception" phrase.

## 28 Aug 2026 afternoon — codex rounds 4 through 8 (the convergence tail)

- **Round 4 (on 25cc683/v0.4.0): NO.** Verified most round-3 adoptions; new catches: the RANDOMIZED log's test-date feedback was copied intact by the sanitizer (real leak path — fixed in v0.5.0); sanitized hashes listed but not enforced (fixed); scanner missed path-join/read_csv access (fixed); the production ledger was already "converged" from setup runs — resolved by the principled official-run scoping: budget/convergence/best/6h-clock all begin at the first `start-run` marker, prior entries are phase-tagged setup consuming nothing; plus consistency items (base fields + random ids on all entry types, ledger-identity namespaces, raw-seconds gating) all adopted in v0.5.0.
- **Round 5 (on ea30c3b/v0.5.0): NO, one blocker** — my documentation contradicted the implementation (clock/budget wording, stale version string) while "the executable behavior appears otherwise freeze-ready." Fixed in v6 along with three of its four non-blocking hardening notes (fail-closed empty sanitized section; per-ledger final locks; digest fallback parity). The fourth (HYPOTHESIS recovery handles only simple quoted assignments) accepted as a documented limitation.
- **Round 6 (on b044531): NO, one blocker** — more stale doc strings I'd missed spot-fixing (header still v0.2.0, a comment contradicting the code beneath it, STATE round pointer). Fixed in v7 via a repo-wide grep sweep instead of spot fixes; the recovery limitation documented at the mechanism.
- **Round 7 (on dd0ab3e): NO, one blocker** — the overnight TEMP-PROGRESS-LOG still presented itself as live guidance with stale counts. Rewritten in v8 with a supersession notice and live counts delegated to the digest.
- **Round 8 (on 16768ef): NO, one blocker** — the review trail itself was internally inconsistent: this diary stopped at round 3 while TEMP claimed seven documented rounds; STATE pointed at a completed round; and the TEMP rewrite timestamp was future-dated against the commit clock. THIS entry, the STATE/TEMP corrections, and a real clock-sourced timestamp are the fix.
- Standing lesson adopted into practice: the diary gets its round entry IN THE SAME COMMIT as the round's fixes, and hand-written timestamps come from `date`, never from guesses.

- **Round 9 (on c9039bf): NO, one blocker** — two imprecise historical sentences in STATE ("executable unchanged since round 5" — false, round 6 hardened integrity enforcement and final locking; the same-commit diary rule framed as historical when rounds 4-7 were backfilled). Corrected in 7a71753.
- **Round 10 (on 7a71753): NO overall, YES for the executable** — the round-9 fix commit itself violated the prospective same-commit policy (no round-9 diary entry in it), TEMP still counted eight rounds, STATE's timestamp was stale. THIS commit is the compliance pattern: the round-10 entry, the round-9 entry it was missing, the count, the pointer, and a clock-sourced timestamp all land together. Pre-commit consistency ritual now standing: every review-response commit updates DECISIONS (round entry) + TEMP (count) + STATE (pointer + clock timestamp) together.

- **Round 11 (on 35de8ba): NO, one blocker** — STATE's header line still said "review trail completed through round 8", contradicting the round-11 pointer beneath it (the ritual updated the header's timestamp but missed its round marker). This commit fixes the header, records this entry per the same-commit ritual, and adds a grep check over all three ritual files for stale round numbers before committing.

- **Round 12 (on d4e4ea7): YES — Track 2 harness review loop CLOSED.** "Remaining blockers: none. Overall freeze verdict under the declared cooperative trust model: YES." Twelve rounds total: 8 findings → rebuild → 8-claim adversarial audit → sanitized-dataset round → then a documentation-consistency tail until the trail audited itself clean. Verdict preserved in Project/audits/track2_harness_verdict_round12.md. The executable bench had been freeze-ready since round 5; the tail rounds forged the audit-trail discipline now standing as ritual. Next: user freeze (TEMP-PROGRESS-LOG steps), then start-run.

- **Final handoff drills (both repos, zero-context read-only agents): PASSED.** Both independently verified hashes, closed loops, and pending user gates; correctly refused to act. Adopted from their findings: Track 3's sibling-status staleness fixed; the amendment-bundling suggestion (one re-freeze for shape-14 oracle + official subcommand); and the meter-honesty policy above (no solution authoring before start-run).

## 28 Aug 2026 afternoon — Track 2 webinar intel (user-provided transcript + slides 8-9; MEETING-NOTES.md)

Adopted into the plan (user-approved):
- **Video is officially optional for this track** (organizer will update the statement) — but the USER CHOOSES TO MAKE ONE anyway. Kept in deliverables; the report stays detailed regardless.
- **Slide 9 confirms our design verbatim:** the agent sees the training split and public validation "used freely, every iteration"; it never sees the hidden test; "final ranking is computed once … from the submission the agent marks as final." Our mechanical label-strip + sealed once-only final is the official diagram, implemented.
- **Final-model policy:** train on the TRAIN window only — never fold validation into final training (the engineer's own words plus his war story: touching test data once cost his production model ~10 AUC points; quotable in the report as motivation for our mechanical guards).
- **Intervention definition (official):** only changes to the agent's BEHAVIOR count as manual interventions; restarting a crashed process — manually or via a second babysitter session — does not. → Our journal treats restarts as recovery events (which are separately graded evidence of robustness); the intervention counter tracks behavior changes only.
- **Designated-run confirmation:** multiple development runs are fine; one official run is designated; earlier runs are disclosed. Exactly our setup-phase / start-run design.
- **log_random usage remains officially unresolved** (deferred to email by the organizers). Conservative standing policy until clarified: sanitized version only, validation-analysis only, never training data.
- Follow-up task: the engineer said "I have updated the starter toolkit" — verify our pinned kit matches the latest wiki download (read-only hash comparison) before the run starts.
- Deadline hard-confirmed: registration AND submission close 1 Sep 12:00 noon; People's Choice voting 1–7 Sep.

## 28 Aug 2026 16:48 — auto-audit per best experiment (user-directed, mechanically triggered)

Mirror of Track 3's mechanism: Project/tools/best_watch.py (hook-invoked; detects a change in the best FINALIZABLE experiment using the bench's own rule) + Project/tools/audit_best.py (detached blind codex audit of the journal entry incl. full solution source; verdict appended to Project/audits/verdicts.jsonl; digest displays the audit column). Non-blocking; hook added to .claude/settings.json this setup era (disclosed), arms at the user's restart. First real firing: the baseline best's audit launched at build time.

- **16:58 — first auto-audit completed: PASS** on the baseline best experiment (end-to-end with zero agent involvement). Wrapper fixes mirrored from Track 3 (flush-before-hash; wait-for-idle-bench before recording).

## 28 Aug 2026 17:22 — dual strategy review adoptions (user-approved; full texts in the Track 3 repo's audits)

Two independent zero-context reviewers (fresh Fable, fresh codex) assessed both tracks. For this repo:
- **Run priorities refined (codex):** keep a baseline-level fallback always; then pairwise BPR on the FM representation → user-grouped/listwise objective → blend/rank-average diverse successes. Sequences/multitask/watch-time only AFTER a clear above-noise gain — excellent report material, risky under the 6h clock. Randomized log stays out pending clarification (existing policy).
- **Pitch recorded:** "an autonomous recommender researcher operating inside a tamper-evident lab" — lead with the score trajectory and autonomous decisions; machinery explained briefly after.
- **Token metering:** attempt real provider-usage capture during the run; where impossible, self-reported figures are documented as such.
- **Packaging:** this track packages FIRST (CPU-bound, 6h-capped); judge-facing README (user applies) + merge initial-architecture into main BEFORE submitting (both repos' default branches are nearly empty — a judge following the link today would miss the work); TEMP/handoff files out of the judge reading path; triage rule if time collapses: Track 3 ships polished, this track ships as-is (the journal self-documents).

## 28 Aug 2026 17:25 — FREEZE APPROVED (Track 2) — the user's formal sign-off

Same act as Track 3's: the user pasted the harness deny lines into this repo's .claude/settings.json (verified: valid JSON, both lines present, hooks intact) and approved the freeze conditionally on that verification — condition met.
- Frozen artifact: Project/harness/iterate.py v0.5.0, sha256 starting 1babd3606bb23f5a; approval recorded at commit 5e3f97b; reviewer sign-off round 12 YES.
- Same honest caveat as Track 3: locks/hooks arm at the next session start (user on remote access); until then, hash verification + git history + behavioral compliance are the active layers.
- From this moment: no edits to Project/harness/** or any protected file by Claude's tools; changes only via a formal re-freeze.
