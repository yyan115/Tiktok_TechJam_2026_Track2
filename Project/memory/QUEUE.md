# Live hypothesis queue (re-ranked after every iteration — reflection ritual)

Created at official-run start, 28 Aug 2026. Seed order per PLAN + dual strategy review.
Baselines: valid primary 0.6015 (GAUC 0.6671 / nDCG@5 0.5358). Ceiling valid 0.8484.
Noise floor: 0.0008 seed std; only >0.002 counts as a real win (convergence ε).

## Queue (current order — re-ranked after it4; convergence fired on the ranking-loss plateau, continuing under journaled override: organizer-ranked information-adding families untried at 0.13h/6h)
1. Behavioral cross feature: user→author long_view affinity from the TRAIN window (leave-one-out on train rows), one new FM field. The cheapest "sequence-lite" — new information, proven pointwise recipe. [idea #2-lite] ← NEXT (it5)
2. DIN-lite attention over the user's train-window long_view'd item embeddings (full sequence modeling). [idea #2]
3. Multi-task heads over other feedback signals (click, …) via data_sanitized. [idea #3]
4. Rank-average blend of diverse successes (standing cheap win; also the planned ender). [#1c]
5. Watch-time modeling (censored regression, CWM-style). [idea #4]
6. Time features / drift handling. [idea #6]
DEAD (LESSONS 1-2, 11-12): more static fields, bigger k, pure user-side features, ALL ranking-loss swaps/fine-tunes on the stock representation.

## Ritual log (one line per iteration: result → re-rank decision)
- it1 s000 0.601469: official floor anchored → proceed to queue head.
- it2 s001 0.600805 UNDER FLOOR: aliasing bug (LESSON 10) → fix & rerun, don't touch ordering.
- it3 s002 0.601469 (floor held; BPR epochs 0.6006-0.6009 all below warm start): BPR-on-stock-FM is a null result (LESSON 11) → listwise stays #1 (different mechanism: top-heavy joint normalization), sequences promoted to #2, BPR retune demoted.
- it4 s003 0.601469 (floor held; listwise epochs sink 0.6000→0.5990): whole ranking-loss family dead (LESSON 12); CONVERGENCE FIRED on this single-family plateau → continue under journaled override; behavioral cross (author affinity, LOO) promoted to #1 as the cheapest new-information play.
- it5 s004 0.601309 ≈ baseline: count-crosses FM already embeds are redundant (LESSON 12a) → pivot to LABEL-side information (feedback columns via sanitized files).
- it6 s005 0.601759 NEW BEST (+0.0003): soft watch-ratio aux head at α=0.3 moved it — first positive signal of the run → amplify: dose-response (α=1.0) at it7, then single-head soft-target-only, then head-blend. Multi-task/watch-time family promoted to the whole top of the queue.
- it7 s006 0.600038 at α=1.0 (worse than baseline): dose-response NON-monotonic — heavy aux weight distorts shared V against the scored head. Optimum near small α. Untested: the aux head's OWN ranking quality → it8 evaluates the full head space {main, aux, blends} at α=0.3 and returns the valid-best combo.
- it8 s007 0.601759 (identical champion; main wins every epoch, aux/blends never): watch-ratio aux = pure small regularizer, family ceiling sub-noise → stop refining it; go to SEQUENCES (DIN-lite causal attention over engaged train history via time_ms) — strongest untried prior, w-gated to start exactly at the baseline.
- it9 s008 0.602185 NEW BEST (+0.0007; gate self-opened to +0.17 at champion epoch): SEQUENCES ARE LIVE (LESSON 13) → amplify: it10 = scale the mechanism (HIST_LEN 10→30 + zero-init learned positional bias); it11 = author-level second attention gate; it12 = stack the watch-ratio aux (α=0.3) on the best sequence model; then blend ender.
- it10 s009 0.602059 (≈it9−0.0001, 2× wall): longer history + positional bias DON'T compound — L=10 engaged already covers most users; it9 config stands. Plan: it11 author-attention (new granularity), it12 seed-average of best config (variance-kill, near-sure small win), it13 aux-stack, it14 grand rank-average, then honest convergence assessment.
- it11 s010 0.602248 marginal new best (+0.00006 vs it9; author gate opens +0.09 but adds ~nothing — u×a is already an FM cross). Total +0.0008 = exactly the seed-noise floor → it12 MUST test seed robustness; seed-ensemble (mean of per-seed standardized scores, seeds 0/1/2) both answers it and reduces variance.
- it12 s011 0.602445 NEW BEST (+0.0010 cumulative, first above-noise): 3-seed ensembling adds +0.0002 on top of the single-seed champion — sequence gain is seed-robust.
- it13 s012 0.601372 STACK FAILS (worse than either component; aux inflates attention gates early — interference through shared V): joint training of proven mechanisms is NEGATIVE → combine at the SCORE level only. it14 = grand rank-average blend of separately-trained diverse champions (3-seed dual-DIN + 3-seed pointwise FM + aux model), tiny candidate set of weightings picked on valid. Then honest convergence declaration.
- it14 s013 0.602636 NEW BEST (2D+F wins; rank-avg beats standardized-mean; improvised aux member was degraded 0.6006 and lost every race — candidates rightly dropped it).
- it15 s014 0.602992 NEW BEST & FINAL PLANNED (5-seed D+F, faithful aux member reproduced it6 exactly at 0.601759 and NOW adds at score level: 2D+F+A wins). Cumulative +0.00152 valid over baseline.

## RUN CLOSED (28 Aug ~20:53) — convergence declared
Every queue family tried and characterized: ranking losses DEAD; static crosses DEAD; aux label-side small-positive; sequences LIVE (the run's main discovery); joint stacking NEGATIVE; score-level ensembling positive. Marginal iteration value fell to +0.0001–0.0003 ≪ ε=0.002; organizers' convergence rule satisfied. Validation-best finalizable entry: it15 (20260828-204207-8fb9a6, 0.602992). Awaiting USER sign-off to run `final`.
