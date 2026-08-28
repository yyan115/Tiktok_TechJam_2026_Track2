# LESSONS — rules learned; check before working, add the moment one is learned

1. **Organizer-tested dead ends — never retry these** (published in their starter-kit README with numbers): adding more feature fields (CWM's 13 fields: 0.5940 vs 0.5950, noise); bigger embeddings (k=8/16/32: 0.5895/0.5902/0.5887, flat). Capacity and static features are NOT the bottleneck.
2. **Pure user-side features are mathematically worthless here.** Ranking happens within each user, so any per-user-constant term cancels (organizers verified: identical scores to the digit). User signals only help through crosses with item-side features or through sequences.
3. **The doc's "Limits" row is wrong** (says NDCG@10 / Recall@50 / click-positive). The shipped scoring code is authoritative: GAUC + nDCG@5, label = long_view, primary = their mean. Recall isn't scored (≈0.999 for any model — each user has ~5 impressions).
4. **Judge progress against 0.8645, not 1.0.** 27.1% of test users are all-negative (nDCG 0 forever), 9.2% all-positive. Baseline 0.5946 already holds ~31% of the attainable range; headroom ≈ 0.27 (0.247 on validation, ceiling 0.8484).
5. **FM seed noise is 0.0008** → convergence rule ε=0.002 over N=3 iterations. Don't celebrate sub-noise "wins".
6. **Submission format is strict:** row_id must be 0-based, gapless, aligned with data.load() order; (user_id, video_id) is NOT unique (3.06% duplicate pairs). Always run `submit.py --check` before designating anything final.
7. **Baselines reproduced on this machine 28 Aug** (random 0.4757 / pop 0.5715 exact / FM 0.5953) — environment trusted; if a future run's random self-check drifts from ~0.475, fix the harness before believing anything else.
8. **Test labels live on this disk but are off-limits during development.** Agent sees validation only; test is scored once at the end. Leak-guard: training rows must have date ≤ 20220421.
9. **Track 3's harness lessons carry over:** cheats to guard for (caching, timer games), noise floors before comparisons, journal written by the harness not by hand, champion = current harness version only.
