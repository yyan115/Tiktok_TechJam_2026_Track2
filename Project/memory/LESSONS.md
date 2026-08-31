# LESSONS — rules learned; check before working, add the moment one is learned

0. **A model warning is not a brake.** If the doer can ignore a reviewer or pass
   a "reasonable circumstances" flag, the reviewer has no authority. The real
   runner must consume exact approval and must make terminal transitions
   structurally unreachable.
0. **Convergence must remember the earliest triggering prefix.** Recomputing
   only the current last-three window allows a later gain to erase a prior
   convergence—the exact failure that occurred. Prefix-scan and freeze it.
0. **Fail closed on isolation.** A sandbox probe failure must cancel official
   execution, never silently fall back to host networking or a same-process
   candidate.
0. **Separate search quality from rule authority.** Diverse portfolios,
   falsifiers, and adversarial agents improve research. Only deterministic
   state owns competition stops, artifact promotion, and hidden-test access.
0. **Hashes establish identity, not truth.** A content-addressed research note
   proves which bytes were cited. It still needs primary-source provenance,
   bounded claims, caveats, and semantic challenge.
0. **Diagnostics are output channels.** Candidate-controlled error text, exit
   codes, byte counts, and even hashes can encode data. Public failure messages
   must be invariant; sensitive diagnostics stay owner-private.
0. **Feature provenance is part of split integrity.** An item table can leak
   future outcomes without containing a row-level label. KuaiRand's month-
   averaged engagement statistics are therefore withheld from the date-split
   candidate view.
0. **Component tests do not prove the installed system.** At least one clean,
   committed composition test must exercise the real service, socket client,
   and sandbox mount graph together.
0. **Persist idempotency keys before external effects.** A generated request ID
   that exists only in process memory cannot recover a completed request after
   a client crash.
0. **Do not claim a network boundary that does not exist.** A filesystem
   sandbox with shared host networking still needs provider-only egress or a
   dedicated VM to mechanically prevent public benchmark downloads.
0. **Protocol bounds must agree end to end.** Reserving a large escaped WAL
   response is useless if the client or MCP bridge cannot receive and persist
   the same bytes. Test the worst valid encoding across every hop.
0. **Type-check after a nonblocking open.** An attacker-writable filename can
   become a FIFO between enumeration and `fstat`; `O_NONBLOCK` prevents the
   authority from hanging before it can reject the special file.
0. **Broad runtime mounts are disclosure channels.** Reject overlap with every
   private repository/runtime/workspace path and disclose the remaining host
   runtime surface; a sealed VM image is the stronger boundary.
0. **Durability cannot depend on owner Git defaults.** Fixed Git operations
   must disable object substitution and lazy fetches, reject external object
   stores, and force both objects and refs through a durable fsync policy.
0. **Reserve recovery capacity before beginning a transition.** A WAL must
   prove that both its pending row and worst-case completion row fit before any
   side effect is admitted.

1. **Organizer-tested dead ends — never retry these** (published in their starter-kit README with numbers): adding more feature fields (CWM's 13 fields: 0.5940 vs 0.5950, noise); bigger embeddings (k=8/16/32: 0.5895/0.5902/0.5887, flat). Capacity and static features are NOT the bottleneck.
2. **Pure user-side features are mathematically worthless here.** Ranking happens within each user, so any per-user-constant term cancels (organizers verified: identical scores to the digit). User signals only help through crosses with item-side features or through sequences.
3. **The doc's "Limits" row is wrong** (says NDCG@10 / Recall@50 / click-positive). The shipped scoring code is authoritative: GAUC + nDCG@5, label = long_view, primary = their mean. Recall isn't scored (≈0.999 for any model — each user has ~5 impressions).
4. **Judge progress against 0.8645, not 1.0.** 27.1% of test users are all-negative (nDCG 0 forever), 9.2% all-positive. Baseline 0.5946 already holds ~31% of the attainable range; headroom ≈ 0.27 (0.247 on validation, ceiling 0.8484).
5. **FM seed noise is 0.0008** → convergence rule ε=0.002 over N=3 iterations. Don't celebrate sub-noise "wins".
6. **Submission format is strict:** row_id must be 0-based, gapless, aligned with data.load() order; (user_id, video_id) is NOT unique (3.06% duplicate pairs). Always run `submit.py --check` before designating anything final.
7. **Baselines reproduced on this machine 28 Aug** (random 0.4757 / pop 0.5715 exact / FM 0.5953) — environment trusted; if a future run's random self-check drifts from ~0.475, fix the harness before believing anything else.
8. **Test labels live on this disk but are off-limits during development.** Agent sees validation only; test is scored once at the end. Leak-guard: training rows must have date ≤ 20220421.
9. **Track 3's harness lessons carry over:** cheats to guard for (caching, timer games), noise floors before comparisons, journal written by the harness not by hand, champion = current harness version only.
