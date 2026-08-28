# TEMP — read this when you're back (Track 2)

*Written 28 Aug ~09:30 by Claude before you closed the session. Delete this file once done.*

## Where things stand (10-second version)

Fully set up and committed on branch `initial-architecture` (pushed to GitHub).
The dataset is downloaded, the organizers' three reference scores all reproduce on your
machine (their baseline: 0.5953 vs their published 0.5946 — within their own noise),
and our experiment bench ran its first logged experiment successfully. The 50-experiment
competition budget has 49 left.

## YOUR TO-DO, in order

**1. Nothing technical.** Setup is done.

**2. After you've done Track 3's freeze steps** (see that repo's TEMP file), just say
**"go track 2"** in a Claude session opened in THIS folder (or tell the Track 3 session —
it knows the way). What happens then, automatically:
   - the second AI (codex) reviews this repo's experiment bench, same ceremony as Track 3's referee
   - you say "freeze approved" for it (one line in this repo's `.claude/settings.json`, Claude will show you)
   - then the autonomous run starts: the agent works through its experiment queue
     (best ideas first, taken from the organizers' own published hints), logging every
     attempt, until scores stop improving. No babysitting needed — that's literally
     what gets graded.

**3. That's it until packaging day** (report + 3-min video + Devpost form, same as Track 3).

## What the plan is

Beat the official baseline score (0.5946) on their hidden test set, with a fully
autonomous, fully logged run. The experiment queue and all rules: `Project/PLAN.md`.
Things we must never waste tries on (the organizers already tested them): `Project/memory/LESSONS.md`.
Current status always in: `Project/memory/STATE.md`.
