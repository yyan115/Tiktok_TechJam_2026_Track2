# Supplied organizer reference: what transfers and what does not

The supplied starter kit is for **KuaiRand-Pure**, not KuaiRand-1K. Its evaluator
defines the shared task contract, but every published number below is a Pure
number. Never call one of these an official 1K baseline or a measured 1K result.

## Metric authority

- Rank only the impressions already logged for each user.
- Label: `long_view`.
- Primary: mean of GAUC and nDCG@5.
- GAUC excludes all-positive and all-negative users and weights eligible users by
  their positive count. Score ties use average ranks.
- nDCG@5 includes every user; an all-negative user contributes zero.
- Input row order is authoritative because `(user_id, video_id)` is not unique.

The controller's NumPy implementation is regression-tested against the supplied
organizer evaluator. Candidate code must not implement or alter scoring.

## Supplied Pure baseline

The official Pure model is a pointwise factorization machine with fields
`user_id`, `video_id`, `author_id`, `tab`, and duration bucket; embedding width 16,
learning rate 0.001, batch size 8192, at most 40 epochs, and patience 4. Its
published Pure scores are validation 0.6016 and hidden test 0.5946, with test
primary standard deviation about 0.0008 over five seeds.

Treat a port of this model to 1K as a **reference adaptation**, whose score and
noise are unknown until the controller measures it. No official 1K baseline was
present in the supplied files.

## Evidence from Pure, not facts about 1K

On Pure, adding all static fields or increasing FM width from 8 to 32 did not
help. The supplied notes identify ranking losses, history/sequence modelling,
multi-task feedback, watch-time modelling, temporal drift, and model-family
changes as open directions. These are hypotheses for 1K, not a mandated queue.

KuaiRand's own 1K README reports 1,000 users, 4,369,953 items, 11,713,045 standard
interactions, and thousands of historical interactions per user on average. That
is a very different sparsity and sequence regime from Pure. Use measured 1K EDA
and primary literature to decide what transfers.

## Current campaign boundary

The random-exposure log is excluded because it overlaps evaluation dates, and the
month-aggregated video-statistics file is excluded because its temporal provenance
is unresolved. Do not plan around either file. Basic video metadata, user
metadata, all training feedback, and label-free evaluation impressions are
available.
