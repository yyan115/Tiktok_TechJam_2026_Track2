# Research workflow and candidate contract

## Required research sequence

1. Read this file and `PROBLEM.md` completely.
2. Request bounded EDA that can distinguish useful modelling directions.
3. Interpret the returned observations rather than merely repeating them.
4. Research relevant primary sources online. Connect each source to one concrete
   design decision and record limitations or applicability conditions.
5. Build a diverse portfolio, then choose the highest-value first experiment.
6. For each result: compare prediction with observation, diagnose the mechanism,
   and explicitly choose keep, refine, debug, ensemble, or abandon.

Do not work on videos, slides, submission logistics, or the user's schedule. Do not
stop because time feels short. The supervisor and controller own lifecycle limits.

## Research files

- `/workspace/eda_request.json`: bounded questions chosen by you.
- `/evidence/eda_results/*.json`: trusted answers generated from training data.
- `/evidence/research_plan.json`: accepted source-grounded portfolio.
- `/evidence/STATE.json`: controller-generated attempt history and current best.
- `/evidence/feedback/`: critic and failure feedback.
- `/workspace/candidate/`: the next frozen proposal.

`/evidence` is read-only. You may update your proposal in `/workspace`, but you
cannot rewrite measured results, accepted research, or critic history.

## Proposal files

`candidate/proposal.json` must contain:

- `kind`: `baseline`, `new_family`, `refine`, `debug`, or `ensemble`;
- `hypothesis`: causal reason this should improve ranking;
- `evidence`: concrete EDA result IDs, prior attempt results, and/or sources;
- `change`: what differs from the relevant parent;
- `falsifier`: result that would reject the hypothesis.

`candidate/candidate.py` is invoked as:

```text
candidate.py --mode attempt|final --data-root /data --output-dir /output \
  --cache-dir /cache [--checkpoint-dir /checkpoint]
```

In attempt mode it must write:

- `/output/validation_predictions.npy`: one finite float32/float64 score per row;
- `/output/checkpoint/`: every artifact needed for exact final inference.

In final mode it must load `/checkpoint` and write
`/output/test_predictions.npy`. It must not change the learned model. Available data
files are `train.parquet`, `evaluation_features.parquet`, `user_features.parquet`,
and `video_features.parquet`.

Before an attempt becomes score-eligible, the controller reruns final mode against
the same label-free validation features with training data absent. The frozen
checkpoint must reproduce the attempt predictions within numerical tolerance. This
is a deterministic checkpoint-integrity check inside the same counted attempt, not
another experiment or score query.

Candidate execution has no network and cannot see the repository, controller,
validation targets, event ledger, or raw benchmark directory. CUDA, PyTorch,
LightGBM, Polars, scikit-learn, NumPy, pandas, SciPy, and PyArrow are available.
