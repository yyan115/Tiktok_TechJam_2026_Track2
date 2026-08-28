"""Iteration: behavioral cross feature — user→author long_view affinity.

Why (research-grounded): counting/historical-engagement features between the
user and the candidate's side are a standard, repeatedly-validated gain path
in industrial CTR ranking (e.g. Google's counting features; arXiv:2012.15522),
and they are exactly the mechanism the organizers' own ablations bless: user
signals help only through CROSSES with item-side features (their static-field
and pure-user-side ablations were flat — LESSONS 1-2). The stock 5 fields
carry no engagement history at all; ranking-loss swaps on that representation
are proven dead (LESSONS 11-12). This adds ONE new FM field: the user's
train-window long_view rate on the candidate's AUTHOR, smoothed toward the
global rate (prior=20, mirroring the pop baseline), bucketized on train
quantiles, with a distinct no-history value.

Leak safety: the affinity table is built from TRAIN-window rows only (dates
<= 20220421); valid/test rows only LOOK UP frozen counts. On train rows the
row's OWN label is subtracted first (leave-one-out) so no row can read its
target off its feature. No file access — everything comes from the splits
the harness hands in.

Training: the proven pointwise recipe, unchanged (FM k=16, lr 1e-3, bs 8192,
early stop on valid primary, patience 4, seed 0), from scratch.
"""

import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"
sys.path.insert(0, str(KIT))

HYPOTHESIS = "One behavioral cross field — LOO-smoothed user-to-author long_view rate from the train window — gives FM the engagement-history signal the stock 5 fields lack (counting features; organizers' own lesson that user signals must cross item-side); trained with the untouched pointwise recipe."

PRIOR = 20.0
N_BUCKETS = 10


def run(splits):
    import numpy as np
    from baseline import FM
    from evaluate import evaluate

    train_rows = splits["train"]

    # ---- affinity table from the train window only ----
    cnt = {}
    tot_pos = 0
    for x in train_rows:
        key = (x[1], x[3])  # (user_id, author_id)
        c = cnt.get(key)
        if c is None:
            cnt[key] = c = [0, 0]
        c[0] += x[6]
        c[1] += 1
        tot_pos += x[6]
    gmean = tot_pos / max(1, len(train_rows))

    def affinity_rate(u, a, y_own=None):
        c = cnt.get((u, a))
        if c is None:
            return None
        pos, imp = c
        if y_own is not None:  # leave-one-out on train rows
            pos -= y_own
            imp -= 1
        if imp <= 0:
            return None
        return (pos + PRIOR * gmean) / (imp + PRIOR)

    # Bucket edges from the train rows' own (LOO) rates — history rows only.
    rates = [r for r in (affinity_rate(x[1], x[3], x[6]) for x in train_rows)
             if r is not None]
    edges = np.quantile(np.asarray(rates),
                        np.linspace(0, 1, N_BUCKETS + 1)[1:-1])

    def aff_bucket(x, in_train):
        r = affinity_rate(x[1], x[3], x[6] if in_train else None)
        if r is None:
            return "NOHIST"
        return str(int(np.searchsorted(edges, r)))

    # ---- encoding: the organizers' 5 fields + the affinity field ----
    dur_edges = np.quantile(np.asarray([x[5] for x in train_rows]),
                            np.linspace(0, 1, 11)[1:-1])

    def raw(x, in_train):
        return [x[1], x[2], x[3], x[4],
                str(int(np.searchsorted(dur_edges, x[5]))),
                aff_bucket(x, in_train)]

    n_fields = 6
    vocabs = [dict() for _ in range(n_fields)]
    for x in train_rows:
        for i, v in enumerate(raw(x, True)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    dim = int(sum(field_dims))

    enc = {}
    for name, rws in splits.items():
        in_train = name == "train"
        X = np.empty((len(rws), n_fields), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x, in_train)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)

    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, _, _ = enc["test"]  # labels stripped by the harness; unused

    # ---- the untouched pointwise recipe ----
    model = FM(dim, k=16, lr=0.001, seed=0)
    rng = np.random.default_rng(0)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            model.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        print(f"[affinity] epoch {ep:2d} valid primary {primary:.6f} "
              f"(best {best:.6f})", flush=True)
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (model.V.copy(), model.W.copy(), model.b)
        else:
            bad += 1
            if bad >= 4:
                break

    model.V = best_state[0].copy()
    model.W = best_state[1].copy()
    model.b = best_state[2]
    return {"valid": model.predict(Xva), "test": model.predict(Xte)}
