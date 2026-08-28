"""Iteration: in-user LISTWISE softmax fine-tune on the warm-started FM.

Why (research-grounded): softmax cross-entropy over each user's impression
list is a proven bound on nDCG under binary relevance (Bruch et al., ICTIR'19,
Google Research), and nDCG@5 is our weak metric half (valid 0.536 vs ceiling
0.697, while GAUC is already 0.667). Unlike BPR's uniform pair pressure
(null result, LESSON 11), softmax jointly normalizes the WHOLE list, so
gradient concentrates on positives that are currently ranked low — top-heavy,
like the metric. Per-user equal weighting matches nDCG's per-user mean.

Risk control (LESSON 10 applied): warm start = the official pointwise FM
(identical to s000, seed 0); fine-tune under a fresh Adam at low LR; the
champion state is saved AND restored by deep copy; returned model is the
global validation best across both phases — the baseline is the floor.

Contract note: splits arrive with test labels stripped; validation labels
are used only for early stopping/model selection, as the competition allows.
"""

import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"
sys.path.insert(0, str(KIT))

HYPOTHESIS = "In-user listwise softmax fine-tune on warm FM: softmax CE over each user's impressions is a proven nDCG bound (Bruch et al. ICTIR'19) and targets our weak half (nDCG@5 0.536, ceiling 0.697); per-user equal weighting mirrors nDCG's mean; deep-copied champion keeps the baseline floor."

FT_LR = 2e-4
FT_EPOCHS = 25
FT_PATIENCE = 4
USERS_PER_BATCH = 1024


def run(splits):
    import numpy as np
    from data import encode
    from evaluate import evaluate
    from baseline import FM

    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, _, _ = enc["test"]  # labels stripped by the harness; unused

    # ---- phase 1: the official pointwise FM warm start (identical to s000) --
    model = FM(dim, k=16, lr=0.001, seed=0)
    rng = np.random.default_rng(0)
    best, best_state, bad = -1.0, None, 0
    for _ in range(40):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            model.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (model.V.copy(), model.W.copy(), model.b)
        else:
            bad += 1
            if bad >= 4:
                break
    model.V, model.W, model.b = best_state[0].copy(), best_state[1].copy(), best_state[2]
    global_best, global_state = best, best_state

    # ---- phase 2: listwise softmax CE over each user's impression list ------
    # Usable users = mixed (>=1 pos and >=1 neg) — the ones the metric can move.
    by_user = {}
    for i, label in enumerate(ytr):
        by_user.setdefault(utr[i], []).append(i)
    seg_rows, seg_off, seg_len, seg_npos = [], [], [], []
    for u, rows in by_user.items():
        npos = int(sum(1 for i in rows if ytr[i] > 0.5))
        if npos == 0 or npos == len(rows):
            continue
        seg_off.append(len(seg_rows))
        seg_rows.extend(rows)
        seg_len.append(len(rows))
        seg_npos.append(npos)
    seg_rows = np.asarray(seg_rows, dtype=np.int64)
    seg_off = np.asarray(seg_off, dtype=np.int64)
    seg_len = np.asarray(seg_len, dtype=np.int64)
    seg_npos = np.asarray(seg_npos, dtype=np.float32)
    n_users = len(seg_len)
    is_pos_flat = (ytr[seg_rows] > 0.5).astype(np.float32)

    mV = np.zeros_like(model.V); vV = np.zeros_like(model.V)
    mW = np.zeros_like(model.W); vW = np.zeros_like(model.W)
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    # Positions vs rows: is_pos_flat/seg_npos are indexed by POSITION in
    # seg_rows, while Xtr/model need the row indices seg_rows holds — the
    # batch gathers positions first, then maps them to rows.
    def listwise_epoch():
        nonlocal t
        order = rng.permutation(n_users)
        for s in range(0, n_users, USERS_PER_BATCH):
            ks = order[s:s + USERS_PER_BATCH]
            lens = seg_len[ks]
            nb = len(ks)
            pos_pieces = [np.arange(o, o + l) for o, l in zip(seg_off[ks], lens)]
            posn = np.concatenate(pos_pieces)      # positions in seg_rows
            idx = seg_rows[posn]                   # train row indices
            boff = np.zeros(nb, dtype=np.int64)
            np.cumsum(lens[:-1], out=boff[1:])
            rep = np.repeat(np.arange(nb), lens)
            X = Xtr[idx]
            z, E, S = model.logits(X)
            zmax = np.maximum.reduceat(z, boff)
            p = np.exp(z - zmax[rep])
            denom = np.add.reduceat(p, boff)
            p = (p / denom[rep]).astype(np.float32)
            gz = ((p - is_pos_flat[posn] / np.repeat(seg_npos[ks], lens)) / nb
                  ).astype(np.float32)
            gV = np.zeros_like(model.V); gW = np.zeros_like(model.W)
            np.add.at(gW, X, gz[:, None])
            np.add.at(gV, X, gz[:, None, None] * (S[:, None, :] - E))
            gV += model.l2 * model.V
            gW += model.l2 * model.W
            t += 1
            for P, G, M, Vv in ((model.V, gV, mV, vV), (model.W, gW, mW, vW)):
                M *= b1; M += (1 - b1) * G
                Vv *= b2; Vv += (1 - b2) * (G * G)
                P -= FT_LR * (M / (1 - b1 ** t)) / (np.sqrt(Vv / (1 - b2 ** t)) + eps)

    bad = 0
    for ep in range(1, FT_EPOCHS + 1):
        listwise_epoch()
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        print(f"[listwise] epoch {ep:2d} valid primary {primary:.6f} "
              f"(global best {global_best:.6f})", flush=True)
        if primary > global_best + 1e-5:
            global_best, bad = primary, 0
            global_state = (model.V.copy(), model.W.copy(), model.b)
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                break

    model.V = global_state[0].copy()
    model.W = global_state[1].copy()
    model.b = global_state[2]
    return {"valid": model.predict(Xva), "test": model.predict(Xte)}
