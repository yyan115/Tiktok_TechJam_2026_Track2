"""Iteration: BPR pairwise fine-tune on the warm-started FM.

Why (research-grounded): the metric is within-user ranking (GAUC counts
pos>neg pairs inside each user; nDCG@5 rewards the same order), but the
baseline trains a POINTWISE classifier. Pairwise objectives (BPR; FM-Pair,
arXiv:1812.08254) are the literature's standard fix on implicit feedback.
Our twist: negatives are sampled from the SAME user's logged impressions —
each training pair is exactly a pair GAUC scores. Pure user-side terms
cancel in the pair difference, mirroring the organizers' proof that
user-constant features cannot move this metric.

Risk control: phase 1 reproduces the official FM warm start (identical to
s000, seed 0); phase 2 fine-tunes with BPR at a lower LR under a fresh Adam
state; the returned model is the GLOBAL best validation state across both
phases, so this iteration cannot score below the baseline it warms from.

Contract note: splits arrive with test labels stripped; validation labels
are used only for early stopping/model selection, as the competition allows.
"""

import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"
sys.path.insert(0, str(KIT))

HYPOTHESIS = "BPR pairwise fine-tune on warm-started FM: within-user (pos,neg) impression pairs align the training loss with GAUC's pair counting (BPR/FM-Pair literature); global-best state keeping makes the baseline the floor."

FT_LR = 5e-4
FT_EPOCHS = 15
FT_PATIENCE = 3
BATCH_PAIRS = 8192


def run(splits):
    import numpy as np
    from data import encode
    from baseline import FM, sigmoid
    from evaluate import evaluate

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
    model.V, model.W, model.b = best_state
    global_best, global_state = best, best_state

    # ---- phase 2: within-user BPR pairs on the warmed representation --------
    # Index train impressions per user; only users with >=1 pos AND >=1 neg
    # can form pairs — exactly the users whose order GAUC can score.
    by_user = {}
    for i, (u, label) in enumerate(zip(utr, ytr)):
        by_user.setdefault(u, ([], []))[0 if label > 0.5 else 1].append(i)
    pos_rows, offs, cnts, neg_flat = [], [], [], []
    for u, (pos, neg) in by_user.items():
        if not pos or not neg:
            continue
        off = len(neg_flat)
        neg_flat.extend(neg)
        for i in pos:
            pos_rows.append(i)
            offs.append(off)
            cnts.append(len(neg))
    pos_rows = np.asarray(pos_rows, dtype=np.int64)
    offs = np.asarray(offs, dtype=np.int64)
    cnts = np.asarray(cnts, dtype=np.int64)
    neg_flat = np.asarray(neg_flat, dtype=np.int64)

    # Fresh Adam state for the fine-tune (moments from the pointwise phase
    # describe a different loss surface).
    mV = np.zeros_like(model.V); vV = np.zeros_like(model.V)
    mW = np.zeros_like(model.W); vW = np.zeros_like(model.W)
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    def bpr_epoch():
        nonlocal t
        perm = rng.permutation(len(pos_rows))
        for s in range(0, len(perm), BATCH_PAIRS):
            b = perm[s:s + BATCH_PAIRS]
            p = pos_rows[b]
            draw = (rng.random(len(b)) * cnts[b]).astype(np.int64)
            n = neg_flat[offs[b] + draw]
            Xp, Xn = Xtr[p], Xtr[n]
            zp, Ep, Sp = model.logits(Xp)
            zn, En, Sn = model.logits(Xn)
            # L = -log sigmoid(zp - zn); dL/dzp = -sigmoid(zn - zp)
            g = (-sigmoid(zn - zp) / len(b)).astype(np.float32)
            gV = np.zeros_like(model.V); gW = np.zeros_like(model.W)
            np.add.at(gW, Xp, g[:, None])
            np.add.at(gW, Xn, -g[:, None])
            np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
            np.add.at(gV, Xn, -g[:, None, None] * (Sn[:, None, :] - En))
            gV += model.l2 * model.V
            gW += model.l2 * model.W
            t += 1
            for P, G, M, Vv in ((model.V, gV, mV, vV), (model.W, gW, mW, vW)):
                M *= b1; M += (1 - b1) * G
                Vv *= b2; Vv += (1 - b2) * (G * G)
                P -= FT_LR * (M / (1 - b1 ** t)) / (np.sqrt(Vv / (1 - b2 ** t)) + eps)
            # bias cancels in pair differences; left untouched.

    bad = 0
    for _ in range(FT_EPOCHS):
        bpr_epoch()
        primary = evaluate(uva, yva, model.predict(Xva))["primary"]
        if primary > global_best + 1e-5:
            global_best, bad = primary, 0
            global_state = (model.V.copy(), model.W.copy(), model.b)
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                break

    model.V, model.W, model.b = global_state
    return {"valid": model.predict(Xva), "test": model.predict(Xte)}
