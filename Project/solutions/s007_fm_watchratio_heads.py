"""Iteration: auxiliary soft watch-ratio head on shared FM embeddings.

Why (research-grounded): long_view binarizes away the continuous engagement
evidence in play_time_ms/duration_ms. Multi-task ranking with watch-time
auxiliaries is a validated gain path (MT-FwFM +0.74% AUC, arXiv:2008.09872;
YouTube's watch-time-weighted LR; thresholded-watch auxiliaries in
arXiv:2306.17426). A soft target (watch ratio in [0,1]) gives every train row
a graded label — distillation-like extra bits per row — while the scored head
stays the untouched long_view objective. This is LABEL-side new information,
not a feature FM already embeds (LESSON 12a killed that path).

Data path honesty: play_time_ms comes from the SANITIZED dataset copy
(data_sanitized/ — the manifest's sanctioned file-level path; test-date rows
have all feedback zeroed there). Only TRAIN-window rows (date <= 20220421)
are read, aligned 1:1 against the harness-provided train split and asserted
on user/video ids. Valid/test rows never contribute auxiliary targets.

Model: one shared embedding table V (the interaction term is shared); each
head owns its linear weights and bias. Loss = logloss(long_view) +
ALPHA * logloss(sigmoid(z_aux), watch_ratio). Training recipe otherwise
identical to the baseline (Adam lr 1e-3, bs 8192, early stop patience 4 on
valid primary of the MAIN head, seed 0). Champion saved/restored by copy.
"""

import csv
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"
sys.path.insert(0, str(KIT))

HYPOTHESIS = "Head-space selection on it6's two-head FM (alpha=0.3, the known-good dose): each epoch scores main, aux, and three blends (m+0.5a, m+a, m+2a) on validation and the champion is the best (combo, epoch) pair — measures the aux head's own ranking power (never observed in it6/it7) and harvests any blend gain in one run."

ALPHA = 0.3
TRAIN_LO, TRAIN_HI = 20220408, 20220421


def run(splits):
    import numpy as np
    from data import encode
    from evaluate import evaluate
    from baseline import sigmoid

    train_rows = splits["train"]

    # ---- auxiliary soft targets from the SANITIZED train-window log ----
    san = KIT / "KuaiRand-Pure" / "data_sanitized" / "log_standard_4_08_to_4_21_pure.csv"
    watch = []
    meta = []
    with open(san, newline="") as fh:
        for r in csv.DictReader(fh):
            d = int(r["date"])
            if TRAIN_LO <= d <= TRAIN_HI:
                dur = float(r["duration_ms"])
                p = float(r["play_time_ms"]) / dur if dur > 0 else 0.0
                watch.append(min(max(p, 0.0), 1.0))
                meta.append((r["user_id"], r["video_id"]))
    if len(watch) != len(train_rows):
        raise SystemExit(f"aux alignment: {len(watch)} sanitized train rows vs "
                         f"{len(train_rows)} split rows")
    for probe in (0, 1, len(watch) // 2, len(watch) - 1, 200000, 400000, 800000):
        if probe < len(watch):
            x = train_rows[probe]
            if meta[probe] != (x[1], x[2]):
                raise SystemExit(f"aux alignment: row {probe} mismatch "
                                 f"{meta[probe]} vs {(x[1], x[2])}")
    p_watch = np.asarray(watch, dtype=np.float32)

    # ---- organizers' encoding, unchanged ----
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, _, _ = enc["test"]  # labels stripped by the harness; unused

    # ---- two-head FM with a shared embedding table ----
    k, lr, l2 = 16, 0.001, 1e-6
    rng = np.random.default_rng(0)
    V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
    Wm = np.zeros(dim, dtype=np.float32)
    Wa = np.zeros(dim, dtype=np.float32)
    bm = np.float32(0.0)
    ba = np.float32(0.0)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mWm = np.zeros_like(Wm); vWm = np.zeros_like(Wm)
    mWa = np.zeros_like(Wa); vWa = np.zeros_like(Wa)
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    def interaction(X):
        E = V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2))), E, S

    def step(X, y, p):
        nonlocal t, bm, ba
        B = len(y)
        inter, E, S = interaction(X)
        zm = bm + Wm[X].sum(1) + inter
        za = ba + Wa[X].sum(1) + inter
        gm = ((sigmoid(zm) - y) / B).astype(np.float32)
        ga = (ALPHA * (sigmoid(za) - p) / B).astype(np.float32)
        gV = np.zeros_like(V); gWm = np.zeros_like(Wm); gWa = np.zeros_like(Wa)
        np.add.at(gWm, X, gm[:, None])
        np.add.at(gWa, X, ga[:, None])
        np.add.at(gV, X, (gm + ga)[:, None, None] * (S[:, None, :] - E))
        gV += l2 * V; gWm += l2 * Wm; gWa += l2 * Wa
        t += 1
        for P, G, M, Vv in ((V, gV, mV, vV), (Wm, gWm, mWm, vWm), (Wa, gWa, mWa, vWa)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= lr * (M / (1 - b1 ** t)) / (np.sqrt(Vv / (1 - b2 ** t)) + eps)
        bm -= np.float32(lr * gm.sum())
        ba -= np.float32(lr * ga.sum())

    def predict_heads(X, bs=200_000):
        zm_out, za_out = [], []
        for i in range(0, len(X), bs):
            Xb = X[i:i + bs]
            inter, _, _ = interaction(Xb)
            zm_out.append(bm + Wm[Xb].sum(1) + inter)
            za_out.append(ba + Wa[Xb].sum(1) + inter)
        return np.concatenate(zm_out), np.concatenate(za_out)

    # Candidate score combos over the two heads; weights (wm, wa).
    COMBOS = [("main", 1.0, 0.0), ("aux", 0.0, 1.0),
              ("m+0.5a", 1.0, 0.5), ("m+a", 1.0, 1.0), ("m+2a", 1.0, 2.0)]

    best, best_state, best_combo, bad = -1.0, None, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            b = idx[i:i + 8192]
            step(Xtr[b], ytr[b], p_watch[b])
        zm, za = predict_heads(Xva)
        ep_best, ep_combo = -1.0, None
        for name, wm_, wa_ in COMBOS:
            primary = evaluate(uva, yva, wm_ * zm + wa_ * za)["primary"]
            if primary > ep_best:
                ep_best, ep_combo = primary, (name, wm_, wa_)
        print(f"[heads] epoch {ep:2d} best {ep_best:.6f} via {ep_combo[0]} "
              f"(global {best:.6f})", flush=True)
        if ep_best > best + 1e-5:
            best, bad = ep_best, 0
            best_combo = ep_combo
            best_state = (V.copy(), Wm.copy(), np.float32(bm),
                          Wa.copy(), np.float32(ba))
        else:
            bad += 1
            if bad >= 4:
                break

    V = best_state[0].copy()
    Wm = best_state[1].copy()
    bm = np.float32(best_state[2])
    Wa = best_state[3].copy()
    ba = np.float32(best_state[4])
    name, wm_, wa_ = best_combo
    print(f"[heads] champion combo: {name} at valid {best:.6f}", flush=True)
    zm_va, za_va = predict_heads(Xva)
    zm_te, za_te = predict_heads(Xte)
    return {"valid": wm_ * zm_va + wa_ * za_va,
            "test": wm_ * zm_te + wa_ * za_te}
