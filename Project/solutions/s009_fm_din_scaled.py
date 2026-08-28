"""Iteration: DIN-lite — target-aware attention over the user's engaged history.

Why (research-grounded): DIN (Zhou et al., KDD'18) showed that activating the
user's PAST BEHAVIOR by its similarity to the CANDIDATE item lifts ranking
AUC; the mechanism needs sequence data that our stock FM completely ignores.
This is capacity + information outside the (static fields x label) span —
the only open front after LESSONS 11-12a killed loss swaps and static
count-crosses. History order comes from time_ms in the SANITIZED train log
(the manifest's sanctioned file path).

Mechanism: z = FM(x) + w * sum_j softmax_j(e_cand . e_hist_j / sqrt(k)) *
(e_cand . e_hist_j), where hist = the last HIST_LEN train-window videos the
user engaged (long_view=1) STRICTLY BEFORE the row being scored (causal
within train, per-row); valid/test rows use the user's full train-window
history (their dates all lie after it). Video embeddings are shared with the
FM's video field — no new tables.

Risk control: the gate w starts at 0.0, so training starts from EXACTLY the
baseline model; the attention term only enters if the optimizer opens the
gate. Champion state saved/restored by deep copy (LESSON 10). Leak safety:
history events are train-window rows only; a row's own event never enters
its own history; only long_view (already the training label) is read from
the sanitized file — no test-date feedback exists there by construction.

Training recipe otherwise identical to the baseline (Adam lr 1e-3, bs 8192,
early stop patience 4 on valid primary, seed 0).
"""

import csv
import sys
from collections import deque
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"
sys.path.insert(0, str(KIT))

HYPOTHESIS = "Scale it9's working mechanism: HIST_LEN 10 to 30 (longer engaged history per user) plus a zero-initialized learned positional bias on attention slots (recency prior the data can shape). Same gate-from-zero safety; it9's +0.0007 says the sequence term is live, so give it more evidence to attend over."

HIST_LEN = 30
TRAIN_LO, TRAIN_HI = 20220408, 20220421


def run(splits):
    import numpy as np
    from data import encode
    from evaluate import evaluate
    from baseline import sigmoid

    train_rows = splits["train"]

    # ---- organizers' encoding, unchanged ----
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, _, ute = enc["test"]  # labels stripped by the harness; unused

    # ---- event times from the sanitized train-window log, aligned 1:1 ----
    san = KIT / "KuaiRand-Pure" / "data_sanitized" / "log_standard_4_08_to_4_21_pure.csv"
    times = []
    meta = []
    with open(san, newline="") as fh:
        for r in csv.DictReader(fh):
            d = int(r["date"])
            if TRAIN_LO <= d <= TRAIN_HI:
                times.append(int(r["time_ms"]))
                meta.append((r["user_id"], r["video_id"]))
    if len(times) != len(train_rows):
        raise SystemExit(f"seq alignment: {len(times)} sanitized train rows vs "
                         f"{len(train_rows)} split rows")
    for probe in (0, 1, len(times) // 2, len(times) - 1, 250000, 700000):
        if probe < len(times):
            x = train_rows[probe]
            if meta[probe] != (x[1], x[2]):
                raise SystemExit(f"seq alignment: row {probe} mismatch "
                                 f"{meta[probe]} vs {(x[1], x[2])}")

    # ---- causal per-row history (encoded video slot ids; -1 pad) ----
    by_user = {}
    for pos, x in enumerate(train_rows):
        by_user.setdefault(x[1], []).append(pos)
    Htr = np.full((len(train_rows), HIST_LEN), -1, dtype=np.int64)
    user_final = {}
    for u, poss in by_user.items():
        poss.sort(key=lambda p: (times[p], p))
        hist = deque(maxlen=HIST_LEN)
        for p in poss:
            for j, hv in enumerate(hist):
                Htr[p, j] = hv
            if ytr[p] > 0.5:
                hist.append(int(Xtr[p, 1]))
        user_final[u] = list(hist)

    def split_hist(users):
        H = np.full((len(users), HIST_LEN), -1, dtype=np.int64)
        for n, u in enumerate(users):
            h = user_final.get(u)
            if h:
                H[n, : len(h)] = h
        return H

    Hva = split_hist(uva)
    Hte = split_hist(ute)

    # ---- FM + gated target-aware attention ----
    k, lr, l2 = 16, 0.001, 1e-6
    tau = float(np.sqrt(k))
    rng = np.random.default_rng(0)
    V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
    W = np.zeros(dim, dtype=np.float32)
    b = np.float32(0.0)
    w_gate = 0.0
    pb = np.zeros(HIST_LEN, dtype=np.float32)   # learned positional bias
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mpb = np.zeros_like(pb); vpb = np.zeros_like(pb)
    mw = 0.0; vw = 0.0
    t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8

    def attn_forward(X, H):
        E = V[X]                                    # (B,F,k)
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        zfm = b + W[X].sum(1) + inter
        valid = H >= 0
        Hs = np.where(valid, H, 0)
        Eh = V[Hs]                                  # (B,L,k)
        ev = E[:, 1, :]                             # candidate video embedding
        s = np.einsum("blk,bk->bl", Eh, ev) / tau + pb[None, :]
        s_use = np.where(valid, s, 0.0)
        s_m = np.where(valid, s, -np.inf)
        m = s_m.max(1)
        m = np.where(np.isfinite(m), m, 0.0)
        e = np.exp(np.where(valid, s - m[:, None], -np.inf))
        asum = e.sum(1)
        a = e / np.maximum(asum, 1e-12)[:, None]
        f = (a * s_use).sum(1)
        out = tau * f                               # sum_j a_j (ev.eh_j)
        return zfm + w_gate * out, E, S, Eh, ev, a, s_use, f, out, valid, Hs

    def step(X, H, y):
        nonlocal t, b, w_gate, mw, vw
        B = len(y)
        z, E, S, Eh, ev, a, s_use, f, out, valid, Hs = attn_forward(X, H)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(V); gW = np.zeros_like(W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        # attention path
        c = a * (1.0 + s_use - f[:, None])          # (B,L); 0 at pads
        gw_coef = (g * w_gate).astype(np.float32)
        gv_cand = gw_coef[:, None] * (c[:, :, None] * Eh).sum(1)      # (B,k)
        np.add.at(gV, X[:, 1], gv_cand.astype(np.float32))
        gh = (gw_coef[:, None, None] * c[:, :, None] * ev[:, None, :])
        np.add.at(gV, Hs, gh.astype(np.float32))
        g_w = float((g * out).sum())
        g_pb = (tau * (gw_coef[:, None] * c).sum(0)).astype(np.float32)
        gV += l2 * V; gW += l2 * W
        t += 1
        for P, G, M, Vv in ((V, gV, mV, vV), (W, gW, mW, vW), (pb, g_pb, mpb, vpb)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= lr * (M / (1 - b1 ** t)) / (np.sqrt(Vv / (1 - b2 ** t)) + eps)
        mw = b1 * mw + (1 - b1) * g_w
        vw = b2 * vw + (1 - b2) * g_w * g_w
        w_gate -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
        b -= np.float32(lr * g.sum())

    def predict(X, H, bs=100_000):
        outs = []
        for i in range(0, len(X), bs):
            z = attn_forward(X[i:i + bs], H[i:i + bs])[0]
            outs.append(z)
        return np.concatenate(outs)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            bt = idx[i:i + 8192]
            step(Xtr[bt], Htr[bt], ytr[bt])
        primary = evaluate(uva, yva, predict(Xva, Hva))["primary"]
        print(f"[din] epoch {ep:2d} valid primary {primary:.6f} "
              f"(best {best:.6f}, gate {w_gate:+.4f})", flush=True)
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (V.copy(), W.copy(), np.float32(b), float(w_gate), pb.copy())
        else:
            bad += 1
            if bad >= 4:
                break

    V = best_state[0].copy()
    W = best_state[1].copy()
    b = np.float32(best_state[2])
    w_gate = float(best_state[3])
    pb = best_state[4].copy()
    return {"valid": predict(Xva, Hva), "test": predict(Xte, Hte)}
