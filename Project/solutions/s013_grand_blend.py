"""Iteration: grand score-level blend of the run's diverse champions.

it13 proved the mechanisms interfere when JOINTLY trained (stack < either
component); the classic remedy — and the queue's designed ender — is to
combine at the SCORE level. Components, each trained independently exactly
as their winning iterations defined them: (D) the 3-seed dual-gate DIN-lite
ensemble (it12, current best 0.6024), (F) a 3-seed ensemble of the official
pointwise FM baseline (architecture diversity), (A) the alpha=0.3 watch-ratio
aux model (label-side diversity, it6). Per split, each model's scores become
normalized ranks (scale-free); seeds average within a component; a SMALL
fixed candidate set of component weightings {D, D+F, 2D+F, D+F+A, 2D+F+A} is
scored on validation and the winner is returned — rank-averaging diverse
models is the standing cheap win from the plan, now applied once, at the end.
"""

import csv
import sys
from collections import deque
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "kuairand-starter-kit"
sys.path.insert(0, str(KIT))

HYPOTHESIS = "Grand rank-average blend of separately-trained diverse champions (3-seed dual-DIN + 3-seed pointwise FM + watch-ratio-aux model): it13 showed joint training interferes, so combine at the score level; small weighting set picked on validation."

HIST_LEN = 10
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
    watch = []
    with open(san, newline="") as fh:
        for r in csv.DictReader(fh):
            d = int(r["date"])
            if TRAIN_LO <= d <= TRAIN_HI:
                times.append(int(r["time_ms"]))
                meta.append((r["user_id"], r["video_id"]))
                dur = float(r["duration_ms"])
                p = float(r["play_time_ms"]) / dur if dur > 0 else 0.0
                watch.append(min(max(p, 0.0), 1.0))
    if len(times) != len(train_rows):
        raise SystemExit(f"seq alignment: {len(times)} sanitized train rows vs "
                         f"{len(train_rows)} split rows")
    for probe in (0, 1, len(times) // 2, len(times) - 1, 250000, 700000):
        if probe < len(times):
            x = train_rows[probe]
            if meta[probe] != (x[1], x[2]):
                raise SystemExit(f"seq alignment: row {probe} mismatch "
                                 f"{meta[probe]} vs {(x[1], x[2])}")

    p_watch = np.asarray(watch, dtype=np.float32)

    # ---- causal per-row history: engaged (video_id, author_id) slot pairs ----
    by_user = {}
    for pos, x in enumerate(train_rows):
        by_user.setdefault(x[1], []).append(pos)
    Hv_tr = np.full((len(train_rows), HIST_LEN), -1, dtype=np.int64)
    Ha_tr = np.full((len(train_rows), HIST_LEN), -1, dtype=np.int64)
    user_final = {}
    for u, poss in by_user.items():
        poss.sort(key=lambda p: (times[p], p))
        hist = deque(maxlen=HIST_LEN)
        for p in poss:
            for j, (hv, ha) in enumerate(hist):
                Hv_tr[p, j] = hv
                Ha_tr[p, j] = ha
            if ytr[p] > 0.5:
                hist.append((int(Xtr[p, 1]), int(Xtr[p, 2])))
        user_final[u] = list(hist)

    def split_hist(users):
        Hv = np.full((len(users), HIST_LEN), -1, dtype=np.int64)
        Ha = np.full((len(users), HIST_LEN), -1, dtype=np.int64)
        for n, u in enumerate(users):
            h = user_final.get(u)
            if h:
                for j, (hv, ha) in enumerate(h):
                    Hv[n, j] = hv
                    Ha[n, j] = ha
        return Hv, Ha

    Hv_va, Ha_va = split_hist(uva)
    Hv_te, Ha_te = split_hist(ute)

    # ---- FM + two gated attention terms (video- and author-granularity) ----
    def train_one(seed):

        k, lr, l2 = 16, 0.001, 1e-6
        tau = float(np.sqrt(k))
        rng = np.random.default_rng(seed)
        V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        W = np.zeros(dim, dtype=np.float32)
        b = np.float32(0.0)
        w1 = 0.0  # video-attention gate
        w2 = 0.0  # author-attention gate
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mw1 = vw1 = mw2 = vw2 = 0.0
        t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8

        def attn_block(Ecand, H):
            """Target-aware softmax attention; returns (out, aux) for backprop."""
            valid = H >= 0
            Hs = np.where(valid, H, 0)
            Eh = V[Hs]                                  # (B,L,k)
            s = np.einsum("blk,bk->bl", Eh, Ecand) / tau
            s_use = np.where(valid, s, 0.0)
            s_m = np.where(valid, s, -np.inf)
            m = s_m.max(1)
            m = np.where(np.isfinite(m), m, 0.0)
            e = np.exp(np.where(valid, s - m[:, None], -np.inf))
            asum = e.sum(1)
            a = e / np.maximum(asum, 1e-12)[:, None]
            f = (a * s_use).sum(1)
            out = tau * f
            return out, (Eh, Hs, a, s_use, f)

        def forward(X, Hv, Ha):
            E = V[X]
            S = E.sum(1)
            inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
            zfm = b + W[X].sum(1) + inter
            out1, aux1 = attn_block(E[:, 1, :], Hv)
            out2, aux2 = attn_block(E[:, 2, :], Ha)
            return zfm + w1 * out1 + w2 * out2, E, S, out1, aux1, out2, aux2

        def attn_backprop(gV, X, field, gate, g, Ecand, aux):
            Eh, Hs, a, s_use, f = aux
            c = a * (1.0 + s_use - f[:, None])
            coef = (g * gate).astype(np.float32)
            gv_cand = coef[:, None] * (c[:, :, None] * Eh).sum(1)
            np.add.at(gV, X[:, field], gv_cand.astype(np.float32))
            gh = coef[:, None, None] * c[:, :, None] * Ecand[:, None, :]
            np.add.at(gV, Hs, gh.astype(np.float32))

        def step(X, Hv, Ha, y):
            nonlocal t, b, w1, w2, mw1, vw1, mw2, vw2
            B = len(y)
            z, E, S, out1, aux1, out2, aux2 = forward(X, Hv, Ha)
            g = ((sigmoid(z) - y) / B).astype(np.float32)
            gV = np.zeros_like(V); gW = np.zeros_like(W)
            np.add.at(gW, X, g[:, None])
            np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
            attn_backprop(gV, X, 1, w1, g, E[:, 1, :], aux1)
            attn_backprop(gV, X, 2, w2, g, E[:, 2, :], aux2)
            g_w1 = float((g * out1).sum())
            g_w2 = float((g * out2).sum())
            gV += l2 * V; gW += l2 * W
            t += 1
            for P, G, M, Vv in ((V, gV, mV, vV), (W, gW, mW, vW)):
                M *= b1; M += (1 - b1) * G
                Vv *= b2; Vv += (1 - b2) * (G * G)
                P -= lr * (M / (1 - b1 ** t)) / (np.sqrt(Vv / (1 - b2 ** t)) + eps)
            mw1 = b1 * mw1 + (1 - b1) * g_w1
            vw1 = b2 * vw1 + (1 - b2) * g_w1 * g_w1
            w1 -= lr * (mw1 / (1 - b1 ** t)) / (np.sqrt(vw1 / (1 - b2 ** t)) + eps)
            mw2 = b1 * mw2 + (1 - b1) * g_w2
            vw2 = b2 * vw2 + (1 - b2) * g_w2 * g_w2
            w2 -= lr * (mw2 / (1 - b1 ** t)) / (np.sqrt(vw2 / (1 - b2 ** t)) + eps)
            b -= np.float32(lr * g.sum())

        def predict(X, Hv, Ha, bs=100_000):
            outs = []
            for i in range(0, len(X), bs):
                outs.append(forward(X[i:i + bs], Hv[i:i + bs], Ha[i:i + bs])[0])
            return np.concatenate(outs)

        best, best_state, bad = -1.0, None, 0
        for ep in range(1, 41):
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), 8192):
                bt = idx[i:i + 8192]
                step(Xtr[bt], Hv_tr[bt], Ha_tr[bt], ytr[bt])
            primary = evaluate(uva, yva, predict(Xva, Hv_va, Ha_va))["primary"]
            print(f"[din2 s{seed}] epoch {ep:2d} valid primary {primary:.6f} "
                  f"(best {best:.6f}, gates v{w1:+.3f} a{w2:+.3f})", flush=True)
            if primary > best + 1e-5:
                best, bad = primary, 0
                best_state = (V.copy(), W.copy(), np.float32(b), float(w1), float(w2))
            else:
                bad += 1
                if bad >= 4:
                    break

        V = best_state[0].copy()
        W = best_state[1].copy()
        b = np.float32(best_state[2])
        w1 = float(best_state[3])
        w2 = float(best_state[4])
        return (predict(Xva, Hv_va, Ha_va), predict(Xte, Hv_te, Ha_te), best)

    def fm_one(seed, aux_alpha=0.0):
        """The official pointwise FM recipe; optional watch-ratio aux head."""
        from baseline import FM
        model = FM(dim, k=16, lr=0.001, seed=seed)
        rng2 = np.random.default_rng(seed)
        Wa = np.zeros(dim, dtype=np.float32); ba = np.float32(0.0)
        mWa = np.zeros_like(Wa); vWa = np.zeros_like(Wa); ta = 0
        best, best_state, bad = -1.0, None, 0
        for ep in range(1, 41):
            idx = rng2.permutation(len(ytr))
            for i in range(0, len(idx), 8192):
                bt = idx[i:i + 8192]
                X, y = Xtr[bt], ytr[bt]
                if aux_alpha > 0.0:
                    E = model.V[X]; S = E.sum(1)
                    inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
                    za = ba + Wa[X].sum(1) + inter
                    ga = (aux_alpha * (sigmoid(za) - p_watch[bt]) / len(y)).astype(np.float32)
                    gV = np.zeros_like(model.V); gWa = np.zeros_like(Wa)
                    np.add.at(gWa, X, ga[:, None])
                    np.add.at(gV, X, ga[:, None, None] * (S[:, None, :] - E))
                    ta += 1
                    for P, G, M, Vv in ((model.V, gV, model.mV, model.vV),):
                        M *= 0.9; M += 0.1 * G
                        Vv *= 0.999; Vv += 0.001 * (G * G)
                        P -= 0.001 * (M / (1 - 0.9 ** max(model.t, 1))) / (np.sqrt(Vv / (1 - 0.999 ** max(model.t, 1))) + 1e-8)
                    for P, G, M, Vv in ((Wa, gWa, mWa, vWa),):
                        M *= 0.9; M += 0.1 * G
                        Vv *= 0.999; Vv += 0.001 * (G * G)
                        P -= 0.001 * (M / (1 - 0.9 ** ta)) / (np.sqrt(Vv / (1 - 0.999 ** ta)) + 1e-8)
                    ba -= np.float32(0.001 * ga.sum())
                model.step(X, y)
            primary = evaluate(uva, yva, model.predict(Xva))["primary"]
            if primary > best + 1e-5:
                best, bad = primary, 0
                best_state = (model.V.copy(), model.W.copy(), model.b)
            else:
                bad += 1
                if bad >= 4:
                    break
        model.V, model.W, model.b = best_state[0].copy(), best_state[1].copy(), best_state[2]
        print(f"[blend] fm seed {seed} aux {aux_alpha}: champion {best:.6f}", flush=True)
        return model.predict(Xva), model.predict(Xte), best

    def to_rank(z):
        z = np.asarray(z)
        r = np.empty(len(z), dtype=np.float64)
        r[np.argsort(z, kind="stable")] = np.arange(len(z), dtype=np.float64)
        return r / max(len(z) - 1, 1)

    # component D: 3-seed dual-gate DIN
    Dv, Dt = None, None
    for seed in (0, 1, 2):
        zv, zt, best_seed = train_one(seed)
        print(f"[blend] din seed {seed}: champion {best_seed:.6f}", flush=True)
        rv, rt = to_rank(zv), to_rank(zt)
        Dv = rv if Dv is None else Dv + rv
        Dt = rt if Dt is None else Dt + rt
    Dv /= 3.0; Dt /= 3.0
    # component F: 3-seed pointwise FM
    Fv, Ft = None, None
    for seed in (0, 1, 2):
        zv, zt, _ = fm_one(seed)
        rv, rt = to_rank(zv), to_rank(zt)
        Fv = rv if Fv is None else Fv + rv
        Ft = rt if Ft is None else Ft + rt
    Fv /= 3.0; Ft /= 3.0
    # component A: watch-ratio aux FM (seed 0)
    zv, zt, _ = fm_one(0, aux_alpha=0.3)
    Av, At = to_rank(zv), to_rank(zt)

    candidates = {
        "D": (1.0, 0.0, 0.0),
        "D+F": (1.0, 1.0, 0.0),
        "2D+F": (2.0, 1.0, 0.0),
        "D+F+A": (1.0, 1.0, 1.0),
        "2D+F+A": (2.0, 1.0, 1.0),
    }
    best_name, best_p = None, -1.0
    for name, (wd, wf, wa) in candidates.items():
        blend_v = wd * Dv + wf * Fv + wa * Av
        p = evaluate(uva, yva, blend_v)["primary"]
        print(f"[blend] {name}: valid primary {p:.6f}", flush=True)
        if p > best_p:
            best_p, best_name = p, name
    wd, wf, wa = candidates[best_name]
    print(f"[blend] champion: {best_name} at {best_p:.6f}", flush=True)
    return {"valid": wd * Dv + wf * Fv + wa * Av,
            "test": wd * Dt + wf * Ft + wa * At}
