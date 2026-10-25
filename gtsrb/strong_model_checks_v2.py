import sys, glob
import numpy as np
from scipy.stats import rankdata

import warnings; warnings.filterwarnings('ignore')
MIN_CLEAN = 30

def auc(s, y):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def tpr_at_fpr(s, y, f=0.01):
    thr = np.quantile(s[y == 0], 1 - f)
    return float((s[y == 1] > thr).mean())

def prec_at(s, y, b):
    top = np.argsort(-s)[:int(b * len(y))]
    return float(y[top].mean())

def zs(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-9)

def strat_auc(s, y, diff, nb=20, minn=20):
    edges = np.quantile(diff, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, diff, side="right") - 1, 0, nb - 1)
    num = den = 0.0
    for b in range(nb):
        m = idx == b
        if y[m].sum() >= minn and (1 - y[m]).sum() >= minn:
            w = y[m].sum() * (1 - y[m]).sum(); num += w * auc(s[m], y[m]); den += w
    return num / den if den else float("nan")

def trusted_scores(X, lab, y, n_trust, k, rng):
    score = np.zeros(len(y)); held = np.zeros(len(y), bool)
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        clean = idx[y[idx] == 0]
        if len(clean) < MIN_CLEAN: continue
        nt = min(n_trust, int(0.4 * len(clean)))
        trust = rng.choice(clean, nt, replace=False)
        mu, sd = X[trust].mean(0), X[trust].std(0) + 1e-9
        Zt = (X[trust] - mu) / sd; zm = Zt.mean(0)
        _, _, Vt = np.linalg.svd(Zt - zm, full_matrices=False)
        V = Vt[:min(k, nt - 1)]
        Za = (X[idx] - mu) / sd - zm
        recon = Za - (Za @ V.T) @ V
        score[idx] = np.linalg.norm(recon, axis=1) / np.sqrt(Za.shape[1])
        held[idx] = True; held[trust] = False
    return score, held

def conformal_block(X, lab, y, n_ref, n_cal, k=20, reps=3):
    out = {}
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        score = np.zeros(len(y)); held = np.zeros(len(y), bool); cal = {}
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]; clean = idx[y[idx] == 0]
            if len(clean) < MIN_CLEAN: continue
            nr = min(n_ref, int(0.4 * len(clean))); nc = min(n_cal, int(0.4 * len(clean)))
            pick = rng.permutation(clean)[:nr + nc]
            ref, cl = pick[:nr], pick[nr:]
            held[idx] = True
            mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-9
            Zr = (X[ref] - mu) / sd; zm = Zr.mean(0)
            _, _, Vt = np.linalg.svd(Zr - zm, full_matrices=False); V = Vt[:min(k, nr - 1)]
            Za = (X[idx] - mu) / sd - zm
            score[idx] = np.linalg.norm(Za - (Za @ V.T) @ V, axis=1) / np.sqrt(Za.shape[1])
            cal[c] = score[cl].copy(); held[pick] = False
        for alpha in (0.01, 0.05):
            flag = np.zeros(len(y), bool)
            for c in cal:
                idx = np.where(lab == c)[0]
                q = min(np.ceil((len(cal[c]) + 1) * (1 - alpha)) / len(cal[c]), 1.0)
                flag[idx] = score[idx] > np.quantile(cal[c], q)
            h = held
            pr = y[h & flag].mean() if (h & flag).sum() else float("nan")
            out.setdefault(alpha, []).append((flag[h & (y == 0)].mean(), flag[h & (y == 1)].mean(), pr))
    return out

def run(path):
    z = np.load(path, allow_pickle=True)
    L, C, y = z["loss"].astype(np.float32), z["conf"].astype(np.float32), z["is_poisoned"].astype(int)
    lab = z["y"]
    X = np.hstack([np.diff(L, axis=1), np.diff(C, axis=1)])
    med = np.median(np.bincount(lab))
    TRUST_SIZES, CAL_SIZES = ((50, 200, 500), (200, 500)) if med >= 1250 else ((20, 50, 100), (30, 60))
    if y.sum() == 0:
        print("\n=== %s  NULL run (no poison)  n=%d  classes=%d ===" % (path, len(y), len(np.unique(lab))))
        print("C) CONFORMAL on a clean run: achieved FPR should match the target")
        for n_ref in CAL_SIZES:
            res = conformal_block(X, lab, y, n_ref, n_ref)
            for alpha in (0.01, 0.05):
                r = np.array(res[alpha])
                print("   ref=%d cal=%d  target FPR=%d%%: achieved FPR=%.3f" % (n_ref, n_ref, alpha * 100, r[:, 0].mean()))
        return
    diff = L.mean(1); tgt = np.bincount(lab[y == 1]).argmax(); mt = lab == tgt
    print("\n=== %s  poison=%d  acc=%.3f  (chance precision %.3f) ===" % (path, y.sum(), float(z["clean_acc"]), y.mean()))
    U, S, _ = np.linalg.svd(zs(X), full_matrices=False)
    sp = np.linalg.norm((U * S)[:, :2], axis=1)
    f = -sp
    print("A) FLIPPED spectral_dyn (flag the LOWEST top-2 norm)")
    print("   AUC=%.3f  TPR@1%%FPR=%.3f  precision@5%%=%.3f  precision@10%%=%.3f" % (auc(f, y), tpr_at_fpr(f, y), prec_at(f, y, .05), prec_at(f, y, .10)))
    print("   controls: within-difficulty-bin AUC=%.3f   same-label-only AUC=%.3f (chance within label %.3f)" % (strat_auc(f, y, diff), auc(f[mt], y[mt]), y[mt].mean()))
    print("B) TRUSTED-REFERENCE detector (small clean set per label)")
    for n_trust in TRUST_SIZES:
        s, held = trusted_scores(X, lab, y, n_trust, 20, np.random.default_rng(0))
        yy, ss = y[held], s[held]
        tt = held & mt
        order = np.argsort(-ss)[:int(.05 * len(yy))]
        print("   trusted=%3d/class: AUC=%.3f  TPR@1%%FPR=%.3f  precision@5%%=%.3f  in-target-class AUC=%.3f" % (n_trust, auc(ss, yy), tpr_at_fpr(ss, yy), yy[order].mean(), auc(s[tt], y[tt])))
    print("C) CONFORMAL threshold (false-alarm rate set from a clean calibration split, per label)")
    for n_ref in CAL_SIZES:
        res = conformal_block(X, lab, y, n_ref, n_ref)
        for alpha in (0.01, 0.05):
            r = np.array(res[alpha])
            print("   ref=%d cal=%d  target FPR=%d%%: achieved FPR=%.3f  TPR=%.3f  precision of flagged=%.3f" % (n_ref, n_ref, alpha * 100, r[:, 0].mean(), r[:, 1].mean(), np.nanmean(r[:, 2])))

files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
if not files:
    sys.exit("no files matched")
for f in files:
    run(f)
