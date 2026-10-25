"""
signed_pc_scan.py -- Phase 1 diagnostic for the training-dynamics detector.

For each run (.npz from train_log.py) it asks: where in the spectrum of the
per-sample dynamics matrix does the poison signal live, and does the current
spectral_dyn score (norm of the top-2 projection) throw it away?

Usage (from this folder):
    python3 signed_pc_scan.py "runs_preact_50k/dyn_*.npz" "runs_vgg_50k/dyn_*.npz" "runs_cnn_50k/dyn_*.npz"

Notes
  * "ORACLE" columns use the poison labels -> they are an UPPER BOUND on what an
    unsupervised rule could achieve, NOT a deployable result. The number that
    matters for Gate 1 is the unsupervised max-kurtosis line.
  * PC signs are arbitrary, so a raw PC AUC can sit below 0.5; |AUC-0.5| is what
    matters (an AUC of 0.09 means a STRONG signal that is just sign-flipped).
  * Runs with zero poison are reported as null statistics (needed for Phase 2).
  * Files whose sample count != EXPECT_N are skipped, to stop mixed-scale results.

Needs: numpy, scipy   (pip install scipy)   -- does NOT use detect.py
"""
import sys, glob
import numpy as np
from scipy.stats import rankdata, kurtosis, skew

EXPECT_N, BUDGET, NPC = 50000, 0.05, 40

def auc(s, y):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def zs(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-9)

def scan(path):
    z = np.load(path, allow_pickle=True)
    L, C, y = z["loss"].astype(np.float32), z["conf"].astype(np.float32), z["is_poisoned"].astype(int)
    if len(y) != EXPECT_N:
        print(f"SKIP {path}: n={len(y)} (expected {EXPECT_N})"); return
    F = zs(np.hstack([np.diff(L, axis=1), np.diff(C, axis=1)]))
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    P = U * S
    kurt = np.array([kurtosis(P[:, j]) for j in range(NPC)])
    sk = np.array([skew(P[:, j]) for j in range(NPC)])
    k = int(BUDGET * len(y))
    acc = float(z['clean_acc']) if 'clean_acc' in z else float('nan')
    print(f"\n=== {path}  ckpts={L.shape[1]}  poison={int(y.sum())}  acc={acc:.3f} ===")
    jk = int(np.argmax(kurt))
    if y.sum() == 0:
        print(f"NULL run: max excess kurtosis over {NPC} PCs = {kurt.max():.2f} (PC{jk+1}); "
              f"max |skew| = {np.abs(sk).max():.2f}")
        return
    sp = np.linalg.norm(P[:, :2], axis=1)
    top = np.argsort(-sp)[:k]
    print(f"spectral_dyn (top-2 norm)      AUC={auc(sp, y):.3f}  precision@{BUDGET:.0%}={y[top].mean():.3f}  (chance {y.mean():.3f})")
    aucs = np.array([auc(P[:, j], y) for j in range(NPC)])
    best = np.argsort(-np.abs(aucs - 0.5))[:4]
    print("best single PCs (ORACLE):      " + "  ".join(f"PC{j+1}:{aucs[j]:.2f}" for j in best))
    s = np.sign(sk[jk]) * P[:, jk]
    top = np.argsort(-s)[:k]
    print(f"unsupervised max-kurtosis PC{jk+1:<2} AUC={auc(s, y):.3f}  precision@{BUDGET:.0%}={y[top].mean():.3f}")

if __name__ == "__main__":
    files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
    if not files:
        sys.exit("no files matched")
    for f in files:
        scan(f)
