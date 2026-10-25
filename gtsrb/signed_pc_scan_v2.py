import sys, glob
import numpy as np
from scipy.stats import rankdata, kurtosis, skew

BUDGET, NPC = 0.05, 40

def auc(s, y):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def zs(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-9)

def best(P, y, npc=NPC, show=4):
    a = np.array([auc(P[:, j], y) for j in range(min(npc, P.shape[1]))])
    top = np.argsort(-np.abs(a - 0.5))[:show]
    return "  ".join("PC%d:%.2f" % (j + 1, a[j]) for j in top)

def scan(path):
    z = np.load(path, allow_pickle=True)
    L, C, y = z["loss"].astype(np.float32), z["conf"].astype(np.float32), z["is_poisoned"].astype(int)
    lab = z["y"]
    X = np.hstack([np.diff(L, axis=1), np.diff(C, axis=1)])
    U, S, _ = np.linalg.svd(zs(X), full_matrices=False)
    P = U * S
    kurt = np.array([kurtosis(P[:, j]) for j in range(NPC)])
    sk = np.array([skew(P[:, j]) for j in range(NPC)])
    k = int(BUDGET * len(y))
    print("\n=== %s  ckpts=%d  poison=%d  acc=%.3f ===" % (path, L.shape[1], y.sum(), float(z["clean_acc"])))
    jk = int(np.argmax(kurt))
    if y.sum() == 0:
        print("NULL run: max kurtosis %.2f (PC%d), max |skew| %.2f" % (kurt.max(), jk + 1, np.abs(sk).max()))
        return
    sp = np.linalg.norm(P[:, :2], axis=1)
    top = np.argsort(-sp)[:k]
    print("spectral_dyn (top-2 norm)  AUC=%.3f  precision@%d%%=%.3f  (chance %.3f)" % (auc(sp, y), BUDGET * 100, y[top].mean(), y.mean()))
    print("best single PCs, ALL samples (ORACLE):  " + best(P, y))
    s = np.sign(sk[jk]) * P[:, jk]
    top = np.argsort(-s)[:k]
    print("unsupervised max-kurtosis PC%d  AUC=%.3f  precision@%d%%=%.3f" % (jk + 1, auc(s, y), BUDGET * 100, y[top].mean()))
    tgt = np.bincount(lab[y == 1]).argmax()
    m = lab == tgt
    print("-- label control: target label %d, poison %d, clean with same label %d (chance %.3f)" % (tgt, y[m].sum(), (1 - y[m]).sum(), y[m].mean()))
    print("best single PCs, SAME-LABEL (ORACLE):   " + best(P[m], y[m]))
    Uc, Sc, _ = np.linalg.svd(zs(X[m]), full_matrices=False)
    a1 = auc(Uc[:, 0] * Sc[0], y[m])
    print("per-class SVD of target label, top PC:  AUC=%.3f (sign arbitrary, strength %.3f)" % (a1, max(a1, 1 - a1)))

files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
if not files:
    sys.exit("no files matched")
for f in files:
    scan(f)
