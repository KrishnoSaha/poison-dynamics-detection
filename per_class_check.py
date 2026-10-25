import sys, glob, csv, warnings
import numpy as np
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings('ignore')

REPS, N_REF, N_CAL, K, MIN_CLEAN = 5, 60, 60, 20, 30

def auc(s, y):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')

def per_class(X, lab, y, alpha):
    classes = np.unique(lab)
    acc = {c: dict(fpr=[], tpr=[], auc=[]) for c in classes}
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        for c in classes:
            idx = np.where(lab == c)[0]; clean = idx[y[idx] == 0]
            if len(clean) < MIN_CLEAN: continue
            nr = min(N_REF, int(.4 * len(clean))); nc = min(N_CAL, int(.4 * len(clean)))
            pick = rng.permutation(clean)[:nr + nc]
            ref, cl = pick[:nr], pick[nr:]
            mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-9
            Zr = (X[ref] - mu) / sd; zm = Zr.mean(0)
            _, _, Vt = np.linalg.svd(Zr - zm, full_matrices=False); V = Vt[:min(K, nr - 1)]
            Za = (X[idx] - mu) / sd - zm
            sc = np.linalg.norm(Za - (Za @ V.T) @ V, axis=1) / np.sqrt(Za.shape[1])
            cs = np.linalg.norm(((X[cl] - mu) / sd - zm) - (((X[cl] - mu) / sd - zm) @ V.T) @ V, axis=1) / np.sqrt(Za.shape[1])
            q = min(np.ceil((len(cs) + 1) * (1 - alpha)) / len(cs), 1.0)
            thr = np.quantile(cs, q)
            held = ~np.isin(idx, pick)
            yy, ss = y[idx][held], sc[held]
            if (yy == 0).any(): acc[c]['fpr'].append(float((ss[yy == 0] > thr).mean()))
            if (yy == 1).any():
                acc[c]['tpr'].append(float((ss[yy == 1] > thr).mean())); acc[c]['auc'].append(auc(ss, yy))
    return acc

def run(path):
    z = np.load(path, allow_pickle=True)
    L, C, y = z['loss'].astype(np.float32), z['conf'].astype(np.float32), z['is_poisoned'].astype(int)
    lab = z['y']
    X = np.hstack([np.diff(L, axis=1), np.diff(C, axis=1)])
    tag = 'NULL' if y.sum() == 0 else 'poison=%d' % y.sum()
    print('\n=== %s  %s  classes=%d ===' % (path, tag, len(np.unique(lab))))
    tgt = np.bincount(lab[y == 1]).argmax() if y.sum() else None
    for alpha in (0.05,):
        acc = per_class(X, lab, y, alpha)
        rows = []
        for c in sorted(acc):
            n = int((lab == c).sum()); nc_ = int(((lab == c) & (y == 0)).sum())
            f = np.mean(acc[c]['fpr']) if acc[c]['fpr'] else float('nan')
            t = np.mean(acc[c]['tpr']) if acc[c]['tpr'] else float('nan')
            a = np.nanmean(acc[c]['auc']) if acc[c]['auc'] else float('nan')
            rows.append((c, n, nc_, f, t, a))
        sizes = np.array([r[2] for r in rows]); fprs = np.array([r[3] for r in rows])
        ok = ~np.isnan(fprs)
        print('target FPR %d%%: mean per-class FPR=%.3f  (min %.3f, max %.3f)' % (alpha * 100, np.nanmean(fprs), np.nanmin(fprs), np.nanmax(fprs)))
        if ok.sum() > 3:
            rho, p = spearmanr(sizes[ok], fprs[ok])
            print('  FPR vs clean class size: Spearman rho=%.2f (p=%.2f)' % (rho, p))
        order = np.argsort(sizes)
        third = len(order) // 3
        for name, sel in (('smallest third', order[:third]), ('middle third', order[third:2 * third]), ('largest third', order[2 * third:])):
            sel = [i for i in sel if ok[i]]
            if sel: print('  %-15s clean n=%d-%d  mean FPR=%.3f' % (name, sizes[sel].min(), sizes[sel].max(), fprs[sel].mean()))
        worst = sorted([r for r in rows if not np.isnan(r[3])], key=lambda r: -r[3])[:5]
        print('  5 highest-FPR classes (class, clean n, FPR): ' + ', '.join('%d, %d, %.3f' % (r[0], r[2], r[3]) for r in worst))
        if tgt is not None:
            r = [r for r in rows if r[0] == tgt][0]
            print('  TARGET class %d: clean n=%d, poison n=%d, FPR=%.3f, TPR (poison caught)=%.3f, in-class AUC=%.3f' % (tgt, r[2], int(y[lab == tgt].sum()), r[3], r[4], r[5]))
        out = path.replace('.npz', '_perclass.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['class', 'n_total', 'n_clean', 'fpr', 'tpr', 'auc'])
            for r in rows: w.writerow(['%d' % r[0], r[1], r[2], '%.4f' % r[3], '%.4f' % r[4], '%.4f' % r[5]])
        print('  wrote', out)

files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
if not files: sys.exit('no files matched')
for f in files: run(f)
