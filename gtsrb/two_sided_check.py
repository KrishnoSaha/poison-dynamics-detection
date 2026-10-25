import sys, glob, warnings
import numpy as np
warnings.filterwarnings('ignore')
MIN_CLEAN, K, REPS = 30, 20, 3

def block(X, lab, y, n_ref, n_cal):
    res = {}
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        score = np.zeros(len(y)); held = np.zeros(len(y), bool); cal = {}
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]; clean = idx[y[idx] == 0]
            if len(clean) < MIN_CLEAN: continue
            nr = min(n_ref, int(.4 * len(clean))); nc = min(n_cal, int(.4 * len(clean)))
            pick = rng.permutation(clean)[:nr + nc]
            ref, cl = pick[:nr], pick[nr:]
            mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-9
            Zr = (X[ref] - mu) / sd; zm = Zr.mean(0)
            _, _, Vt = np.linalg.svd(Zr - zm, full_matrices=False); V = Vt[:min(K, nr - 1)]
            Za = (X[idx] - mu) / sd - zm
            score[idx] = np.linalg.norm(Za - (Za @ V.T) @ V, axis=1) / np.sqrt(Za.shape[1])
            cal[c] = score[cl].copy(); held[idx] = True; held[pick] = False
        for alpha in (0.01, 0.05):
            for mode in ('upper', 'lower', 'two-sided'):
                a = alpha / 2 if mode == 'two-sided' else alpha
                flag = np.zeros(len(y), bool)
                for c in cal:
                    idx = np.where(lab == c)[0]; n = len(cal[c])
                    qh = min(np.ceil((n + 1) * (1 - a)) / n, 1.0)
                    ql = max(1 - qh, 0.0)
                    hi, lo = np.quantile(cal[c], qh), np.quantile(cal[c], ql)
                    s = score[idx]
                    f = (s > hi) if mode == 'upper' else (s < lo) if mode == 'lower' else ((s > hi) | (s < lo))
                    flag[idx] = f
                fpr = flag[held & (y == 0)].mean()
                tpr = flag[held & (y == 1)].mean() if (held & (y == 1)).any() else float('nan')
                res.setdefault((alpha, mode), []).append((fpr, tpr))
    return res

def run(path):
    z = np.load(path, allow_pickle=True)
    L, C, y = z['loss'].astype(np.float32), z['conf'].astype(np.float32), z['is_poisoned'].astype(int)
    lab = z['y']; X = np.hstack([np.diff(L, axis=1), np.diff(C, axis=1)])
    med = np.median(np.bincount(lab)); n_ref, n_cal = (200, 500) if med >= 1250 else (60, 60)
    print('\n=== %s  %s  ref=%d cal=%d ===' % (path, 'NULL' if y.sum() == 0 else 'poison=%d' % y.sum(), n_ref, n_cal))
    res = block(X, lab, y, n_ref, n_cal)
    for alpha in (0.01, 0.05):
        for mode in ('upper', 'lower', 'two-sided'):
            r = np.array(res[(alpha, mode)]).mean(0)
            print('  target FPR=%d%%  %-9s achieved FPR=%.3f  TPR=%s' % (alpha * 100, mode, r[0], 'n/a' if np.isnan(r[1]) else '%.3f' % r[1]))

files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
if not files: sys.exit('no files matched')
for f in files: run(f)
