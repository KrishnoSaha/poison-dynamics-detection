import sys, glob, argparse, warnings
import numpy as np
from detect import spectral_dynamics, group_coherence, matched_benign_null
warnings.filterwarnings('ignore')

def one_draw(L, C, m, budget, rng, before):
    sub = rng.choice(len(L), min(m, len(L)), replace=False)
    Ls, Cs = L[sub], C[sub]
    s = spectral_dynamics(Ls, Cs)
    top = np.argsort(-s)[:max(2, int(budget * len(sub)))]
    obs = group_coherence(Ls, Cs, top)
    zb = np.nan
    if before:
        zb = matched_benign_null(Ls, Cs, top, rng=rng, n_draw=40)[3]
    return obs, zb

def load(path):
    z = np.load(path, allow_pickle=True)
    return z['loss'].astype(np.float32), z['conf'].astype(np.float32), z['is_poisoned'].astype(int)

def run_file(path, draws, m, budget, seed):
    L, C, y = load(path)
    rng = np.random.default_rng(seed)
    out = [one_draw(L, C, m, budget, rng, True) for _ in range(draws)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), int(y.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--null', nargs='+', required=True, help='no-poison run files')
    ap.add_argument('--test', nargs='*', default=[], help='poisoned run files')
    ap.add_argument('--draws', type=int, default=30)
    ap.add_argument('--m', type=int, default=10000, help='samples per random subsample')
    ap.add_argument('--budget', type=float, default=0.10)
    a = ap.parse_args()
    nulls = sorted(f for p in a.null for f in glob.glob(p))
    tests = sorted(f for p in a.test for f in glob.glob(p))
    if len(nulls) < 3:
        sys.exit('need at least 3 no-poison files (leave-one-out pool); matched: %d' % len(nulls))
    res = {}
    for f in nulls:
        obs, zb, npois = run_file(f, a.draws, a.m, a.budget, 1)
        if npois: sys.exit('%s has poison, not a null file' % f)
        res[f] = (obs, zb); print('null  %-40s done' % f, flush=True)
    tres = {}
    for f in tests:
        obs, zb, npois = run_file(f, a.draws, a.m, a.budget, 2)
        tres[f] = (obs, zb); print('test  %-40s done' % f, flush=True)

    print('\nBEFORE vs AFTER the fix  (draws=%d per file, subsample=%d, flagged budget=%.0f%%, confirmed means z>3)' % (a.draws, a.m, a.budget * 100))
    print('%-40s %14s %12s %14s %12s' % ('file', 'median z (kit)', 'confirmed', 'median z (fix)', 'confirmed'))
    tot_b = tot_a = tot_n = 0
    print('-- NO-POISON files: every confirmation here is false --')
    for f in nulls:
        obs, zb = res[f]
        pool = np.concatenate([res[g][0] for g in nulls if g != f])
        za = (obs - pool.mean()) / (pool.std() + 1e-9)
        cb, ca = (zb > 3).mean(), (za > 3).mean()
        tot_b += (zb > 3).sum(); tot_a += (za > 3).sum(); tot_n += len(zb)
        print('%-40s %14.1f %11.0f%% %14.1f %11.0f%%' % (f, np.median(zb), 100 * cb, np.median(za), 100 * ca))
    print('%-40s %14s %11.1f%% %14s %11.1f%%   <- false-confirmation rate over %d draws' % ('ALL NO-POISON DRAWS', '', 100 * tot_b / tot_n, '', 100 * tot_a / tot_n, tot_n))
    if tests:
        print('-- POISONED files (confirmation is legitimate only if it survives the fix) --')
        poolall = np.concatenate([res[g][0] for g in nulls])
        for f in tests:
            obs, zb = tres[f]
            za = (obs - poolall.mean()) / (poolall.std() + 1e-9)
            print('%-40s %14.1f %11.0f%% %14.1f %11.0f%%' % (f, np.median(zb), 100 * (zb > 3).mean(), np.median(za), 100 * (za > 3).mean()))

if __name__ == '__main__':
    main()
