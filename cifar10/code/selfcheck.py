"""
selfcheck.py -- validate the detectors on synthetic data with KNOWN ground truth.

Run this FIRST, before touching real data. It builds fake training dynamics where we know
exactly which samples are poisoned, in three difficulty settings, and reports how well each
detector does. If a detector cannot find planted poison here, it will not find real poison.

Three settings:
  obvious   poison learns fast AND coherently          -> every detector should work
  stealthy  poison's loss CURVE matched to benign,     -> curve-based detectors should FAIL,
            but still coherent                            coherence should still work
  decoy     a benign group that learns fast but is     -> nothing should flag it
            NOT coherent (e.g. genuinely easy images)     (this is the false-positive test)
"""
import numpy as np, sys
from detect import DETECTORS, group_coherence, matched_benign_null

T = 40


def make(n_benign=900, n_special=100, mode='obvious', seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)

    def benign(n):
        rate = rng.uniform(1.0, 4.0, (n, 1))
        base = np.exp(-rate * t) * rng.uniform(0.8, 1.2, (n, 1))
        loss = base + 0.05 * rng.standard_normal((n, T))
        # confidence driven by its OWN independent noise -> no shared direction
        conf = (1 - base) + 0.05 * rng.standard_normal((n, T))
        return loss, conf

    Lb, Cb = benign(n_benign)

    if mode == 'decoy':
        # fast but incoherent: each has its own steep rate, independent noise
        rate = rng.uniform(6.0, 9.0, (n_special, 1))
        base = np.exp(-rate * t)
        Ls = base + 0.05 * rng.standard_normal((n_special, T))
        Cs = (1 - base) + 0.05 * rng.standard_normal((n_special, T))
    else:
        # SHARED latent trigger signal: one time series driving all poisoned samples
        shared = np.cumsum(rng.standard_normal(T)) * 0.05
        rate = rng.uniform(6.0, 9.0, (n_special, 1)) if mode == 'obvious' \
            else rng.uniform(1.0, 4.0, (n_special, 1))          # stealthy: benign-like rate
        base = np.exp(-rate * t)
        Ls = base + 0.6 * shared[None, :] + 0.05 * rng.standard_normal((n_special, T))
        Cs = (1 - base) - 0.6 * shared[None, :] + 0.05 * rng.standard_normal((n_special, T))
        if mode == 'stealthy':
            # match the loss curve's level and spread to the benign population, so that any
            # detector reading only the loss curve is blinded
            Ls = (Ls - Ls.mean()) / (Ls.std() + 1e-9) * Lb.std() + Lb.mean()

    L = np.vstack([Lb, Ls]); C = np.vstack([Cb, Cs])
    y = np.concatenate([np.zeros(n_benign), np.ones(n_special)])
    return L, C, y.astype(int)


def auc(score, y):
    s = np.asarray(score, float)
    pos, neg = s[y == 1], s[y == 0]
    c = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return c / (len(pos) * len(neg))


def prec_at_budget(score, y, budget=0.10):
    k = int(budget * len(y))
    top = np.argsort(-score)[:k]
    return y[top].mean()


def main():
    print("=" * 74)
    print("SELF-CHECK on synthetic dynamics with known ground truth")
    print("=" * 74)
    rows = {}
    for mode in ('obvious', 'stealthy', 'decoy'):
        L, C, y = make(mode=mode, seed=1)
        print(f"\n--- {mode.upper()} "
              f"({int(y.sum())} planted of {len(y)}) ---")
        print(f"  {'detector':<14}{'AUC':>8}{'precision@10%':>15}")
        rows[mode] = {}
        for name, fn in DETECTORS.items():
            s = fn(L, C)
            a, p = auc(s, y), prec_at_budget(s, y)
            rows[mode][name] = a
            print(f"  {name:<14}{a:>8.3f}{p:>15.2f}")
        idx = np.where(y == 1)[0]
        obs, nm, n95, z = matched_benign_null(L, C, idx, rng=np.random.default_rng(2), n_draw=60)
        print(f"  group coherence of the planted set: {obs:+.3f}   "
              f"matched-benign null {nm:+.3f} (p95 {n95:+.3f})   z = {z:+.1f}")

    print("\n" + "=" * 74)
    print("WHAT TO CHECK")
    print("=" * 74)
    ok = True
    def req(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    req(rows['obvious']['loss_auc'] > 0.9, "obvious poison: loss-curve baseline works (AUC > 0.9)")
    req(rows['obvious']['spectral_dyn'] > 0.9, "obvious poison: shared-trigger detector works (AUC > 0.9)")
    req(rows['stealthy']['loss_auc'] < 0.75,
        "stealthy poison: loss-curve baseline degrades (AUC < 0.75) -- the attack works")
    best_base = max(rows['stealthy']['loss_auc'], rows['stealthy']['loss_shape'],
                    rows['stealthy']['early_loss'])
    req(rows['stealthy']['spectral_dyn'] > best_base + 0.10,
        f"stealthy poison: shared-trigger detector beats the BEST baseline "
        f"({best_base:.3f}) by > 0.10 AUC")
    req(rows['decoy']['spectral_dyn'] < 0.75,
        "decoy (benign but fast): shared-trigger detector does NOT flag it (AUC < 0.75)")
    print("\n" + ("All checks passed." if ok else "Some checks failed -- read the tables above."))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
