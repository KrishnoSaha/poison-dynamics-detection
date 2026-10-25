import argparse, json
import numpy as np
from detect import DETECTORS, group_coherence, matched_benign_null

def auc(score, y):
    s = np.asarray(score, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def at_budget(score, y, budget):
    k = max(1, int(budget * len(y)))
    top = np.argsort(-np.asarray(score))[:k]
    caught = y[top].sum()
    return dict(k=k, recall=caught / max(y.sum(), 1), precision=caught / k,
                clean_removed=(k - caught) / max((y == 0).sum(), 1), idx=top)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--budget', type=float, default=0.10)
    a = ap.parse_args()

    for f in a.files:
        z = np.load(f, allow_pickle=True)
        L, C, y = z['loss'], z['conf'], z['is_poisoned'].astype(int)
        cfg = json.loads(str(z['config']))
        print(f"\n{'='*74}")
        # Removed the file name {f} from this print statement
        print(f"attack={cfg['attack']} rate={cfg['rate']}  "
              f"poisoned {y.sum()}/{len(y)}  checkpoints {L.shape[1]}")
        print(f"clean test acc {float(z['clean_acc']):.3f}   attack success rate "
              f"{float(z['asr']):.3f}")
        if y.sum() == 0:
            print("  no poison in this run (attack=none): use it as a false-alarm test.")
        print(f"{'='*74}")
        print(f"  {'detector':<14}{'AUC':>7}{'recall':>9}{'precision':>11}"
              f"{'clean lost':>12}   at budget {a.budget:.0%}")

        best = None
        for name, fn in DETECTORS.items():
            s = fn(L, C)
            r = at_budget(s, y, a.budget)
            print(f"  {name:<14}{auc(s, y):>7.3f}{r['recall']:>9.2f}"
                  f"{r['precision']:>11.2f}{r['clean_removed']:>12.2%}")
            if name == 'spectral_dyn':
                best = r

        if best is not None and y.sum() > 0:
            obs, nm, n95, zz = matched_benign_null(L, C, best['idx'],
                                                   rng=np.random.default_rng(0), n_draw=60)
            print(f"\n  CONFIRMATION on the flagged set (shared-trigger detector):")
            print(f"    move-together score {obs:+.3f}   equally-easy benign groups "
                  f"{nm:+.3f} (95th pct {n95:+.3f})   z = {zz:+.1f}")
            verdict = ("CONFIRMED: moves together far more than equally-easy benign data."
                       if zz > 3 else
                       "NOT CONFIRMED: no more coherent than equally-easy benign data.")
            print(f"    {verdict}")
            if best['precision'] < 0.25:
                print(f"    [!] precision is only {best['precision']:.2f}, so this number "
                      f"describes whatever got flagged, NOT the poison. Read precision first.")

if __name__ == '__main__':
    main()
