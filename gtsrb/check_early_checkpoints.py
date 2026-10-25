import numpy as np
from detect import spectral_dynamics
from evaluate import at_budget, auc

for f in ["runs/dyn_badnets.npz", "runs/dyn_adaptive_shape.npz"]:
    z = np.load(f, allow_pickle=True)
    L, C, y = z['loss'], z['conf'], z['is_poisoned'].astype(int)
    n_ckpt = L.shape[1]
    print(f"\n=== {f} ({n_ckpt} checkpoints total) ===")
    for frac in [0.1, 0.25, 0.5, 1.0]:
        k = max(2, int(frac * n_ckpt))
        s = spectral_dynamics(L[:, :k], C[:, :k])
        r = at_budget(s, y, 0.10)
        print(f"  first {frac:.0%} of checkpoints ({k:>4} ckpts): "
              f"AUC={auc(s, y):.3f}  precision={r['precision']:.3f}  recall={r['recall']:.2f}")
