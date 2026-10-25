import numpy as np, glob
from detect import matched_benign_null

for f in sorted(glob.glob("runs/dyn_*.npz")):
    z = np.load(f, allow_pickle=True)
    L, C, y = z['loss'], z['conf'], z['is_poisoned'].astype(int)
    if y.sum() == 0:
        print(f"{f}: no poison, skipped"); continue
    true_idx = np.where(y == 1)[0]
    obs, nm, n95, zz = matched_benign_null(L, C, true_idx, rng=np.random.default_rng(0), n_draw=30)
    print(f"{f:35s}  n_poison={len(true_idx):5d}  ground-truth z={zz:8.1f}")
