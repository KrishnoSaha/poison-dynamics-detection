"""
positive_control.py -- the "is my test blind?" check (PROJECT_GUIDE / README rule).

Before writing "coherence does not detect poison on real architectures", we must
show the detector CAN find a coherent signal of the size we care about, planted
into THIS data shape. If it finds the planted signal but not the real poison, the
negative result is real. If it cannot even find the planted signal, the detector
pipeline is broken on this data -- an implementation bug, not a scientific finding.

What it does, for each run file:
  1. load the real per-sample dynamics (loss, conf) -- the clean, no-poison 'none'
     run is ideal, but any file works (we overwrite a random 5% with a signal).
  2. pick a random 5% of samples to be "fake poison".
  3. inject a SHARED coherent push into their loss/conf increments -- exactly the
     structure spectral_dynamics is designed to detect (one common direction added
     to all of them, at a controllable strength).
  4. run the real spectral_dynamics detector and report AUC + precision at 5%.

Read it like this:
  planted-signal AUC ~0.9+  -> detector WORKS on this data; your real-poison
                               negative result is genuine and defensible.
  planted-signal AUC ~0.5   -> detector is blind on this data shape; fix the
                               pipeline before claiming any negative.

Sweeps a few signal strengths so you can see how strong a coherent signal has to
be before the detector sees it -- which also quantifies how weak the real poison's
signal must be.

Usage (from the kit folder, next to detect.py):
    python3 positive_control.py "runs/dyn_none_preact.npz" "runs/dyn_none_vgg.npz"
    # or any run(s) you like:
    python3 positive_control.py "runs/dyn_*_preact.npz"

Needs: numpy, and detect.py on the path (uses the REAL detector, no reimplementation).
"""
import sys, glob
import numpy as np
from detect import spectral_dynamics

RATE = 0.05          # fraction made into fake poison
STRENGTHS = [0.5, 1.0, 2.0, 4.0]   # how strong the shared push is (in std units)
SEED = 0

def auc(score, y):
    # rank-based AUC, no sklearn dependency
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def inject(loss, conf, idx, strength, rng):
    """Add ONE shared coherent pattern to the increments of the idx samples.
    This is the cleanest possible 'these move together' signal."""
    L = loss.copy(); C = conf.copy()
    T = loss.shape[1]
    # one shared random time-pattern (same for every planted sample) over the steps
    patt_L = rng.standard_normal(T); patt_L /= patt_L.std() + 1e-9
    patt_C = rng.standard_normal(T); patt_C /= patt_C.std() + 1e-9
    # scale by each file's own increment spread so 'strength' means the same everywhere
    sL = np.diff(loss, axis=1).std() + 1e-9
    sC = np.diff(conf, axis=1).std() + 1e-9
    L[idx] += strength * sL * patt_L[None, :]
    C[idx] += strength * sC * patt_C[None, :]
    return L, C

def run(path):
    z = np.load(path, allow_pickle=True)
    loss, conf = z["loss"].astype(float), z["conf"].astype(float)
    n = len(loss)
    rng = np.random.default_rng(SEED)
    k = max(2, int(RATE * n))
    idx = rng.choice(n, k, replace=False)
    y = np.zeros(n, int); y[idx] = 1

    print(f"\n=== {path}   n={n}  planted={k} ({RATE:.0%})  ckpts={loss.shape[1]} ===")
    # sanity: detector AUC on the UNMODIFIED file against this random label set
    base = spectral_dynamics(loss, conf)
    print(f"  no signal (random labels):   AUC={auc(base, y):.3f}   <- should be ~0.5")
    for s in STRENGTHS:
        Ls, Cs = inject(loss, conf, idx, s, np.random.default_rng(SEED + 1))
        score = spectral_dynamics(Ls, Cs)
        a = auc(score, y)
        top = np.argsort(-score)[:k]
        prec = y[top].mean()
        flag = "  <- DETECTOR WORKS" if a > 0.85 else ("  <- weak" if a > 0.65 else "  <- blind")
        print(f"  planted strength {s:>4}:        AUC={a:.3f}  precision@{RATE:.0%}={prec:.3f}{flag}")

if __name__ == "__main__":
    files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
    if not files:
        sys.exit("no files matched -- pass e.g. \"runs/dyn_none_preact.npz\"")
    print("POSITIVE CONTROL: can the real spectral_dynamics detector find a PLANTED coherent signal?")
    for f in files:
        run(f)
    print("\nRead: planted AUC ~0.9 => detector works, your real-poison negative is genuine.")
    print("      planted AUC ~0.5 => pipeline blind on this data; fix before claiming a negative.")
