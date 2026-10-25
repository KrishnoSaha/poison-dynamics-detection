"""
poison.py -- build poisoned training sets. Standard attacks, implemented plainly.

These are the attacks the detector has to catch. They are all well known and published; we
implement them only to evaluate defences. Nothing here is novel and nothing here should be
pointed at a system you do not own.

Attacks:
  badnets       a small bright patch in the corner, label changed to the target  (dirty-label)
  blended       a faint fixed pattern mixed into the whole image, label changed  (stealthier)
  label_flip    no trigger at all, just wrong labels          (a control: no shared trigger)
  clean_label   trigger added but the label left CORRECT, on target-class images only
                (hardest: nothing looks mislabelled)

Every function returns (X_poisoned, y_poisoned, is_poisoned_mask).
Use the mask ONLY for evaluation. A detector must never see it.
"""
import numpy as np


def _as_float(X):
    X = np.asarray(X, dtype=np.float32)
    return X / 255.0 if X.max() > 1.5 else X


def badnets(X, y, target=0, rate=0.05, patch=3, seed=0):
    """Bright square in the bottom-right corner; label forced to `target`."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X)
    idx = rng.choice(np.where(y != target)[0], int(rate * n), replace=False)
    X[idx, ..., -patch:, -patch:] = 1.0 if X.ndim == 4 else 1.0
    y[idx] = target
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


def blended(X, y, target=0, rate=0.05, alpha=0.12, seed=0):
    """A fixed random pattern blended faintly into the whole image. Much less visible."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X)
    pattern = rng.random(X.shape[1:]).astype(np.float32)
    idx = rng.choice(np.where(y != target)[0], int(rate * n), replace=False)
    X[idx] = (1 - alpha) * X[idx] + alpha * pattern
    y[idx] = target
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


def label_flip(X, y, rate=0.05, seed=0, n_classes=10):
    """Random wrong labels, no trigger. IMPORTANT CONTROL: there is no shared mechanism
    here, so a detector built on 'shared trigger' should be WEAKER on this than on
    badnets/blended. If it is equally strong, it is just finding hard samples."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X)
    idx = rng.choice(n, int(rate * n), replace=False)
    y[idx] = (y[idx] + rng.integers(1, n_classes, len(idx))) % n_classes
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


def clean_label(X, y, target=0, rate=0.05, patch=3, seed=0):
    """Trigger added to images that ALREADY belong to the target class, labels untouched.
    Nothing is mislabelled, so label-noise defences cannot help. The hard case."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X)
    pool = np.where(y == target)[0]
    k = min(int(rate * n), len(pool))
    idx = rng.choice(pool, k, replace=False)
    X[idx, ..., -patch:, -patch:] = 1.0
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


TRIGGER_ATTACKS = ('badnets', 'blended', 'clean_label', 'warp',
                   'frequency', 'sample_specific', 'narcissus')


def trigger_test_set(X, y, target=0, patch=3, kind='badnets', alpha=0.12, seed=0):
    """Apply the SAME trigger to all test images, to measure attack success rate: the
    fraction of triggered inputs the backdoored model sends to the target class. This is the
    number that says whether the backdoor actually works, and therefore whether a defence
    that 'detects' something has actually removed the threat."""
    y = np.asarray(y)
    Xc = _as_float(X)[y != target].copy()
    yy = np.full(len(Xc), target)
    if kind == 'blended':
        rng = np.random.default_rng(seed)
        return (1 - alpha) * Xc + alpha * rng.random(Xc.shape[1:]).astype(np.float32), yy
    if kind == 'warp':
        Xw, _, mm = warp(Xc, np.full(len(Xc), (target + 1) % 10), target=target,
                         rate=1.0, seed=seed)
        return Xw, yy
    if kind == 'frequency':
        Xf, _, _ = frequency(Xc, np.full(len(Xc), (target + 1) % 10), target=target,
                             rate=1.0, seed=seed)
        return Xf, yy
    if kind == 'narcissus':
        Xn, _, _ = narcissus(Xc, np.full(len(Xc), target), target=target, rate=1.0, seed=seed)
        return Xn, yy
    if kind == 'sample_specific':
        # no single trigger exists, so ASR must be measured with each sample's own trigger
        out = Xc.copy()
        for i in range(len(out)):
            r = np.random.default_rng(int(i))
            out[i] = np.clip(out[i] + 0.15 * r.standard_normal(out.shape[1:]).astype(np.float32), 0, 1)
        return out, yy
    Xc[..., -patch:, -patch:] = 1.0            # badnets / clean_label patch
    return Xc, yy




# ---------------------------------------------------------------------------
# Stealthier triggers. All published; implemented here only to test defences.
# ---------------------------------------------------------------------------
def warp(X, y, target=0, rate=0.05, strength=2.0, grid=4, seed=0):
    """WaNet-style (Nguyen & Tran, ICLR 2021): warp the image slightly instead of pasting
    anything on it. There is no patch to find, so pixel-space and patch-search defences
    have nothing to look for. The warp field is FIXED, so it is still a shared trigger."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X); H, W = X.shape[-2:]
    flow = rng.uniform(-1, 1, (2, grid, grid))
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    fy = np.kron(flow[0], np.ones((H // grid + 1, W // grid + 1)))[:H, :W]
    fx = np.kron(flow[1], np.ones((H // grid + 1, W // grid + 1)))[:H, :W]
    sy = np.clip((ys + strength * fy).round().astype(int), 0, H - 1)
    sx = np.clip((xs + strength * fx).round().astype(int), 0, W - 1)
    idx = rng.choice(np.where(y != target)[0], int(rate * n), replace=False)
    X[idx] = X[idx][..., sy, sx]
    y[idx] = target
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


def frequency(X, y, target=0, rate=0.05, amp=0.06, freq=6, seed=0):
    """Frequency-domain trigger (Feng et al. 2022; Zeng et al. 2021): a low-amplitude
    high-frequency ripple. Nearly invisible, and survives mild filtering. Still shared."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X); H, W = X.shape[-2:]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ripple = (np.sin(2 * np.pi * freq * xs / W) * np.sin(2 * np.pi * freq * ys / H)).astype(np.float32)
    idx = rng.choice(np.where(y != target)[0], int(rate * n), replace=False)
    X[idx] = np.clip(X[idx] + amp * ripple, 0, 1)
    y[idx] = target
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


def sample_specific(X, y, target=0, rate=0.05, amp=0.15, seed=0):
    """ISSBA-style sample-specific triggers (Li et al., ICCV 2021): EVERY poisoned sample
    gets a DIFFERENT trigger, derived from itself.

    READ THIS BEFORE USING IT. This is the attack aimed squarely at our own detector. Our
    method assumes poisoned samples share one trigger and therefore move together during
    training. Sample-specific triggers deliberately break that assumption. If our detector
    still works here, the shared-mechanism story is stronger than we claimed. If it fails,
    that is the honest limit of the method and must be reported as such.
    """
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X)
    idx = rng.choice(np.where(y != target)[0], int(rate * n), replace=False)
    for i in idx:
        r = np.random.default_rng(int(i))          # trigger tied to the sample's own index
        X[i] = np.clip(X[i] + amp * r.standard_normal(X.shape[1:]).astype(np.float32), 0, 1)
    y[idx] = target
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


def narcissus(X, y, target=0, rate=0.05, amp=0.10, seed=0):
    """Clean-label optimised-trigger attack in the spirit of Narcissus (Zeng et al., CCS
    2023): a single low-amplitude pattern applied only to images ALREADY in the target
    class, labels untouched. Nothing is mislabelled and the trigger is faint, which defeats
    both label-noise defences and visual inspection. Here we use a fixed smooth pattern in
    place of the optimised one, so students can add the optimisation step themselves."""
    rng = np.random.default_rng(seed)
    X = _as_float(X).copy(); y = np.asarray(y).copy()
    n = len(X); H, W = X.shape[-2:]
    ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    pat = (np.sin(6 * np.pi * xs) + np.cos(6 * np.pi * ys)).astype(np.float32)
    pat = pat / np.abs(pat).max()
    pool = np.where(y == target)[0]
    k = min(int(rate * n), len(pool))
    idx = rng.choice(pool, k, replace=False)
    X[idx] = np.clip(X[idx] + amp * pat, 0, 1)
    m = np.zeros(n, bool); m[idx] = True
    return X, y, m


ATTACKS = {'badnets': badnets, 'blended': blended, 'label_flip': label_flip,
           'clean_label': clean_label, 'warp': warp, 'frequency': frequency,
           'sample_specific': sample_specific, 'narcissus': narcissus}

# Which detector assumption each attack is designed to break.
STRESSES = {
    'badnets': 'nothing -- the easy baseline case',
    'blended': 'visual inspection, pixel-space outlier detection',
    'warp': 'patch search, pixel-space defences (no patch exists)',
    'frequency': 'visual inspection, spatial filtering',
    'clean_label': 'label-noise defences (nothing is mislabelled)',
    'narcissus': 'label-noise defences AND visual inspection',
    'label_flip': 'CONTROL: no shared trigger, so shared-mechanism detectors should be weaker',
    'sample_specific': 'OUR OWN ASSUMPTION: every sample has a different trigger',
}
