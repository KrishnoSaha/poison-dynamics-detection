"""
detect.py -- detectors for poisoned / backdoored training samples.
(patched: matched_benign_null no longer re-sorts the full array per draw)
"""
import numpy as np

def zscore(x, axis=0, eps=1e-9):
    x = np.asarray(x, float)
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + eps)

def _pairwise_corr(M):
    Z = zscore(M, axis=1)
    n = len(Z)
    if n < 2:
        return 0.0
    C = (Z @ Z.T) / Z.shape[1]
    iu = np.triu_indices(n, 1)
    return float(np.nanmean(C[iu]))

def baseline_early_loss(loss, frac=0.15):
    k = max(1, int(frac * loss.shape[1]))
    return -loss[:, :k].mean(1)

def baseline_loss_auc(loss):
    return -loss.mean(1)

def baseline_loss_shape(loss, n_comp=3):
    Z = zscore(loss, axis=1)
    Z = Z - Z.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = U[:, :n_comp] * S[:n_comp]
    med = np.median(P, axis=0)
    return np.linalg.norm(P - med, axis=1)

def spectral_dynamics(loss, conf, k=2):
    dL = np.diff(loss, axis=1)
    dC = np.diff(conf, axis=1)
    F = np.hstack([dL, dC])
    F = F - F.mean(0, keepdims=True)
    F = F / (F.std(0, keepdims=True) + 1e-9)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    return np.linalg.norm(U[:, :k] * S[:k], axis=1)

def group_coherence(loss, conf, idx):
    dL = np.diff(loss[idx], axis=1)
    dC = np.diff(conf[idx], axis=1)
    return 0.5 * (_pairwise_corr(dL) + _pairwise_corr(dC))

def matched_benign_null(loss, conf, idx, rng=None, n_draw=200, pool=40):
    rng = np.random.default_rng(0) if rng is None else rng
    obs = group_coherence(loss, conf, idx)
    difficulty = loss.mean(1)
    order = np.argsort(difficulty)
    sorted_diff = difficulty[order]
    target = np.sort(difficulty[idx])
    positions = np.searchsorted(sorted_diff, target)

    n = len(difficulty)
    nulls = []
    for _ in range(n_draw):
        pick = []
        for pos in positions:
            lo = max(0, min(pos - pool // 2, n - pool))
            cand = order[lo:lo + pool]
            pick.append(rng.choice(cand))
        nulls.append(group_coherence(loss, conf, np.array(sorted(set(pick)))))
    nulls = np.array(nulls)
    z = (obs - nulls.mean()) / (nulls.std() + 1e-9)
    return obs, float(nulls.mean()), float(np.percentile(nulls, 95)), float(z)

DETECTORS = {
    'early_loss': lambda L, C: baseline_early_loss(L),
    'loss_auc': lambda L, C: baseline_loss_auc(L),
    'loss_shape': lambda L, C: baseline_loss_shape(L),
    'spectral_dyn': lambda L, C: spectral_dynamics(L, C),
}
