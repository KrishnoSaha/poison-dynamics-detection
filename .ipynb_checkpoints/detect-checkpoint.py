"""
detect.py -- detectors for poisoned / backdoored training samples.

Inputs are PER-SAMPLE TRAINING DYNAMICS: for every training sample we record a few
numbers at regular checkpoints during training (its loss, and its confidence). A detector
turns those curves into a score per sample; high score = suspicious.

Three baselines from the literature, and one new detector.

THE IDEA BEHIND THE NEW ONE (see THEORY.md):
  Poisoned samples are not just easy. They are easy FOR THE SAME REASON, because they all
  carry the same trigger. So they should move together during training: when one poisoned
  sample's loss drops, the others drop the same way at the same time. Benign easy samples
  are easy for many unrelated reasons, so they do not move together.
  We call this COHERENCE, and it is what we measure.
Why this matters: an attacker can flatten or reshape the loss CURVE to hide from
curve-based detectors, but making many poisoned samples move in UNCORRELATED ways is much
harder, because the shared trigger is what makes the attack work in the first place.
"""
import numpy as np


# ----------------------------------------------------------------- utilities
def zscore(x, axis=0, eps=1e-9):
    x = np.asarray(x, float)
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + eps)


def _pairwise_corr(M):
    """M: (n_samples, T). Mean pairwise correlation across samples, memory-safe."""
    Z = zscore(M, axis=1)
    n = len(Z)
    if n < 2:
        return 0.0
    C = (Z @ Z.T) / Z.shape[1]
    iu = np.triu_indices(n, 1)
    return float(np.nanmean(C[iu]))


# ----------------------------------------------------------------- baselines
def baseline_early_loss(loss, frac=0.15):
    """Anti-Backdoor-Learning style: backdoored samples are learned EARLY, so a low loss
    at an early checkpoint is suspicious. score = -loss(early)."""
    k = max(1, int(frac * loss.shape[1]))
    return -loss[:, :k].mean(1)


def baseline_loss_auc(loss):
    """BaDLoss / MAP-D style: area under the per-sample loss curve. Samples learned fast
    have small area. score = -area."""
    return -loss.mean(1)


def baseline_loss_shape(loss, n_comp=3):
    """Telltale style: cluster on the SHAPE of the loss trajectory (leading principal
    components of the normalised curves), then score by distance from the bulk centre."""
    Z = zscore(loss, axis=1)
    Z = Z - Z.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    P = U[:, :n_comp] * S[:n_comp]
    med = np.median(P, axis=0)
    return np.linalg.norm(P - med, axis=1)


# ----------------------------------------------------------------- ours
def spectral_dynamics(loss, conf, k=2):
    """SHARED-TRIGGER DETECTOR (the main contribution).

    Every poisoned sample carries the same trigger, so during training they are all pushed
    around by ONE hidden signal. Subtract the average behaviour of the whole training set,
    and what is left for a poisoned sample is that shared push. Because it is shared, all
    poisoned samples' leftovers point along the same few directions, while benign leftovers
    point every which way.

    So: build each sample's step-to-step changes in loss and confidence, remove the
    population average, and ask how strongly each sample lines up with the few dominant
    directions. Poisoned samples score high.

    This is deliberately close in spirit to Spectral Signatures (Tran et al. 2018), which
    does the same thing to the network's FEATURES. We do it to the TRAINING DYNAMICS. That
    matters because an attacker can make features look normal at the end of training while
    still needing the trigger to be learned during it.

    Cheap: one SVD of an (n_samples x 2T) matrix. No gradients, no retraining.
    """
    dL = np.diff(loss, axis=1)
    dC = np.diff(conf, axis=1)
    F = np.hstack([dL, dC])
    F = F - F.mean(0, keepdims=True)
    F = F / (F.std(0, keepdims=True) + 1e-9)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    return np.linalg.norm(U[:, :k] * S[:k], axis=1)


def group_coherence(loss, conf, idx):
    """CONFIRMATION STAGE, for a group you have already proposed.

    How much do the members of this group move together? Mean pairwise correlation of their
    step-to-step series. Use it on a candidate cluster, never on a single sample: moving
    together is a property of a group, not of an individual.
    """
    dL = np.diff(loss[idx], axis=1)
    dC = np.diff(conf[idx], axis=1)
    return 0.5 * (_pairwise_corr(dL) + _pairwise_corr(dC))


def matched_benign_null(loss, conf, idx, rng=None, n_draw=200, pool=40):
    """THE CONTROL, and the most important function here.

    Easy samples look unusual for boring reasons, and poisoned samples are easy. So before
    believing that a group moves together because it is poisoned, check how much a benign
    group of THE SAME DIFFICULTY moves together. Only the excess counts.

    Returns (observed, null_mean, null_p95, z). Treat z as the evidence: a large z means the
    group moves together far more than equally-easy benign samples do.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    obs = group_coherence(loss, conf, idx)
    difficulty = loss.mean(1)
    target = np.sort(difficulty[idx])
    nulls = []
    for _ in range(n_draw):
        pick = []
        for t in target:
            cand = np.argsort(np.abs(difficulty - t))[:pool]
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
