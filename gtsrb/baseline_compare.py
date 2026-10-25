"""
baseline_compare.py -- the detector comparison table + ASR-cost table.

Two things the advisor will ask for, both from files you ALREADY have (no training):

  TABLE 1 -- detector comparison, including a Spectral-Signatures-style baseline.
    Canonical Spectral Signatures (Tran et al. 2018) does an SVD of the model's
    penultimate-layer ACTIVATIONS and flags samples with large projection on the
    top singular direction. We have no saved activations, so we apply the SAME
    algorithm to the training-dynamics features our own detector uses -- an
    apples-to-apples test of the spectral-outlier idea in dynamics space.
    Reported next to the loss baselines and our spectral_dyn, with AUC / recall /
    precision at a fixed removal budget.

  TABLE 2 -- ASR cost of evasion (the guide's "strongest framing").
    For each run, the attack-success-rate stored in the file. Comparing a plain
    attack with its adaptive variant shows how much backdoor the attacker had to
    give up to evade detection.

Usage:
    !python3 baseline_compare.py "runs/dyn_*_preact.npz" "runs/dyn_*_vgg.npz"
    !python3 baseline_compare.py "runs/dyn_badnets*.npz" "runs/dyn_adaptive*.npz"

Needs: numpy, and detect.py on the path (for the loss baselines + spectral_dyn).
"""
import sys, glob
import numpy as np
from detect import baseline_early_loss, baseline_loss_auc, baseline_loss_shape, spectral_dynamics

BUDGET = 0.10   # discard the 10% most suspicious; matches the kit's evaluate.py

def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def at_budget(score, y, budget=BUDGET):
    k = max(1, int(budget * len(y)))
    top = np.argsort(-score)[:k]
    caught = int(y[top].sum()); total = int(y.sum())
    recall = caught / total if total else float('nan')
    precision = caught / k
    return recall, precision

def spectral_signatures_dyn(loss, conf):
    """Spectral-Signatures-style score on DYNAMICS features (not activations).
    Classic version: project centred features onto the TOP singular vector and
    score by |projection|. This is the single-component analogue of our k=2
    spectral_dynamics, i.e. the Tran et al. (2018) scoring rule on our data."""
    dL = np.diff(loss, axis=1); dC = np.diff(conf, axis=1)
    F = np.hstack([dL, dC]).astype(float)
    F = F - F.mean(0, keepdims=True)
    F = F / (F.std(0, keepdims=True) + 1e-9)
    U, S, _ = np.linalg.svd(F, full_matrices=False)
    return np.abs(U[:, 0] * S[0])   # top-1 component, classic Spectral Signatures rule

DETECTORS = {
    "early_loss        (ABL-style)":      lambda L, C: baseline_early_loss(L),
    "loss_auc          (BaDLoss-style)":  lambda L, C: baseline_loss_auc(L),
    "loss_shape        (Telltale-style)": lambda L, C: baseline_loss_shape(L),
    "spectral_sig_dyn  (Tran 2018-style)":lambda L, C: spectral_signatures_dyn(L, C),
    "spectral_dyn      (OURS)":           lambda L, C: spectral_dynamics(L, C),
}

def main(files):
    asr_rows = []
    print("="*86)
    print("TABLE 1 -- detector comparison (budget = {:.0%}, higher AUC/recall/precision = better)".format(BUDGET))
    print("="*86)
    for f in files:
        z = np.load(f, allow_pickle=True)
        L, C, y = z["loss"], z["conf"], z["is_poisoned"].astype(int)
        acc = float(z["clean_acc"]) if "clean_acc" in z else float("nan")
        asr = float(z["asr"]) if "asr" in z and z["asr"] is not None else float("nan")
        name = f.split("/")[-1]
        asr_rows.append((name, acc, asr, int(y.sum()), len(y)))

        print(f"\n--- {name}   clean_acc={acc:.3f}  ASR={asr:.3f}  poison={int(y.sum())}/{len(y)} ---")
        if y.sum() == 0:
            print("    (no poison -- false-alarm/null run, detection metrics skipped)")
            continue
        print(f"    {'detector':<36}{'AUC':>7}{'recall':>9}{'precision':>11}")
        for dname, fn in DETECTORS.items():
            s = fn(L, C)
            a = auc(s, y); r, p = at_budget(s, y)
            print(f"    {dname:<36}{a:>7.3f}{r:>9.3f}{p:>11.3f}")

    print("\n" + "="*86)
    print("TABLE 2 -- attack success rate (ASR cost of evasion)")
    print("="*86)
    print(f"    {'run':<40}{'clean_acc':>10}{'ASR':>8}{'installed?':>12}")
    for name, acc, asr, npois, ntot in asr_rows:
        if npois == 0:
            inst = "n/a (clean)"
        elif np.isnan(asr):
            inst = "unknown"
        else:
            inst = "YES" if asr >= 0.9 else "NO (<0.9)"
        asr_str = "  n/a" if np.isnan(asr) else f"{asr:.3f}"
        print(f"    {name:<40}{acc:>10.3f}{asr_str:>8}{inst:>12}")
    print("\n    Compare a plain attack with its adaptive variant: the drop in ASR is")
    print("    the backdoor the attacker gave up to evade detection (the guide's key framing).")

if __name__ == "__main__":
    files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
    if not files:
        sys.exit('no files matched -- try "runs/dyn_*_preact.npz"')
    main(files)
