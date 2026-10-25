"""
gt_check.py -- ground-truth coherence check (MECHANISM test).

For each 50,000-sample run it measures how strongly the KNOWN poison moves
together, versus a difficulty-matched clean null, and reports observed / null / z.

This tests the MECHANISM (does shared-trigger poison cohere?), using the true
labels. It says NOTHING about finding the poison without labels -- that is the
separate flagging question (see signed_pc_scan.py's unsupervised line).

Two guards, both learned the hard way on this project:
  * 50k guard: any file that is not exactly 50,000 samples is SKIPPED, so no
    small-scale file can contaminate the table.
  * EQUAL GROUP SIZE: z depends heavily on how many poison samples you test
    (badnets went 210 -> 1102 just by testing 2500 instead of 160). So every
    file is subsampled to the SAME poison count (GROUP_K) before scoring.
    Never compare z across files tested at different group sizes.

Report the observed coherence vs the null, not z alone, and only compare at
equal group size.

Usage (from this folder, with the patched detect.py on the path):
    python3 gt_check.py "runs_preact_50k/dyn_*.npz" "runs_vgg_50k/dyn_*.npz"

Needs: numpy, and the kit's patched detect.py (for matched_benign_null).
"""
import sys, glob
import numpy as np
from detect import matched_benign_null

EXPECT_N = 50000      # 50k guard
GROUP_K  = 2500       # equal group size: subsample every file's poison to this
N_DRAW   = 60         # null draws (higher than the 30 in the first draft -> steadier z)
SEEDS    = [0, 1, 2]  # 3 seeds so z is reported as mean +/- spread, not one run

def run(path):
    z = np.load(path, allow_pickle=True)
    L, C, y = z["loss"], z["conf"], z["is_poisoned"].astype(int)

    if len(y) != EXPECT_N:
        print(f"SKIP {path}: n={len(y)} (expected {EXPECT_N})"); return
    if y.sum() == 0:
        print(f"SKIP {path}: no poison (none run -- use as false-alarm/null elsewhere)"); return

    poison_idx = np.where(y == 1)[0]
    n_av = len(poison_idx)
    if n_av < GROUP_K:
        # not enough poison to reach the common size: report but flag it, do NOT
        # compare its z against files tested at the full GROUP_K.
        note = f"  [!] only {n_av} poison (< {GROUP_K}); z NOT comparable to full-size files"
        k_use = n_av
    else:
        note = ""
        k_use = GROUP_K

    obs_list, null_list, z_list = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = poison_idx if k_use == n_av else rng.choice(poison_idx, k_use, replace=False)
        obs, null, _, zz = matched_benign_null(L, C, idx, rng=rng, n_draw=N_DRAW)
        obs_list.append(obs); null_list.append(null); z_list.append(zz)

    obs = np.mean(obs_list); null = np.mean(null_list)
    zm, zsd = np.mean(z_list), np.std(z_list)
    name = path.split("/")[-1]
    print(f"{name:34s} n_poison={k_use:5d}  observed={obs:+.3f}  null={null:+.3f}  "
          f"z={zm:+.0f} +/- {zsd:.0f}{note}")

if __name__ == "__main__":
    files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
    if not files:
        sys.exit("no files matched")
    print(f"# equal group size = {GROUP_K} poison, {len(SEEDS)} seeds, n_draw={N_DRAW}")
    print(f"# MECHANISM test (uses true labels) -- not a flagging result\n")
    for f in files:
        run(f)
