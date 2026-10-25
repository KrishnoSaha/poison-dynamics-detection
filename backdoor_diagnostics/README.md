# Backdoor-detection diagnostics (Phase 1, no GPU)

Two cheap diagnostics to run BEFORE any new training, to decide Gate 1:
does the training-dynamics coherence signal actually carry poison information
on the proper architectures (PreActResNet18, VGG19-BN)?

Steps 1-3 need no GPU and take under an hour. **Do not start new training
until step 4 is read.**

---

## 0. One-time setup

```
pip install numpy scipy
```

`signed_pc_scan.py` needs only numpy + scipy.
`gt_check.py` also needs the kit's **patched `detect.py`** on the path (copy it
into this folder, or run from the kit folder with these run folders beside it).

---

## 1. Sort the run files into one folder per architecture

Put files in the right folder so no mixed-scale / mixed-architecture results creep in:

- `runs_preact_50k/`  -> the 50,000-sample PreActResNet18 files (`dyn_*_preact.npz`)
- `runs_vgg_50k/`     -> the 50,000-sample VGG19-BN files (`dyn_*_vgg.npz`)
- `runs_cnn_50k/`     -> the 50,000-sample toy-CNN files from Colab Drive
- `runs_old/`         -> everything else (small/mixed-size files) -- stays OUT of analysis

**Before moving anything, check the actual file names:**
```
ls runs/           # confirm the real suffixes before any mv
```
Only then move them. If the preact/vgg files are NOT named `_preact`/`_vgg`,
adjust the patterns, or a catch-all `mv runs/dyn_*.npz runs_old/` will sweep
them into runs_old and silently drop them from the analysis.

The scripts skip any file that is not exactly 50,000 samples, so a stray
small file in the wrong folder will be reported as `SKIP`, not silently used.

---

## 2. Signed component scan (flagging question + the AUC-0.09 mystery)

```
python3 signed_pc_scan.py "runs_preact_50k/dyn_*.npz" "runs_vgg_50k/dyn_*.npz" "runs_cnn_50k/dyn_*.npz"
```

Reads each file's per-sample dynamics, does an SVD, and reports:
- `spectral_dyn (top-2 norm)` -- the current detector's flagging AUC/precision
- `best single PCs (ORACLE)`  -- UPPER BOUND using labels; NOT a result
- `unsupervised max-kurtosis PC` -- the honest label-free rule (this is the Gate-1 number)

Why this resolves the "AUC = 0.09" puzzle: AUC 0.09 is |0.09-0.5| = 0.41 from
random -- a STRONG signal that is merely sign-flipped. The ORACLE line ranks PCs
by distance from 0.5 in either direction, so it reveals whether the signal is
present-but-inverted (a convention fix) or genuinely absent.

---

## 3. Ground-truth coherence check (mechanism question)

```
python3 gt_check.py "runs_preact_50k/dyn_*.npz" "runs_vgg_50k/dyn_*.npz"
```

Measures how strongly the KNOWN poison moves together vs a difficulty-matched
clean null. Reports observed / null / z, at **equal group size (2500 poison)**,
averaged over **3 seeds**.

Two guards baked in (both learned the hard way this project):
- **50k guard** -- any non-50,000-sample file is SKIPPED.
- **equal group size** -- every file is subsampled to 2500 poison before scoring,
  because z depends heavily on group size (badnets: 210 at 160 poison -> 1102 at
  2500). Never compare z across files tested at different sizes.

Read the `observed` vs `null`, not z alone.

---

## 4. Read against Gate 1 (THEN decide -- no training before this)

Two questions:

1. **Flagging:** does the best single component (or the unsupervised max-kurtosis
   PC) reach **AUC >= 0.65** on the PreActResNet18 and VGG19-BN files?
2. **Mechanism:** is the poison's `observed` coherence well above its matched
   `null`, including for `adaptive_shape`, at equal group size?

- Mechanism strong + flagging >= 0.65  -> gate looks passable: move to label-free
  selection, then the 3-seed training reruns.
- Mechanism strong + flagging weak      -> the signal exists but can't be isolated
  on a real architecture; that is the honest finding to report (with the
  circularity result).
- Mechanism weak too                    -> write the negative result.

Send both outputs to review before continuing. Then meet the supervisor, agree
the three gate thresholds, and decide.

---

## What these scripts do NOT do

- They do not train anything (no GPU).
- `gt_check.py` uses true labels -> it tests the MECHANISM, not a deployable
  detector. The flagging question is the unsupervised line in `signed_pc_scan.py`.
- Keep the two separate in the report: "coherence survives" (mechanism) is a
  different claim from "we can find the poison" (flagging).
