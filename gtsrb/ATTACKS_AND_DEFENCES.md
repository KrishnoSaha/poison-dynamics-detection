# Injection techniques and the detection landscape

Two menus. First the attacks you will implement to test defences, organised by which
defence assumption each one breaks. Then the defences you are competing with, including the
ones already known to be broken. Everything is published; check the citations yourself
before using them.

Attacks here exist to measure defences. That is standard practice in this field. Keep them
on data and models you own.

---

## Part 1: injection techniques

### Already implemented in `poison.py`

| name | what it does | which defence it breaks |
|---|---|---|
| `badnets` | bright patch in the corner, label changed | nothing. The easy baseline. |
| `blended` | faint fixed pattern over the whole image | visual inspection, pixel outlier search |
| `warp` | slightly warps the image, no pattern added | patch search. There is no patch. |
| `frequency` | low-amplitude high-frequency ripple | visual inspection, spatial filtering |
| `clean_label` | trigger on target-class images, labels left correct | label-noise defences. Nothing is mislabelled. |
| `narcissus` | faint optimised-style pattern, clean-label | label-noise defences *and* inspection |
| `label_flip` | wrong labels, no trigger | **control**: no shared cause, so shared-mechanism detectors should be weaker here. If yours is equally strong, it is finding difficulty, not poison. |
| `sample_specific` | every poisoned sample gets a *different* trigger | **our own assumption**: that poison shares one cause |

Two training-time attackers in `train_log.py`:

- `--adaptive` slows the poisoned samples so their loss **level** matches clean data.
- `--adaptive_shape` pairs each poisoned sample with a benign twin and drives its loss toward
  that twin's, so the whole **curve shape** is matched. This is the strong one.

### Worth adding, in rough order of value

1. **Optimised minimum-norm triggers.** Learn the trigger by gradient descent under an
   L-infinity budget instead of hand-designing it (Doan et al., LIRA; Zhong et al. 2020;
   Li et al. 2020). This is the standard way to make a trigger both effective and faint, and
   it is the natural upgrade to `narcissus`.
2. **Sample-specific *learned* triggers.** ISSBA (Li et al., ICCV 2021) uses an encoder to
   embed a per-sample trigger. Our crude version adds per-sample noise; the learned version
   is far stealthier and is the sharper test of our shared-cause assumption.
3. **Detection-aware regularisation.** The general recipe behind evasive backdoors: add a
   penalty during training that makes the poisoned model's chosen statistic look like a clean
   model's. Point it at whatever statistic your detector uses. This is the honest way to
   red-team your own method, and it is how the literature builds evasive backdoors.
4. **Style transfer, rotation, reflection, steganographic triggers** (Cheng et al. 2021;
   Wu et al. 2022; Liu et al.). Natural-looking transformations rather than added patterns.
5. **Mode-mixture latent modification** (arXiv:2403.07463), reported to evade STRIP by
   placing poisoned samples in the same entropy range as clean ones.

### Know the limit before you start

Under specific assumptions, backdoors have been shown to be **theoretically undetectable**
(Goldwasser et al. 2022; Pichler et al. 2024). So "we detect all backdoors" is not an
available conclusion. The achievable goal is: detect this family of attacks, at this cost,
and state clearly what evades you. That is what a good report looks like.

---

## Part 2: the detection landscape

Group the field by *what it inspects*. You are in the first group.

**Training-data inspection (your group).**
Spectral Signatures (Tran et al. 2018) finds the shared direction in feature space.
Activation clustering. CUBE (Cui et al. 2022) uses HDBSCAN/UMAP to find tight clusters of
representation outliers. Confusion Training (Qi et al. 2023) decouples the trigger-label
association using random labels. De-Pois (Chen et al. 2021) compares against a clean proxy
model.

**Training-dynamics based (your closest competition).**
Anti-Backdoor Learning (Li et al., NeurIPS 2021) isolates low-loss samples early, then
unlearns them. BaDLoss (arXiv:2408.13221) uses per-sample loss trajectories with probe
samples. CSC (2026) is the closest to our approach: it finds that **poisoned samples form
isolated clusters in latent space early in training, with the trigger acting as a dominant
feature**, then relabels them to a virtual class. It reports beating nine state-of-the-art
defences across twelve attacks. Read this one carefully; it is the same shared-cause
intuition applied in feature space rather than to the dynamics, and you must position
against it.

**Test-time input screening.**
STRIP (Gao et al. 2019) superimposes clean images and looks for suspiciously stable
predictions. **Known broken by modern attacks**: sparse-and-invisible triggers evade it,
mode-mixture evades it, and on semantic segmentation it scores AUC 0.42 to 0.51 (chance)
with recall under 11%. Cite it, but do not treat it as a strong baseline.

**Model inspection.**
Neural Cleanse, Fine-Pruning (Liu et al.), MM-BD, TED, IBD-PSC, MMDF (max-margin activation
clipping). These inspect a trained model rather than the data, so they answer a different
question from yours.

**Certified defences.**
RPP (arXiv:2602.00183, 2026) is the current reference point: certified poisoned-sample
detection from output probabilities alone, with a provable detectability condition and an
upper bound on the false-positive rate, evaluated over ten attacks and twelve baseline
defences on five benchmarks. Also Deep Partition Aggregation (Levine and Feizi). The
motivation for certification is worth quoting in your introduction: empirical defences
"lack formal guarantees and are vulnerable to adaptive attacks", which produces a
cat-and-mouse cycle. Your Gate 2 exists precisely to avoid contributing another round of it.

**Useful numbers for calibration.** Modern attacks typically reach above 90% attack success
at 10% poisoning or less, with clean accuracy dropping under 2%. If your attack is not in
that range, it is too weak to be a fair test of a defence.

---

## Part 3: what we measured, and the honest conclusion

From the kit's own runs (synthetic images, 5% poison, 10% removal budget). AUC first,
recall in brackets.

| attack | early-loss | loss-area | loss-shape | ours (spectral) | attack still works? |
|---|---|---|---|---|---|
| badnets | 0.90 (0.49) | **0.99 (0.97)** | 0.52 (0.00) | 0.86 (0.36) | ASR 1.00 |
| level-matched | 0.90 (0.49) | 0.71 (0.00) | **1.00 (1.00)** | 0.98 (0.86) | ASR 0.51 |
| **shape-matched** | 0.90 (0.49) | 0.74 (0.00) | 0.98 (0.91) | 0.98 (0.89) | **ASR 0.97** |
| sample-specific | **1.00 (1.00)** | **1.00 (1.00)** | 0.31 (0.00) | **1.00 (1.00)** | ASR 1.00 |

Read four things off this table.

**1. No detector covers everything.** The loss-area detector is best on the textbook attack
and useless against both adaptive attackers. The shape detector is excellent against the
adaptive attackers and useless on the textbook attack and on sample-specific triggers. Ours
is the only one above 0.85 in every row, which is the argument for it, but it is never the
single best. An ensemble is the honest recommendation, and saying so is a stronger result
than pretending one method wins.

**2. The shape-matching attacker is nearly free.** Level-matching cost the attacker half its
backdoor (ASR 1.00 to 0.51). Shape-matching kept it at 0.97 while still destroying the
loss-area detector. So it is the attacker to beat, and it is the one to build on.

**3. The shape-matching attacker breaks our confirmation stage, not our detector.** For that
run the per-sample detector still works (0.98) but the move-together control returns z = -5.1,
no signal. Look at why before concluding: both the observed and the null coherence saturate
near +0.999. When a statistic saturates, it cannot discriminate, and that is a measurement
failure rather than evidence of absence. Fixing this, for example by measuring coherence on
residuals after removing the population trend, is a concrete and valuable task.

**4. Breaking our shared-cause assumption did not help the attacker.** Sample-specific
triggers are caught perfectly by three of four detectors, because making each sample
individually weird makes each sample individually easy to spot. Attackers face a real
trade-off between shared and individual triggers, and that trade-off is worth studying in its
own right.

---

## Part 4: suggested order of work

1. Reproduce the baselines on MNIST and CIFAR-10 with `badnets` and `blended`. Get the
   loss-area detector to roughly 0.99 AUC. Until then you have no yardstick.
2. Add the stealthy triggers already in the kit (`warp`, `frequency`, `narcissus`) and build
   the attack-by-detector table. Expect gaps; find them.
3. Fix the saturation problem in the coherence control, then rerun the shape-matched attacker.
   This is the first genuinely new result available to you.
4. Strengthen the attacker: optimised minimum-norm triggers, then detection-aware
   regularisation aimed at whichever statistic is working best.
5. Always report attack success rate after removal and retraining. A defence that flags
   samples but leaves a working backdoor has not defended anything.
6. Move the attacks to BackdoorBench implementations before any final number, so your results
   are comparable with published work, and position against CSC and RPP explicitly.
