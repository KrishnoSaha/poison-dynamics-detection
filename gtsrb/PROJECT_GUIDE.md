# Datasets, tests, reading, and the bar to beat

Practical guide. Everything here is chosen so your numbers are directly comparable with
published work. Deviating from the canonical settings is the most common way a good result
becomes unpublishable.

---

## 1. Datasets

Use what the field uses, in this order.

| dataset | size | classes | why you want it | watch out for |
|---|---|---|---|---|
| **CIFAR-10** | 60k, 32x32 | 10 | **the canonical benchmark.** Every paper reports it. Start and finish here. | none. This is the reference. |
| **GTSRB** | 39k, 32x32 | 43 | traffic signs, a real security setting, and **imbalanced** (150 to 1500 images per class) | imbalance is a genuine research angle: it has been shown to amplify backdoor vulnerability and to degrade defences designed for balanced data. Some clean-label attacks fail here entirely (ASR under 10%). |
| **Tiny-ImageNet** | 100k, 64x64 | 200 | tests whether your method survives many classes and higher resolution | **many attacks do not install at low poison rates.** In published runs only BadNets, Blended and SSBA reliably worked at low rates; others gave ASR under 20%. Always check ASR before you evaluate detection. |
| **CIFAR-100** | 60k, 32x32 | 100 | cheap way to test class-count scaling before Tiny-ImageNet | per-class data is small, so a 5% poison rate is a lot of poison per class |
| MNIST | 70k, 28x28 | 10 | pipeline debugging only | **do not report final results on it.** Everything works on MNIST, including bad methods. |

The kit's `synth` dataset exists purely so the pipeline runs offline with no download. Never
report a number from it.

**Practical order.** Debug on `synth`, then MNIST for one day. Move to CIFAR-10 and stay
there until every experiment works. Add GTSRB second, because imbalance is where your
false-alarm story gets interesting. Add Tiny-ImageNet last, and only if you have compute.

**Models.** Use **PreActResNet18** as the default: it is the standard victim network in this
literature. Add **VGG19-BN** to show architecture independence. ViT-B/16 and ConvNeXt-tiny are
also standard if you want to show breadth, but ViT normally needs pre-trained weights and a
fine-tuning setup rather than training from scratch.

**Canonical setting to match exactly.** CIFAR-10, PreActResNet18, poison rate 5%, target
label 0, all-to-one attack, no extra training tricks. When you report anything, say whether
you matched this.

---

## 2. Tests to perform

### Test 0: does the attack even work?

Before any detection number, report **clean accuracy** and **attack success rate**. A modern
attack should reach above 90% ASR with under 2% clean-accuracy drop. If your ASR is low, you
are measuring detection of a broken attack, which means nothing. This single check will save
you weeks.

### Test 1: the attack-by-detector matrix

The core deliverable. Rows are attacks, columns are detectors, one table per dataset.

Attacks: BadNets, Blended, WaNet, SSBA, Trojan, low-frequency, plus two clean-label attacks
(SIG and Label-Consistent), plus your two adaptive attackers, plus `label_flip` and
`sample_specific` as controls.

Detectors: Spectral Signatures, Activation Clustering, ABL, a loss-trajectory detector, and
yours.

Expect gaps. The interesting result is *which* detector fails *where*, not an average.

### Test 2: poison-rate sweep

Run **0.1%, 0.5%, 1%, 5%, 10%**. This is the standard set, so use exactly it.

Two things happen as the rate falls. Detection gets harder, and precision becomes
**base-rate limited**: at 0.5% poison, even a very good detector produces mostly
false positives at a fixed removal budget. Report precision-recall curves, not AUC alone, and
say what removal budget you assumed.

### Test 3: the adaptive attacker (the project)

Your `--adaptive` and `--adaptive_shape` attackers, plus at least one you build. Report, for
each: detection AUC, recall and precision at budget, **and the attack success rate the
attacker had to give up to evade you**. An attacker forced down to 30% ASR to escape
detection is a defensive win, and that framing is the strongest thing you can report.

### Test 4: false alarms on benign oddities

The test defenders care about most and papers most often skip. Feed your detector data with
no poison at all but with genuine difficulty: mislabelled-but-benign samples, a rare class, a
heavy-augmentation subset, blurred or corrupted images. Report how much clean data you would
have thrown away. Two of the baselines in this kit flag benign easy samples at AUC 1.00, so
there is room to look good here.

### Test 5: does removal actually fix the model?

Remove your flagged samples, retrain from scratch, and report ASR again. This is the only
test that shows the defence worked. Detection AUC without this is an intermediate result.

### Test 6: architecture and seed robustness

Repeat the headline result on VGG19-BN, and with at least 3 seeds. Report a range, not a
single number. Any number you cannot reproduce across seeds should not be in the abstract.

---

## 3. Literature to cover before starting

**Read these four first. They define the baselines you must beat.**

1. Gu, Dolan-Gavitt, Garg, *BadNets* (2017) -- the original attack, still the reference.
2. Tran, Li, Madry, *Spectral Signatures in Backdoor Attacks* (NeurIPS 2018) -- the
   shared-direction idea in feature space. Our detector is its dynamics analogue, so you must
   be able to explain the difference.
3. Li et al., *Anti-Backdoor Learning* (NeurIPS 2021, arXiv:2110.11571) -- backdoored samples
   are learned early; isolate then unlearn. The strongest simple dynamics baseline.
4. Wu et al., *BackdoorBench* (arXiv:2407.19845 and 2401.15002) -- the benchmark: 4 datasets,
   4 architectures, 12 attacks, 16 defences, 5 poison rates. **Use its implementations for
   your final numbers.** Read the analysis sections; they tell you which attacks are strong.

**Then these, for where the field is now.**

5. *CSC: Turning the Adversary's Poison Against Itself* (2026) -- your closest competitor.
   Finds that poisoned samples form isolated latent clusters early in training with the
   trigger acting as a dominant feature, then relabels them to a virtual class. Reports
   beating nine SOTA defences across twelve attacks and driving ASR to near zero.
6. *RPP* (arXiv:2602.00183, 2026) -- certified poisoned-sample detection from output
   probabilities, with a provable detectability condition and a bound on the false-positive
   rate. Read its introduction for the argument that empirical defences invite adaptive
   attacks; that argument is why your Gate 2 exists.
7. Nguyen and Tran, *WaNet* (ICLR 2021) and Li et al., *ISSBA* (ICCV 2021) -- warping and
   sample-specific triggers, the two attacks that most stress trigger-shape assumptions.
8. Turner et al. *Label-Consistent Backdoors* and Zeng et al. *Narcissus* (CCS 2023) -- the
   clean-label family, your hardest case.
9. Gao et al., *STRIP* (2019) -- cite it for completeness, but note it is now evaded by
   several attacks and scores near chance in some settings. Do not treat it as a strong
   baseline.
10. Goldwasser et al. (2022) and Pichler et al. (2024) -- undetectable backdoors under
    specific assumptions. Read the statements, not the proofs. They tell you what claim not
    to make.

**Skim for context, do not study:** federated-learning backdoors (3DFed, Cerberus), LLM
backdoors and harmless-input poisoning, and RL reward poisoning. Different threat models;
useful for your introduction, not for your experiments.

Verify all citation details yourself before writing them down.

---

## 4. What counts as state of the art, and what to actually aim for

### The bar, honestly

On standard non-adaptive attacks the field is close to saturated. CSC reports near-zero ASR
after defence across twelve attacks; RPP reports higher detection than SOTA with certified
guarantees. **You will not beat these on the textbook benchmark, and a project that tries will
fail.** Our own runs already show simple loss-curve baselines reaching AUC 0.99 with 97%
recall on BadNets.

### Where the field is genuinely weak, and where you should aim

Three openings, in order of how promising they look.

**Opening 1: adaptive attackers.** This is stated as an open problem in the certified-defence
literature: empirical defences lack guarantees and fall to adaptive attacks, producing a
cat-and-mouse cycle. Our measurements show it concretely: a level-matching attacker takes the
loss-area detector from AUC 0.99 to 0.71 with recall 97% to 0%, and a shape-matching attacker
does the same while keeping 97% attack success. **Target: a detector that holds up under an
attacker built specifically against it, with the attacker's cost in ASR reported.**

**Opening 2: false alarms.** Almost every paper reports detection quality, few report what
happens on benign-but-odd data. Our runs show baselines flagging merely-easy clean samples at
AUC 1.00 while the dynamics detector stays near chance. **Target: matched detection at
substantially lower clean-data loss, especially on imbalanced GTSRB.**

**Opening 3: low poison rates with imbalance.** RPP showed that imbalance amplifies
vulnerability and that defences built for balanced data degrade. GTSRB is imbalanced and
standard. **Target: hold up at 0.5% poison on GTSRB, where precision is base-rate limited.**

### A realistic goal for this project

Not "we beat the state of the art". Rather:

> Match the standard detectors (Spectral Signatures, Activation Clustering, ABL) on
> BackdoorBench's canonical settings, then show that when an attacker is built specifically
> to defeat loss-curve detection, our dynamics-coherence detector still works, or that
> evading it costs the attacker most of its backdoor.

That is achievable in a project of this length, it is a real contribution, and it does not
depend on beating a certified method at its own game. If the result comes out negative, the
report says which attacks evade the method and what the attacker had to sacrifice, which is
also worth publishing.

### Numbers to quote when you calibrate

A modern attack: above 90% ASR at 10% poison or less, clean-accuracy drop under 2%. One 2025
clean-image attack reaches 97.9% ASR on CIFAR-10 at a 0.5% poison rate with under 1%
accuracy drop, and resists most existing defences. If your attack is far from this, it is too
weak to be a fair test.
