# Detecting Data-Poisoning and Backdoor Attacks from Training-Dynamics Signatures

A starter kit. Everything here runs, and every claim in it has been measured, not assumed.

You need Python, numpy, and PyTorch. Strong programming and a security background are the
right preparation. You do not need advanced maths: the whole method is arrays and
correlations, explained in `THEORY.md`.

---

## Start here (5 minutes, no GPU, no downloads)

```bash
pip install numpy torch
python3 selfcheck.py          # validates the detectors on data with known answers
bash run_pipeline.sh          # full pipeline on offline synthetic data
```

`selfcheck.py` must print "All checks passed" before you trust anything else. It plants
poison in fake training dynamics and checks that each detector finds it, and equally
importantly that a benign-but-easy group is *not* flagged.

When that works, switch to real data:

```bash
DATASET=mnist   EPOCHS=8  bash run_pipeline.sh
DATASET=cifar10 EPOCHS=20 bash run_pipeline.sh
```

---

## The one-sentence idea

Poisoned samples all carry the same trigger, so during training they are pushed around by
one shared cause and therefore **move together**. Existing detectors look at each sample
alone and ask "was this learned suspiciously fast?", which an attacker can hide from. We ask
"do these samples move together more than equally-easy clean samples do?", which is harder to
hide because the shared trigger is the attack.

---

## Files

| File | What it does |
|---|---|
| `THEORY.md` | the idea, no maths. **Read this first.** |
| `PROJECT_GUIDE.md` | **datasets, the test matrix, the reading list, and the bar to beat** |
| `ATTACKS_AND_DEFENCES.md` | the attack menu, the detection landscape, and the measured no-free-lunch table |
| `selfcheck.py` | validates every detector against known ground truth. Run before anything. |
| `poison.py` | eight attacks: badnets, blended, warp, frequency, clean-label, narcissus, label-flip (control), sample-specific (attacks our own assumption) |
| `train_log.py` | trains a CNN and records per-sample loss and confidence over time |
| `detect.py` | three published-style baselines, plus our detector and its control |
| `evaluate.py` | scores detectors with security metrics, not accuracy |
| `run_pipeline.sh` | one command: self-check, train all attacks, evaluate |

---

## What we already measured, so you start ahead

These came from the runs in this kit. They shape the project.

**The simple baselines are strong.** On BadNets, a plain "area under the loss curve"
detector reaches AUC 0.99 and catches 97% of poison at a 10% removal budget. On blended
triggers, a curve-shape detector reaches AUC 1.00. Reproduce these first. **Do not expect to
beat them on textbook attacks, and do not build a project around trying.**

**They break against an attacker who is paying attention.** We include one: it slows the
poisoned samples so their loss level matches the clean data. The curve-area detector falls
from AUC 0.99 to 0.71, and its recall from 97% to **0%**, while the backdoor still works.
Our detector goes the other way, from 0.858 to 0.975.

**They also cry wolf.** On a group of merely easy clean samples, two baselines fire at AUC
1.00 while ours stays near chance at 0.56. For a defender, false alarms are the cost that
actually matters.

**But one baseline survives our attacker.** The curve-*shape* detector still scores 1.00
against it, because our attacker matched the loss level and not its shape.

**We built that attacker for you** (`--adaptive_shape`), and it is nearly free for the
attacker: it keeps 97% attack success while still breaking the loss-area detector, where
level-matching alone cost half the backdoor. It also breaks our coherence *control* (z = -5.1)
while leaving our per-sample detector working (0.98) -- because the coherence statistic
saturates. Fixing that saturation, then rerunning, is the first new result available to you.
See `ATTACKS_AND_DEFENCES.md` for the full table and what it implies.

---

## Go / no-go gates

Do these in order and write down the numbers each time. A gate that fails is a result, not a
setback, provided you can show the test would have detected a real effect.

**Gate 0. Can you see planted poison at all?** `selfcheck.py` passes. Hours.

**Gate 1. Can you match the published baselines on standard attacks?** Run BadNets and
blended on MNIST or CIFAR-10 and get the baselines working at high AUC. If you cannot
reproduce a strong baseline, fix that before proposing anything new. Days.

**Gate 2. Does coherence survive an adaptive attacker? THIS IS THE PROJECT.** Improve the
included attacker so it matches both the level and the shape of clean loss curves. Then
compare all detectors. Two outcomes, both publishable:
- coherence still detects it -> a defence that is robust to the evasion that beats
  curve-based detectors;
- coherence also fails -> a clean negative, with an explanation of what the attacker had to
  give up (measure the attack success rate: if evasion costs the backdoor, that is also a
  defensive win).

**Gate 3. Does it hold up at realistic poison rates and on clean-label attacks?** Sweep the
rate from 0.5% to 10%. Clean-label attacks are the hard case because nothing is mislabelled.

---

## Report these numbers, not accuracy

Poison is rare, so accuracy and AUC alone will mislead you and any security reader will say
so. For a fixed removal budget (say you discard the 10% most suspicious samples), report:

- **recall** -- what fraction of the poison you caught;
- **precision** -- what fraction of what you removed was actually poison;
- **clean data lost** -- the false-alarm cost;
- **attack success rate after removal and retraining** -- did the defence actually work?

`evaluate.py` prints the first three. Gate 2 needs the fourth: retrain on the cleaned set and
measure whether the backdoor still fires.

Also always report the **control**: the `z` score from `matched_benign_null`, which says
whether your flagged group moves together more than *equally easy* clean data. Without it you
may only be detecting that poison is easy.

---

## Three ways to get a wrong answer

1. **Comparing poison to all clean data.** Proves poison is easy. Everyone knows. Compare
   against clean data of matched difficulty.
2. **Trusting a confirmation number when precision is low.** If precision is 5%, your
   "confirmed" group is 95% clean data. `evaluate.py` warns you.
3. **Claiming there is no signal when your test was blind.** Plant a synthetic signal of the
   size you care about and show your detector finds it, before writing "not detected".

---

## Suggested first month

Week 1: read `THEORY.md` then `PROJECT_GUIDE.md`. Run `selfcheck.py` and `run_pipeline.sh`.
Read BadNets, Spectral Signatures, Anti-Backdoor Learning and the BackdoorBench paper. Write
one page on how each baseline works.

Week 2: CIFAR-10 with PreActResNet18 at 5% poison, target label 0, which is the canonical
setting. Reproduce the standard detectors at strength. Always report clean accuracy and attack
success rate first. This is Gate 1.

Week 3: sweep the poison rate over 0.1, 0.5, 1, 5 and 10 percent, and build the
attack-by-detector matrix. Find where each detector breaks. Add GTSRB, where class imbalance
makes the false-alarm story interesting.

Week 4: start Gate 2. Build an attacker aimed at whichever detector is working best, and write
down in advance what would count as success and what as failure.

After that, move the attacks to BackdoorBench implementations so your numbers are comparable,
and position explicitly against CSC and RPP.

---

## Scope and ethics

The attacks here exist so that defences can be measured, which is standard practice in this
field. They are all published. Use them only on data and models you own, and keep the work
inside the project. The deliverable is a detector and an honest evaluation of it, not an
attack.
