# The idea, without the maths

You do not need geometry or statistics theory for this project. Everything below is arrays
of numbers and correlations between them. Read this once, then read `detect.py`, which is
about 100 lines and follows this document exactly.

---

## 1. What we record

Train a model normally. Every so often (say every 20 steps), pause and run one forward pass
over the whole training set, writing down two numbers for every training sample:

- **loss** -- how wrong the model currently is about that sample.
- **confidence** -- how sure the model is about that sample's label.

After training you have two matrices, both `(n_samples, n_checkpoints)`. That is all the
input any detector here uses. No gradients per sample, no retraining, no model internals.
One extra forward pass per checkpoint.

Think of each sample as having a short time series: "here is how this sample was learned".

---

## 2. What everyone already does, and why it works

Poisoned samples are usually **easy**. A trigger patch is a big obvious feature, so the
model latches onto it fast. That gives the standard family of detectors:

- **Learned early?** Look at the loss at an early checkpoint. Low means suspicious.
- **Learned fast overall?** Look at the area under the loss curve. Small means suspicious.
- **Odd curve shape?** Compare the shape of each sample's loss curve to the population.

These are strong. Published methods along these lines work well, and in our own tests they
reach essentially perfect detection on textbook attacks. **Reproduce them first, and expect
them to be hard to beat.** A project that only shows "we also detect BadNets" adds nothing.

---

## 3. Where they break

All of the above read **one sample at a time**. That is the weakness, and an attacker can
exploit it.

If the attacker knows you are looking at loss curves, they can slow the poisoned samples
down so their curves look ordinary. This costs the attacker a little (the backdoor takes
longer to install) but it works: in our tests, an attacker that matched the poison's loss
level dropped a curve-area detector from 0.99 to 0.71 AUC, and its recall from 97% to 0%.
The backdoor was still fully functional.

There is a second, quieter problem. Some benign samples are genuinely easy: clean, typical,
unambiguous. Curve-based detectors flag those too. In our tests a group of merely easy
benign samples was flagged with AUC 1.00 by two of the three baselines, which in a real
deployment means throwing away good data and burning analyst time.

---

## 4. The idea: poison is not just easy, it is easy *for the same reason*

Here is the whole thesis, and it comes from how backdoors work.

Every poisoned sample carries **the same trigger**. That is not incidental; it is what makes
the attack function at test time. So during training, all poisoned samples are being pushed
around by **one shared cause**. When the model starts picking up the trigger, every poisoned
sample improves, together, at the same moment.

Benign easy samples are easy for many unrelated reasons: this one is a clear photo, that one
is a common shape, another has a distinctive colour. They are easy **separately**.

So the question to ask is not "is this sample unusual?" but:

> **Do these samples move together more than ordinary samples of the same difficulty do?**

We call that **coherence**. It is a property of a *group*, never of a single sample. And it
is much harder for an attacker to fake, because removing the shared behaviour means removing
the shared trigger, which removes the attack.

---

## 5. How we measure it, in code

**Per-sample score (`spectral_dynamics`).** Four lines of numpy:

1. Take step-to-step changes: `dL = diff(loss)`, `dC = diff(conf)`. Stack them side by side.
2. Subtract the column average. This removes "what the whole training set was doing", so
   what remains for each sample is *its own* deviation from the crowd.
3. Take the SVD. This finds the few directions that the most samples share.
4. Score each sample by how strongly it lines up with those directions.

Poisoned samples share a hidden cause, so their leftovers all point the same way and score
high. This is deliberately the same trick as Spectral Signatures (Tran et al. 2018), which
does it to the network's *features*. We do it to the *training dynamics*, which an attacker
must disturb in order to install the backdoor at all.

**Group check (`group_coherence`).** Once you have a candidate group, compute the average
pairwise correlation between members' time series. High means they move together.

**The control (`matched_benign_null`) -- the most important function in the kit.** Poisoned
samples are easy, and easy samples move together a bit anyway. So take your flagged group,
build fake groups of *benign* samples with the *same difficulty profile*, and measure their
coherence. Only the excess counts. The output is a `z` score: how many standard deviations
above equally-easy benign data your group sits.

Without this control you will "detect" difficulty and call it poison. With it, you have a
real claim.

---

## 6. Three traps that will invalidate your results

**Trap 1: comparing against the wrong benign group.** Comparing poison to *all* benign data
proves only that poison is easy, which everyone knows. Compare against benign data of
matched difficulty. `matched_benign_null` does this.

**Trap 2: reading a confirmation number on a wrongly-flagged set.** If your detector's
precision is 5%, then your "confirmed coherent group" is 95% clean data and the number
describes nothing. Always read precision first. `evaluate.py` warns you when precision is
below 0.25.

**Trap 3: claiming absence when you were simply blind.** If your detector scores near chance,
that may mean there is nothing there, or it may mean your test cannot see it at that
signal strength. Before writing "no signal", plant a synthetic signal of the size you claim
to care about and check that your detector finds it. `selfcheck.py` is the template.

---

## 7. What we already know, so you do not have to rediscover it

From runs included in this kit:

- On textbook attacks (BadNets, blended) the simple curve baselines are excellent, up to
  AUC 1.00. Ours is competitive but does not beat them. **This is expected.**
- Against an attacker that matches the poison's loss *level*, the curve-area detector
  collapses (0.99 to 0.71 AUC; recall 97% to 0%) while ours rises to 0.975.
- But a *shape*-based baseline survives that attacker too. So level-matching is not enough
  to defeat everything.
- On a benign-but-easy decoy group, curve baselines fire hard (AUC 1.00) and ours does not
  (0.56). Fewer false alarms is a real advantage for a defender.
- Label-flipping (wrong labels, no trigger) behaves in the *opposite* direction: those
  samples are hard, not easy, so a detector tuned for "suspiciously easy" ranks them last.
  One detector cannot cover both without looking at curve shape.

**The open question, and your project:** build an attacker that matches both the level *and*
the shape of the benign loss curves, then find out whether coherence still detects it. If it
does, that is a genuinely new and useful defence, because the attacker has to break the
shared trigger to escape it. If it does not, you have a clean negative result and a clear
explanation of why, which is also a real contribution.

---

## 8. Reading list, in order

Start with the two that define the baselines you must beat:

1. Tran, Li, Madry, *Spectral Signatures in Backdoor Attacks* (2018) -- the shared-direction
   idea, applied to features. Our detector is its dynamics analogue.
2. Li et al., *Anti-Backdoor Learning* (NeurIPS 2021, arXiv:2110.11571) -- backdoored samples
   are learned early; isolate them.

Then the modern dynamics-based detectors and the benchmark:

3. *BaDLoss* (arXiv:2408.13221) -- per-sample loss trajectories with probe samples.
4. *BackdoorBench* (arXiv:2407.19845) -- the standard benchmark and attack implementations.
   Move to these implementations once your pipeline works; do not hand-roll attacks for a
   final evaluation.

Check the citations yourself before using them.
