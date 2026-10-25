Detecting Data-Poisoning and Backdoor Attacks from Training-Dynamics Signatures

COMP6015 Computer Science Project 1, Curtin University (Master of Computing, Cyber Security). Group 6. Supervisor: Victor Calo. Built on the starter kit poison_dynamics_kit.

What this project does

A backdoor attack corrupts a small part of the training data so that a model behaves normally except when a trigger is present. We record how every training sample is learned over time (its loss and confidence at 244-261 checkpoints) and ask whether poisoned samples can be flagged from that trajectory at a fixed, pre-set false-alarm rate.

Main pieces:

Calibrated detection. A per-class conformal threshold set from a small clean calibration set, so the false-positive rate can be checked on clean runs.
Two kinds of detector. An SVD-residual trusted-reference detector (direction of the signal is lost) and single-feature detectors that keep the direction (upper or lower tail).
Controls. Same-label control, per-class false-positive check, batch sizes 64/128/256, multiple seeds, and a selection-matched null for the starter kit's confirmation test.
A signed direction analysis. For eight trajectory features, whether poison sits higher or lower than clean samples of the same label.
Headline results (conditional, not a general detector)
The conformal thresholds hold their target false-positive rate on clean runs (about 4-6% at a 5% target).
SVD-residual detector: catches about 10-30% of poison on GTSRB and 1-5% on CIFAR-10 strong models, where its score is inverted (AUC 0.28-0.34).
The direction of the signal differs by dataset: poison has higher loss and lower confidence than clean samples of its class on GTSRB, and lower loss and higher confidence on CIFAR-10 strong models.
With the direction known, one feature recovers much more (true-positive rate at 5% false-positive rate): early loss on GTSRB 0.30-1.00, final confidence on CIFAR-10 0.00-0.83. No single rule works on both datasets, and results vary between seeds.
The starter kit's confirmation test wrongly confirmed 9% of clean-run draws; our selection-matched version 3%.

Limits. The feature and direction of the single-feature detectors were chosen from labelled data, so this is not label-free and needs a clean calibration set. Two seeds on CIFAR-10 and three on GTSRB. See the final report for details and caveats.

Repository layout
gtsrb/     Starter kit, training/logging code and all analysis scripts (GTSRB runs)
cifar10/   CIFAR-10 run scripts, notebook and result summaries
README.md  This file

Run files (.npz, 70-200 MB each), model weights and datasets are not in the repository because of size. They can be regenerated with the training commands below.

Settings used
Setting	CIFAR-10	GTSRB
Training samples / poisoned	50,000 / 2,500 (5%)	26,640 / 1,332 (5%)
Classes / target class	10 / 0	43 / 1
Models	PreActResNet18, VGG19-BN	PreActResNet18
Epochs / checkpoints	25 / 244	25 / 261
Attacks	BadNets, adaptive_shape	BadNets, adaptive_shape
Seeds	2	3 (batch sizes 64 and 256 for BadNets, one seed)
Requirements
Python 3, PyTorch (Apple MPS or CUDA for training), NumPy.
The analysis scripts use NumPy only, read stored run files, and run in seconds without a GPU.
How to run

Training and logging (one run file per condition). See gtsrb/run_gtsrb.sh, gtsrb/run_gtsrb_seeds.sh and gtsrb/run_bs.sh for the exact commands. The general form is:

python3 train_log.py --dataset gtsrb --attack badnets --rate 0.05 --model preactresnet18 --epochs 25
python3 train_log.py --dataset gtsrb --attack badnets --rate 0.05 --model preactresnet18 --epochs 25 --adaptive_shape
python3 train_log.py --dataset gtsrb --attack none --rate 0.05 --model preactresnet18 --epochs 25

Analysis (from the folder that contains runs/):

python3 strong_model_checks_v2.py "runs/gtsrb_*.npz"          # trusted-reference detector, conformal TPR/FPR
python3 per_class_check.py "runs/gtsrb_*.npz"                 # per-class false-positive rate vs class size
python3 two_sided_check.py "runs/gtsrb_*.npz"                 # upper, lower and two-sided tails
python3 circularity_check.py --null "runs/gtsrb_none*.npz" --test "runs/gtsrb_badnets*.npz" "runs/gtsrb_adaptive_shape*.npz" --draws 20
python3 mechanism_dir.py "runs/gtsrb_badnets*.npz"            # signed AUC of eight features, same-label control
python3 scalar_conformal.py "runs/gtsrb_badnets*.npz" "runs/gtsrb_none*.npz"   # single-feature conformal detector
Scripts
Script	Purpose
train_log.py	Trains a model on poisoned data and logs per-sample loss and confidence (--model, GTSRB loader, batched evaluation added by us)
strong_model_checks_v2.py	Trusted-reference SVD-residual detector with conformal thresholds
per_class_check.py	Per-class false-positive rate against class size
two_sided_check.py	Upper, lower and two-sided conformal tails
circularity_check.py	Starter kit's confirmation test against a selection-matched null
mechanism_dir.py	Signed same-label AUC of trajectory features
scalar_conformal.py	Single-feature conformal detector with a chosen tail
Changes to the starter kit

train_log.py: --model flag (PreActResNet18, VGG19-BN), GTSRB loader, batched final evaluation (fixes a memory crash on the Mac GPU). detect.py: a speed-up with the same output. evaluate.py: a cosmetic print change. The attacks in poison.py are unchanged.

Team
Name	Student ID
Krishno Saha	23516506
Taseen Hasan	23353981
Shajibul Islam Nayem	23396041
Md Fardin Ahmed Sabbir	23293731
Nibraz Khan	23410051

GTSRB runs and analysis scripts: Krishno Saha. CIFAR-10 runs: Taseen Hasan. Other contributions: see the Individual Contributions section of the final report.

Use of AI tools
We used an AI assistant to learn definitions and cybersecurity terms, to debug and fix errors.

Reference

Starter kit: poison_dynamics_kit (course-provided). Datasets: CIFAR-10 (Krizhevsky 2009), GTSRB (Stallkamp et al. 2012).
