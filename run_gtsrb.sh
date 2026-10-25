#!/bin/bash
cd "/Users/utshosaha/poison_dynamics_kit 2"
mkdir -p runs
C="--dataset gtsrb --model preactresnet18 --target 1 --epochs 25 --probe_every 20"
[ -f runs/gtsrb_none.npz ] || python3 train_log.py $C --attack none --out runs/gtsrb_none.npz
[ -f runs/gtsrb_badnets.npz ] || python3 train_log.py $C --attack badnets --out runs/gtsrb_badnets.npz
[ -f runs/gtsrb_adaptive_shape.npz ] || python3 train_log.py $C --attack badnets --adaptive_shape --out runs/gtsrb_adaptive_shape.npz
echo ALL DONE
