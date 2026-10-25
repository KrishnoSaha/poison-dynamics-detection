#!/usr/bin/env bash
# One command to produce every number you need for the first report.
# Runs offline on synthetic data by default. Switch DATASET to mnist or cifar10 when ready.
set -uo pipefail
DATASET="${DATASET:-synth}"; EPOCHS="${EPOCHS:-6}"; RATE="${RATE:-0.05}"
PROBE="${PROBE:-8}"; OUT="${OUT:-runs}"; BUDGET="${BUDGET:-0.10}"
mkdir -p "$OUT"

echo "### step 0: validate the detectors on known ground truth (do this first) ###"
python3 selfcheck.py || { echo "self-check FAILED -- fix before using real data"; exit 1; }

echo; echo "### step 1: train + log dynamics for each attack ###"
for atk in none badnets blended warp frequency clean_label narcissus label_flip sample_specific; do
  echo "--- $atk ---"
  python3 train_log.py --dataset "$DATASET" --attack "$atk" --rate "$RATE" \
    --epochs "$EPOCHS" --probe_every "$PROBE" --out "$OUT/dyn_$atk.npz" | tail -2
done

echo; echo "### step 2: the adaptive attackers (Gate 2 -- the real test) ###"
echo "--- level-matched ---"
python3 train_log.py --dataset "$DATASET" --attack badnets --rate "$RATE" --adaptive \
  --epochs "$EPOCHS" --probe_every "$PROBE" --out "$OUT/dyn_adaptive_level.npz" | tail -2
echo "--- shape-matched (the strong one) ---"
python3 train_log.py --dataset "$DATASET" --attack badnets --rate "$RATE" --adaptive_shape \
  --epochs "$EPOCHS" --probe_every "$PROBE" --out "$OUT/dyn_adaptive_shape.npz" | tail -2

echo; echo "### step 3: score every detector ###"
python3 evaluate.py "$OUT"/dyn_*.npz --budget "$BUDGET" | tee "$OUT/results.txt"
echo; echo "results saved to $OUT/results.txt"
