C="--dataset gtsrb --model preactresnet18 --target 1 --epochs 25 --probe_every 20"
for S in 1 2; do
  [ -f runs/gtsrb_none_s$S.npz ] || python3 train_log.py $C --attack none --seed $S --out runs/gtsrb_none_s$S.npz
  [ -f runs/gtsrb_badnets_s$S.npz ] || python3 train_log.py $C --attack badnets --seed $S --out runs/gtsrb_badnets_s$S.npz
  [ -f runs/gtsrb_adaptive_shape_s$S.npz ] || python3 train_log.py $C --attack badnets --adaptive_shape --seed $S --out runs/gtsrb_adaptive_shape_s$S.npz
done
echo ALL DONE
