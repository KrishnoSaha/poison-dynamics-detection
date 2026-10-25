C="--dataset gtsrb --model preactresnet18 --target 1 --epochs 25"
run() {  # batch size, probe_every
  [ -f runs/gtsrb_none_bs$1.npz ] || python3 train_log.py $C --probe_every $2 --attack none --bs $1 --out runs/gtsrb_none_bs$1.npz
  [ -f runs/gtsrb_badnets_bs$1.npz ] || python3 train_log.py $C --probe_every $2 --attack badnets --bs $1 --out runs/gtsrb_badnets_bs$1.npz
}
run 64 40
run 256 10
echo ALL DONE
