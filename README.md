# Detecting data-poisoning and backdoor attacks from training dynamics
COMP6015 Group 6. Built on poison_dynamics_kit.

- gtsrb/ : the kit, training and logging code, and the analysis scripts (GTSRB runs, Krishno Saha)
- cifar10/ : CIFAR-10 run scripts, notebook and result summaries (Taseen Hasan)

Analysis scripts are numpy only and read stored run files, for example:
  cd gtsrb
  python3 scalar_conformal.py "runs/gtsrb_badnets*.npz" "runs/gtsrb_none*.npz"
Run files (.npz) are not included because of size.
