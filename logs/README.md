# Experiment logs

SigSpatial / WJ-ANN experiment logs consolidated here.

## Active runs (symlinks → `/tmp`, live-updating)

| File | Description |
|------|-------------|
| `matryoshka_triplet_train80.log` | Matryoshka triplet, 80/20 split (GPUs 0–3) |
| `matryoshka_infonce_train80.log` | Matryoshka InfoNCE, 80/20 split (GPUs 4–7) |

## Full-corpus training (completed)

| File | Description |
|------|-------------|
| `matryoshka_triplet.log` | Matryoshka triplet, all 47K queries |
| `matryoshka_infonce.log` | Matryoshka InfoNCE, all 47K queries |
| `plain_2048_triplet.log` | Plain 2048-d triplet |
| `plain_2048_infonce.log` | Plain 2048-d InfoNCE |

## Eval / setup

| File | Description |
|------|-------------|
| `cross_eval_10k.log` | Full-trained models → 10K benchmark |
| `cross_eval_10k_full.log` | Extended cross-eval |
| `query_split_80_20.log` | 80/20 split + area balance check |
| `matryoshka_monitor.log` | Training monitor |
| `matryoshka_alerts.log` | Stall alerts |

## Other

Lakes calibration/encoding logs from earlier work.
