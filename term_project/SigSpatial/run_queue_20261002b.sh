#!/bin/bash
# Recovery for train_ov10m_t3m (2026-10-02 18:40): rank 3 of the kNN mining was starved by another
# user's process on GPU 3 (asif run_pilot.py). Shards 0-2,4-7 are on disk. This script recomputes
# shard 3 (rows 1,125,000-1,500,000) on the 7 free GPUs, assembles + spot-checks the kNN cache,
# then trains / evaluates on the 7 free GPUs (same protocol; world size 7 instead of 8).
set -u
cd /raid/ruban/hpmlproj/term_project/SigSpatial
ENV=/raid/ruban/installs/miniconda3/envs/hpmlproj/bin; PY=$ENV/python; TR=$ENV/torchrun
LOGD=logs_20261002; M=$LOGD/queue_master.log
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export QS_LOGD=$PWD/$LOGD QS_OUT=$PWD/RUN_QUEUE_20261002.md QS_TITLE="Probe/quant/baseline queue (started 2026-10-02)"
export QS_DESC="Recovery chain for ov10m_t3m on GPUs 0,1,2,4,5,6,7 (GPU 3 shared with another user's job)"
export QS_STEPS="train_parks_triplet eval_parks_triplet train_parks_infonce eval_parks_infonce train_ov1m_e10 eval_ov1m_e10 knnfill_ov10m_t3m train_ov10m_t3m_w7 eval_ov10m_t3m slices_ov10m_t3m slices_ov10m quant_parksfull quant_wbfull quant_sportsfull quant_ov1m quant_ov10m unsup_nmf icws_parksfull"
FREE=0,1,2,4,5,6,7
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $M; $PY queue_status.py 2>/dev/null; }
run() { local name=$1; shift; log "START $name $*"; "$@" > $LOGD/$name.log 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then log "DONE $name"; else log "FAILED $name exit=$rc (see $LOGD/$name.log)"; fi; return $rc; }
log "RECOVERY START pid=$$"

CUDA_VISIBLE_DEVICES=$FREE run knnfill_ov10m_t3m $TR --standalone --nproc_per_node=7 knn_fill_rows.py ov10m_t3m 1125000 1500000 3 || exit 1

$PY - >> $LOGD/knnfill_ov10m_t3m.log 2>&1 <<'EOF' || { log "FAILED knn_assemble"; exit 1; }
import numpy as np, sys
sys.path.insert(0, '.')
from fulls_common import CONFIGS, raw_matrix, rows
B = '/raid/ruban/hpmlproj/term_project/SigSpatial'
parts = [np.load(f'{B}/knn_ov10m_t3m_shard_{r}.npy') for r in range(8)]
assert all(p.shape == (375000, 30) for p in parts), [p.shape for p in parts]
knn = np.concatenate(parts); assert knn.shape == (3_000_000, 30) and knn.max() < 3_000_000
# spot-check 3 rows of the recomputed shard (and 1 from an original shard) against brute force
X = raw_matrix(CONFIGS['ov10m_t3m'])
def n(a): a = np.maximum(a, 0).astype(np.float64); return a / np.maximum(a.sum(1, keepdims=True), 1e-10)
for q in [1125000, 1300007, 1499999, 700001]:
    xq = n(rows(X, [q]))[0]; best = []
    for c0 in range(0, 3_000_000, 250_000):
        C = n(rows(X, np.arange(c0, c0 + 250_000))); d = np.abs(C - xq).sum(1)
        if c0 <= q < c0 + 250_000: d[q - c0] = np.inf
        idx = np.argpartition(d, 30)[:30]; best += list(zip(d[idx], idx + c0))
    top = set(i for _, i in sorted(best)[:30]); ov = len(top & set(knn[q].tolist()))
    print(f"spot-check row {q}: {ov}/30 match brute force", flush=True); assert ov >= 28
np.save(f'{B}/corpus_knn_ov10m_t3m.npy', knn)
for r in range(8):
    import os; os.remove(f'{B}/knn_ov10m_t3m_shard_{r}.npy')
print(f"assembled corpus_knn_ov10m_t3m.npy {knn.shape}", flush=True)
EOF
log "DONE knn_assemble"

if CUDA_VISIBLE_DEVICES=$FREE run train_ov10m_t3m_w7 $TR --standalone --nproc_per_node=7 run_fulls_listwise_train.py ov10m_t3m; then
  CUDA_VISIBLE_DEVICES=0 run eval_ov10m_t3m $PY eval_fulls_listwise.py ov10m_t3m --threads 64 & E1=$!
  CUDA_VISIBLE_DEVICES=1 run slices_ov10m_t3m $PY eval_ov10m_slices.py --cfg ov10m_t3m --prefixes 64,256 --threads 64 & E2=$!
  wait $E1 $E2
fi
log "RECOVERY END"
