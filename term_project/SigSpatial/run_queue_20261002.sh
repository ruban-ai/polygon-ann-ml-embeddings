#!/bin/bash
# 2026-10-02 queue: (A) Overture probes, (B) quantized learned signatures, (C) Parks full baselines.
# Launch: setsid nohup bash run_queue_20261002.sh > logs_20261002/queue_nohup.log 2>&1 < /dev/null &
# Master log: logs_20261002/queue_master.log ; live status: RUN_QUEUE_20261002.md
set -u
cd /raid/ruban/hpmlproj/term_project/SigSpatial
ENV=/raid/ruban/installs/miniconda3/envs/hpmlproj/bin
PY=$ENV/python; TR=$ENV/torchrun
LOGD=logs_20261002; M=$LOGD/queue_master.log
mkdir -p $LOGD
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export QS_LOGD=$PWD/$LOGD QS_OUT=$PWD/RUN_QUEUE_20261002.md QS_TITLE="Probe/quant/baseline queue (started 2026-10-02)"
export QS_DESC="GPU chain: parks triplet -> parks infonce -> ov1m 10ep -> ov10m trained on 3M (evals in background) | CPU lane A: ov10m index-size slices -> int8/int4 quant evals (parks, wb, sports, ov1m, ov10m) -> parks NMF | CPU lane B: parks ICWS brute force"
export QS_STEPS="train_parks_triplet eval_parks_triplet train_parks_infonce eval_parks_infonce train_ov1m_e10 eval_ov1m_e10 train_ov10m_t3m eval_ov10m_t3m slices_ov10m_t3m slices_ov10m quant_parksfull quant_wbfull quant_sportsfull quant_ov1m quant_ov10m unsup_nmf icws_parksfull"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $M; $PY queue_status.py 2>/dev/null; }
run() {
  local name=$1; shift
  log "START $name $*"
  "$@" > $LOGD/$name.log 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then log "DONE $name"; else log "FAILED $name exit=$rc (see $LOGD/$name.log)"; fi
  return $rc
}
( while kill -0 $$ 2>/dev/null; do $PY queue_status.py 2>/dev/null; sleep 600; done ) &
log "QUEUE START pid=$$"

# --- CPU lane B: ICWS (GPU0 for signing only, then CPU ranking) -----------------
( CUDA_VISIBLE_DEVICES=0 run icws_parksfull $PY run_icws_brute_full.py ) & LB=$!

# --- CPU lane A: slices -> quant -> NMF ------------------------------------------
(
  CUDA_VISIBLE_DEVICES=0 run slices_ov10m $PY eval_ov10m_slices.py --cfg ov10m --prefixes 64,256 --threads 64
  CUDA_VISIBLE_DEVICES=0 run quant_parksfull $PY eval_quant.py parksfull --prefixes 256 --bits 8,4 --threads 64
  CUDA_VISIBLE_DEVICES=0 run quant_wbfull $PY eval_quant.py wbfull --prefixes 256 --bits 8,4 --threads 64
  CUDA_VISIBLE_DEVICES=0 run quant_sportsfull $PY eval_quant.py sportsfull --prefixes 256 --bits 8,4 --threads 64
  CUDA_VISIBLE_DEVICES=0 run quant_ov1m $PY eval_quant.py ov1m --prefixes 64,256 --bits 8,4 --threads 64
  CUDA_VISIBLE_DEVICES=0 run quant_ov10m $PY eval_quant.py ov10m --prefixes 64,256 --bits 8,4 --no-ref --threads 64
  CUDA_VISIBLE_DEVICES=0 run unsup_nmf $PY run_parksfull_unsup_baselines.py nmf --threads 64
) & LA=$!

# --- GPU chain ---------------------------------------------------------------------
EV=""
if run train_parks_triplet $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py parksfull --method triplet; then
  run eval_parks_triplet $PY eval_fulls_listwise.py parksfull --method triplet --threads 48 & EV="$EV $!"
else log "SKIPPED eval_parks_triplet (training failed)"; fi
if run train_parks_infonce $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py parksfull --method infonce; then
  run eval_parks_infonce $PY eval_fulls_listwise.py parksfull --method infonce --threads 48 & EV="$EV $!"
else log "SKIPPED eval_parks_infonce (training failed)"; fi
if run train_ov1m_e10 $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py ov1m_e10; then
  run eval_ov1m_e10 $PY eval_fulls_listwise.py ov1m_e10 --threads 48 & EV="$EV $!"
else log "SKIPPED eval_ov1m_e10 (training failed)"; fi
if run train_ov10m_t3m $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py ov10m_t3m; then
  run eval_ov10m_t3m $PY eval_fulls_listwise.py ov10m_t3m --threads 64 & EV="$EV $!"
  run slices_ov10m_t3m $PY eval_ov10m_slices.py --cfg ov10m_t3m --prefixes 64,256 --threads 48 & EV="$EV $!"
else log "SKIPPED eval_ov10m_t3m slices_ov10m_t3m (training failed)"; fi

for p in $EV $LA $LB; do wait $p; done
log "QUEUE END"
