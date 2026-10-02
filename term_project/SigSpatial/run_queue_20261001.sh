#!/bin/bash
# Full-scale queue (2026-10-01): Overture 1M, Overture 10M, Sports full -- listwise
# WJ-distillation, Stage-1 recall. Launch: nohup bash run_queue_20261001.sh > logs_20261001/queue_nohup.log 2>&1 &
# Master log: logs_20261001/queue_master.log ; live status: RUN_QUEUE_20261001.md
set -u
cd /raid/ruban/hpmlproj/term_project/SigSpatial
ENV=/raid/ruban/installs/miniconda3/envs/hpmlproj/bin
PY=$ENV/python; TR=$ENV/torchrun
LOGD=logs_20261001; M=$LOGD/queue_master.log
mkdir -p $LOGD
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> $M; $PY queue_status.py 2>/dev/null; }

run() {  # run <name> <cmd...> : foreground, logs to $LOGD/<name>.log, returns exit code
  local name=$1; shift
  log "START $name $*"
  "$@" > $LOGD/$name.log 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then log "DONE $name"; else log "FAILED $name exit=$rc (see $LOGD/$name.log)"; fi
  return $rc
}

# status refresher every 10 min while the queue lives
( while kill -0 $$ 2>/dev/null; do $PY queue_status.py 2>/dev/null; sleep 600; done ) &

log "QUEUE START pid=$$"

# --- CPU: Sports build in parallel with the Overture GPU work ------------------
FREE_GB=$(df -BG --output=avail /raid | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 200 ]; then
  log "SKIPPED build_sports only ${FREE_GB}GB free on /raid (<200GB needed)"; SB_PID=""
else
  run build_sports $PY build_sportsfull.py & SB_PID=$!
fi

# --- GPU chain ------------------------------------------------------------------
EVAL_PIDS=""
if run train_ov1m $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py ov1m; then
  run eval_ov1m $PY eval_fulls_listwise.py ov1m --threads 64 & EVAL_PIDS="$EVAL_PIDS $!"
else log "SKIPPED eval_ov1m (training failed)"; fi

if run train_ov10m $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py ov10m; then
  run eval_ov10m $PY eval_fulls_listwise.py ov10m --threads 96 & EVAL_PIDS="$EVAL_PIDS $!"
else log "SKIPPED eval_ov10m (training failed)"; fi

if [ -n "$SB_PID" ]; then wait $SB_PID; SB_RC=$?; else SB_RC=1; fi
if [ $SB_RC -eq 0 ] && [ -f gt_sportsfull.pkl ]; then
  if run train_sportsfull $TR --standalone --nproc_per_node=8 run_fulls_listwise_train.py sportsfull; then
    run eval_sportsfull $PY eval_fulls_listwise.py sportsfull --threads 96
  else log "SKIPPED eval_sportsfull (training failed)"; fi
else
  log "SKIPPED train_sportsfull (sports build failed or missing GT)"
fi

for p in $EVAL_PIDS; do wait $p; done
log "QUEUE END"
