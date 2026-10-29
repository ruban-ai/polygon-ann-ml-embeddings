#!/bin/bash
# 2026-10-26: QPS re-measurement (Prasad item 7). Sequential, one config at a time, 64 threads.
cd /raid/ruban/hpmlproj/term_project/SigSpatial
L=logs_20261026/bench_v2_master.log
log(){ echo "[$(date +%T)] $*" | tee -a $L; }
for cd in "ov1m 64" "ov1m 256" "ov10m 64" "ov10m 256" "ov10m 128" "ov10m 512" "ov10m 1024"; do
  set -- $cd; log "START $1 d=$2"
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 /home/buddhi/anaconda3/envs/quant-nmslib2/bin/python bench_are_v2.py $1 $2 > logs_20261026/bench_v2_$1_$2.log 2>&1 && log "DONE $1 d=$2: $(grep -E 'median|long \(' logs_20261026/bench_v2_$1_$2.log | tr '\n' ' ')" || log "FAILED $1 d=$2"
done
log "ALL DONE"
