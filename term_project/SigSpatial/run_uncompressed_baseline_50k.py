#!/usr/bin/env python3
"""Uncompressed baseline: reproduces ShapeToVec's OWN method (no learned compression
at all -- full D-dim quadtree simplex vectors, indexed directly under nmslib's custom
WeightedJaccard HNSW space) on our exact 50K benchmark/protocol.

Motivation (author's explicit framing, 2026-09-01): the paper's core contribution is
compression while retaining recall relative to ShapeToVec, not just "our loss beats
other losses." Right now the paper only cites ShapeToVec's own externally-reported 97%
R@50 (their own up-to-1.7M-polygon setup) as Intro motivation -- there is no same-
benchmark, same-protocol comparison anywhere in the paper. This provides one: the
"uncompressed ceiling" on our own 50K corpus/queries/GT, letting the paper state
"a 72x-compressed embedding retains N% of the uncompressed accuracy" as a directly
measured number, not a borrowed one.

CPU-bound (nmslib HNSW build+query), no GPU needed -- runs alongside GPU jobs freely.

Launch: python run_uncompressed_baseline_50k.py"""
import sys, time, csv, datetime, pickle
import numpy as np
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import l1_simplex, nmslib_neighbors, eval_recall

QS = 40000
EF = 200
THREADS = 120
CAND_K = 500
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"


def main():
    t0 = time.time()
    qt = np.load('/raid/ruban/hpmlproj/term_project/SigSpatial/qt_50k.npy')
    gt = pickle.load(open('/raid/ruban/hpmlproj/term_project/SigSpatial/gt_50k.pkl', 'rb'))
    print(f"qt={qt.shape} corpus={QS} queries={qt.shape[0]-QS} dim={qt.shape[1]} "
          f"loaded in {time.time()-t0:.1f}s", flush=True)

    qtn = l1_simplex(qt.copy())  # same simplex normalization ShapeToVec's own vectors carry
    corpus_embs, query_embs = qtn[:QS].astype(np.float32), qtn[QS:].astype(np.float32)
    print(f"corpus vector memory = {corpus_embs.nbytes / 1024**2:.1f} MB "
          f"(vs. our 256-d compressed embedding: {QS*256*4/1024**2:.1f} MB, "
          f"{corpus_embs.nbytes/(QS*256*4):.1f}x larger)", flush=True)

    t1 = time.time()
    nbrs, info = nmslib_neighbors(corpus_embs, query_embs, space="WeightedJaccard",
                                   k=CAND_K, threads=THREADS, query_params={"efSearch": EF})
    print(f"HNSW build+query done in {(time.time()-t1)/60:.1f} min", flush=True)
    metrics = eval_recall(gt, nbrs, QS, CAND_K)
    print(f"[uncompressed-50k, D={qt.shape[1]}] R@10={metrics[10]:.4f} R@50={metrics[50]:.4f} "
          f"R@500={metrics[500]:.4f} QPS={info['qps']:.1f}", flush=True)

    today = datetime.date.today().isoformat()
    note = (f"Uncompressed baseline: full D={qt.shape[1]}-dim simplex vectors (l1_simplex, no "
            f"learned compression) indexed directly under nmslib WeightedJaccard HNSW -- "
            f"reproduces ShapeToVec's own method on our exact 50K corpus/queries/GT, giving a "
            f"same-benchmark uncompressed-accuracy ceiling; efSearch={EF}")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([today, "pk50k", f"50k-uncompressed-d{qt.shape[1]}", qt.shape[1], "base", "",
                    round(metrics[10], 4), round(metrics[50], 4), round(metrics[100], 4),
                    round(metrics[500], 4), round(info['qps'], 1), "run_uncompressed_baseline_50k.py", note])
    print(f"appended 1 row -> {CSV}", flush=True)
    print(f"total wall time = {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
