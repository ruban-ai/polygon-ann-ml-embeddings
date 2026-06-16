#!/usr/bin/env python3
"""Fair re-eval of nb15 (MLP WJ-triplet, log1p) on full, matched settings
(efSearch=200, rerank batch=64, threads=150) so it's directly comparable to
InfoNCE / triplet_AE / random_proj. Appends rows to SigSpatial/NEW_RESULTS.csv."""
import sys, time, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (
    eval_recall, load_dataset_normalized, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu,
)

DEVICE = torch.device("cuda:0")
THREADS, EF, RERANK_BATCH = 150, 200, 64
CKS = [1000, 2000]
CKPT = "/tmp/best_compressor_wj_native_full.pt"
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
METHOD = "MLP-WJ-triplet(nb15,log1p)"


class QuadtreeCompressorWJ(nn.Module):
    def __init__(self, in_dim, out_dim=512, use_log1p=True):
        super().__init__()
        self.use_log1p = use_log1p
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        if self.use_log1p:
            x = torch.log1p(x * 1e6)
        out = F.relu(self.net(x))
        return out / out.sum(dim=1, keepdim=True).clamp(min=1e-10)


@torch.no_grad()
def encode(qt_raw, bs=512):
    m = QuadtreeCompressorWJ(qt_raw.shape[1], 512, use_log1p=True).to(DEVICE)
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True), strict=True)
    m.eval()
    out = []
    for s in range(0, len(qt_raw), bs):
        x = torch.tensor(qt_raw[s:s+bs], dtype=torch.float32, device=DEVICE)
        out.append(m(x).cpu().numpy().astype(np.float32))
    return np.vstack(out)


def main():
    # qt = RAW quadtree vectors (nb15 applies log1p internally, NOT l1_simplex)
    qt, gt, qs, cq, qq, cs, _qt_norm = load_dataset_normalized("full")
    print("encoding nb15 (log1p)...", flush=True)
    embs = encode(qt)
    ce, qe = embs[:qs], embs[qs:]

    max_k = max(max(CKS), 500)
    nbrs, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=max_k,
                                  threads=THREADS, query_params={"efSearch": EF})
    base = eval_recall(gt, nbrs, qs, max_k)
    print(f"[base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@100={base[100]:.4f} "
          f"R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)

    rows = [["base", "", base[10], base[50], base[100], base[500], round(info['qps'])]]
    preload_rerank_corpus(cq, cs)
    for ck in CKS:
        cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck,
                                    threads=THREADS, query_params={"efSearch": EF})
        t0 = time.time()
        rr = rerank_wj_gpu(qq, cand, cq, cs, top_k=ck, batch_size=RERANK_BATCH)
        e2e = len(qq) / max(ci["query_s"] + (time.time()-t0), 1e-9)
        m = eval_recall(gt, rr, qs, ck)
        print(f"[rerank K={ck}] R@10={m[10]:.4f} R@50={m[50]:.4f} R@100={m[100]:.4f} "
              f"R@500={m[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
        rows.append(["rerank", ck, m[10], m[50], m[100], m[500], round(e2e)])
    release_rerank_corpus()

    today = datetime.date.today().isoformat()
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "full", METHOD, 512, stage, ck,
                        round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4),
                        qps, "eval_nb15_fair.py", "matched: efSearch=200, rerank batch=64"])
    print(f"\nappended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == "__main__":
    main()
