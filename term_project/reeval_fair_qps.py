#!/usr/bin/env python3
"""Fair apples-to-apples re-eval (full set): InfoNCE vs triplet_AE vs random_proj.
Identical candidate-gen + rerank settings for all (rerank batch=64, efSearch=200,
threads=150) so the end-to-end QPS comparison is airtight. Base HNSW QPS is already
implementation-clean; this also pins the rerank cost identically across methods."""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (
    eval_recall, load_dataset_normalized, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, shifted_l1_simplex,
)

DEVICE = torch.device("cuda:0")
THREADS = 150
EF = 200
RERANK_BATCH = 64
CKS = [1000, 2000]
OUT_DIM = 512
INFONCE_CKPT = "/tmp/best_filter_recall_infonce_512_full.pt"
TRIPLET_CKPT = "/tmp/best_sota_triplet_autoencoder_wj_512_full.pt"


class TripletAE(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )

    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z / z.sum(dim=1, keepdim=True).clamp(min=1e-10)


@torch.no_grad()
def encode_ckpt(ckpt, qt_norm, bs=512):
    m = TripletAE(qt_norm.shape[1], OUT_DIM).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    enc = {k: v for k, v in state.items() if k.startswith("encoder.")}
    m.load_state_dict(enc, strict=False)
    m.eval()
    out = []
    for s in range(0, len(qt_norm), bs):
        x = torch.tensor(qt_norm[s:s+bs], dtype=torch.float32, device=DEVICE)
        out.append(m.encode(x).cpu().numpy().astype(np.float32))
    return np.vstack(out)


def random_proj_embs(qt, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((qt.shape[1], OUT_DIM)).astype(np.float32) / np.sqrt(OUT_DIM)
    return shifted_l1_simplex(qt.astype(np.float32) @ W)


def evaluate(name, embs, gt, query_start, corpus_qt, query_qt, corpus_sums, rows):
    ce, qe = embs[:query_start], embs[query_start:]
    max_k = max(max(CKS), 500)
    nbrs, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=max_k,
                                  threads=THREADS, query_params={"efSearch": EF})
    base = eval_recall(gt, nbrs, query_start, max_k)
    preload_rerank_corpus(corpus_qt, corpus_sums)
    rr = {}
    for ck in CKS:
        cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck,
                                    threads=THREADS, query_params={"efSearch": EF})
        t0 = time.time()
        out = rerank_wj_gpu(query_qt, cand, corpus_qt, corpus_sums, top_k=ck, batch_size=RERANK_BATCH)
        rr_s = time.time() - t0
        m = eval_recall(gt, out, query_start, ck)
        e2e = len(query_qt) / max(ci["query_s"] + rr_s, 1e-9)
        rr[ck] = (m[50], m[500], ci["qps"], e2e)
    release_rerank_corpus()
    rows.append((name, base[50], base[500], info["qps"], rr))
    print(f"[{name}] base R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)


def main():
    qt, gt, qs, cq, qq, cs, qt_norm = load_dataset_normalized("full")
    rows = []
    print("building embeddings...", flush=True)
    evaluate("InfoNCE",     encode_ckpt(INFONCE_CKPT, qt_norm), gt, qs, cq, qq, cs, rows)
    evaluate("triplet_AE",  encode_ckpt(TRIPLET_CKPT, qt_norm), gt, qs, cq, qq, cs, rows)
    evaluate("random_proj", random_proj_embs(qt),               gt, qs, cq, qq, cs, rows)

    print("\n" + "=" * 92)
    print("FAIR RE-EVAL (full) — identical rerank batch=64, efSearch=200, threads=150")
    print("=" * 92)
    print(f"{'method':<13}{'baseR@50':>9}{'baseR@500':>10}{'HNSW_QPS':>10}"
          f"{'  | K':>6}{'rrR@50':>9}{'rrR@500':>9}{'candQPS':>9}{'e2eQPS':>9}")
    print("-" * 92)
    for name, b50, b500, bq, rr in rows:
        for i, ck in enumerate(CKS):
            r50, r500, cq_, e2e = rr[ck]
            head = f"{name:<13}{b50:>9.4f}{b500:>10.4f}{bq:>10.0f}" if i == 0 else " " * 42
            print(f"{head}{ck:>6}{r50:>9.4f}{r500:>9.4f}{cq_:>9.0f}{e2e:>9.0f}")
    import pickle
    pickle.dump(rows, open("/raid/ruban/hpmlproj/term_project/results_saved/fair_reeval_full.pkl", "wb"))
    print("\nsaved → results_saved/fair_reeval_full.pkl")


if __name__ == "__main__":
    main()
