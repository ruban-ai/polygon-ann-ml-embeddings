#!/usr/bin/env python3
"""Cross-eval: model trained on full (187K+47K) -> Table-1 10K benchmark.

Uses first 10K rows of full 18220-d vectors (same polygons as /tmp/qt_10k.npy)
with /tmp/gt_lookup_10k.pkl. 10k encodings are 18499-d; models expect 18220-d."""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/raid/ruban/hpmlproj/term_project/SigSpatial")
from sota_experiment_common import (
    QUERY_START_10K,
    eval_recall,
    l1_simplex,
    load_dataset,
    nmslib_neighbors,
    preload_rerank_corpus,
    release_rerank_corpus,
    rerank_wj_gpu,
)

DEV = torch.device("cuda:0")
THREADS = 120
EF = 200
RB = 64
CAND_KS = [500, 1000]


def norm(x):
    return x / x.sum(1, keepdim=True).clamp(min=1e-10)


class PlainAE(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )

    def embed(self, x):
        return F.relu(self.encoder(x))


class MatAE(nn.Module):
    def __init__(self, d, emb=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, emb, bias=False), nn.BatchNorm1d(emb),
        )

    def embed(self, x):
        return F.relu(self.encoder(x))


class TripletAE(nn.Module):
    def __init__(self, d, out_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )

    def embed(self, x):
        return F.relu(self.encoder(x))


def load_qt_norm_10k_from_full():
    qt_full = np.load("/tmp/qtree_vectors_full.npy", mmap_mode="r")[:10000]
    qt = np.array(qt_full, dtype=np.float32)
    return l1_simplex(qt)


@torch.no_grad()
def embed(model, qtn, out_dim=None, prefix=None, batch=512):
    model.eval()
    chunks = []
    for i in range(0, len(qtn), batch):
        x = torch.tensor(qtn[i:i + batch], dtype=torch.float32, device=DEV)
        z = model.embed(x)
        if prefix is not None:
            z = z[:, :prefix]
        z = norm(z)
        chunks.append(z.cpu().numpy().astype(np.float32))
    emb = np.vstack(chunks)
    if out_dim is not None:
        emb = emb[:, :out_dim]
        emb = emb / np.maximum(emb.sum(1, keepdims=True), 1e-10)
    return emb


def build_model(arch, in_dim, out_dim):
    if arch == "plain":
        return PlainAE(in_dim, out_dim)
    if arch == "matryoshka":
        return MatAE(in_dim)
    if arch == "funnel":
        return TripletAE(in_dim, out_dim)
    raise ValueError(arch)


def eval_ckpt(name, ckpt, arch, out_dim, prefix=None, base_only=False):
    _, gt, qs, cq, qq, cs = load_dataset("10k")
    qtn = load_qt_norm_10k_from_full()
    assert qtn.shape[0] == 10000 and qs == QUERY_START_10K == 8000

    model = build_model(arch, qtn.shape[1], out_dim).to(DEV)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)

    emb = embed(model, qtn, out_dim=out_dim if arch == "plain" else None, prefix=prefix)
    ce, qe = emb[:qs], emb[qs:]
    max_k = max(max(CAND_KS), 500)

    print(f"\n=== {name} | ckpt={ckpt} | emb={emb.shape[1]} ===", flush=True)
    nbrs, info = nmslib_neighbors(
        ce, qe, space="WeightedJaccard", k=max_k, threads=THREADS, query_params={"efSearch": EF}
    )
    base = eval_recall(gt, nbrs, qs, max_k)
    print(
        f"  base  R@10={base[10]:.4f} R@50={base[50]:.4f} R@100={base[100]:.4f} "
        f"R@500={base[500]:.4f} QPS={info['qps']:.0f}",
        flush=True,
    )

    if not base_only:
        preload_rerank_corpus(cq, cs)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(
                ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF}
            )
            t1 = time.time()
            rr = rerank_wj_gpu(qq, cand, cq, cs, top_k=ck, batch_size=RB)
            e2e = len(qq) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(
                f"  rr@{ck} R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@100={mm[100]:.4f} "
                f"R@500={mm[500]:.4f} e2eQPS={e2e:.0f}",
                flush=True,
            )
        release_rerank_corpus()
    return {"base": base, "qps": info["qps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["all", "mat_tri", "mat_inf", "plain_tri", "plain_inf", "funnel512"], default="all")
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()

    jobs = []
    if args.which in ("all", "mat_tri"):
        for d in (1024, 2048, 4096):
            jobs.append((f"Matryoshka-triplet@{d}", "/tmp/best_matryoshka_triplet_full.pt", "matryoshka", d, d))
    if args.which in ("all", "mat_inf"):
        for d in (1024, 2048, 4096):
            jobs.append((f"Matryoshka-InfoNCE@{d}", "/tmp/best_matryoshka_infonce_full.pt", "matryoshka", d, d))
    if args.which in ("all", "plain_tri"):
        jobs.append(("Plain-triplet@2048", "/tmp/best_plain_triplet_2048_full.pt", "plain", 2048, None))
    if args.which in ("all", "plain_inf"):
        jobs.append(("Plain-InfoNCE@2048", "/tmp/best_plain_infonce_2048_full.pt", "plain", 2048, None))
    if args.which in ("all", "funnel512"):
        jobs.append(("AE WJ-tri+recon funnel@512", "/tmp/best_sota_triplet_autoencoder_wj_512_full.pt", "funnel", 512, None))
        jobs.append(("InfoNCE--WJ funnel@512", "/tmp/best_filter_recall_infonce_512_full.pt", "funnel", 512, None))

    print("10K cross-eval: full-trained ckpts -> 8K corpus / 2K queries (18220-d slice)", flush=True)
    rows = []
    for name, ckpt, arch, out_dim, prefix in jobs:
        m = eval_ckpt(name, ckpt, arch, out_dim, prefix=prefix, base_only=args.base_only)
        b = m["base"]
        rows.append((name, out_dim if arch != "matryoshka" else prefix, b[10], b[50], b[500], m["qps"]))
    print("\nSUMMARY (Stage-1 base)", flush=True)
    print(f"{'method':<32}{'dim':>6}{'R@10':>8}{'R@50':>8}{'R@500':>8}{'QPS':>8}", flush=True)
    for name, dim, r10, r50, r500, qps in rows:
        print(f"{name:<32}{dim:>6}{r10:>8.4f}{r50:>8.4f}{r500:>8.4f}{qps:>8.0f}", flush=True)


if __name__ == "__main__":
    main()
