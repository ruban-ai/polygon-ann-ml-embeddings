#!/usr/bin/env python3
"""
nb28_4 — Filter-recall objective for two-stage WJ ANN (full scale).

Motivation (see thread): on the full corpus the margin-triplet objective
collapses filter R@500 (true broad-neighborhood neighbors fall outside the
candidate pool) because it only supervises the top-`max_pos` positives vs the
single hardest in-batch negative, and saturates once the margin is met.

A two-stage system only needs the *filter* to put every true neighbor inside the
top-K candidate pool — internal ranking is fixed by the exact 18K-dim rerank.
That is a *pool-membership* objective, not a ranking one: pull the whole broad
neighborhood together. We implement it as a non-saturating multi-positive
in-batch InfoNCE over WJ similarity, with a large `max_pos` so mid/tail
neighbors are supervised, FN-masked so a query's other in-batch true neighbors
are not treated as negatives. `--loss triplet` reproduces the nb28 baseline on
identical plumbing for an apples-to-apples comparison.

Examples:
  # 10K sanity (fast), compare both losses:
  python run_28_4_filter_recall.py --dataset 10k --loss infonce --epochs 12 --phase all
  python run_28_4_filter_recall.py --dataset 10k --loss triplet --epochs 12 --phase all

  # full overnight:
  nohup python run_28_4_filter_recall.py --dataset full --loss infonce \
        --epochs 60 --phase all > /tmp/nb28_4_full_infonce.log 2>&1 &
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sota_experiment_common import (
    build_fn_mask,
    build_gt_cache,
    build_gt_gpu,
    eval_recall,
    load_dataset_normalized,
    nmslib_neighbors,
    preload_rerank_corpus,
    release_rerank_corpus,
    rerank_wj_gpu,
    save_result,
)

OUT_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VECS_DEVICE = torch.device("cuda:7" if torch.cuda.device_count() > 7 else "cuda:0")
THREADS = 150
SEED = 42

BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_POS = 256            # broad neighborhood — supervise mid/tail, targets filter R@500
MARGIN = 0.3            # only used by --loss triplet
TEMPERATURE = 0.07      # InfoNCE temperature
LAMBDA_RECON = 0.1
HNSW_EF_SEARCH = 200
RERANK_BATCH = 64


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── losses ────────────────────────────────────────────────────────────────────
def wj_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mins = torch.minimum(a, b).sum(dim=-1)
    maxs = torch.maximum(a, b).sum(dim=-1).clamp(min=1e-10)
    return mins / maxs


def _cross_wj(anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """(B,B) WJ similarity between every anchor and every batch positive."""
    mins_c = torch.min(anchors.unsqueeze(1), positives.unsqueeze(0)).sum(2)
    maxs_c = torch.max(anchors.unsqueeze(1), positives.unsqueeze(0)).sum(2)
    return mins_c / maxs_c.clamp(min=1e-10)


def wj_infonce_inbatch(anchors, positives, temperature=TEMPERATURE, gt_matrix=None):
    """Non-saturating in-batch InfoNCE on WJ sim. Diagonal = the paired positive;
    all other in-batch positives are negatives, except FN-masked true neighbors."""
    sim = _cross_wj(anchors, positives)            # (B, B), diagonal is true positive
    logits = sim / temperature
    if gt_matrix is not None:
        mask = gt_matrix.to(logits.device).clone()
        mask.fill_diagonal_(False)                  # never mask the designated positive
        logits = logits.masked_fill(mask, -1e9)     # drop a query's other true neighbors
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
    return loss, acc


def wj_triplet_inbatch(anchors, positives, margin=MARGIN, gt_matrix=None):
    """nb28 baseline: hardest in-batch negative + margin (saturating)."""
    sim_ap = wj_sim(anchors, positives)
    sim_cross = _cross_wj(anchors, positives)
    sim_cross.fill_diagonal_(-1e9)
    if gt_matrix is not None:
        mask = gt_matrix.to(sim_cross.device)
        if mask.any():
            sim_cross[mask] = -1e9
    sim_an = sim_cross.max(dim=1).values
    loss = F.relu(sim_an - sim_ap + margin)
    violated = loss > 0
    if violated.sum() == 0:
        return torch.tensor(0.0, device=anchors.device, requires_grad=True), 0.0
    return loss[violated].mean(), float(violated.float().mean().item())


# ── data / model ────────────────────────────────────────────────────────────────
class IndexAnchorPositiveDataset(Dataset):
    def __init__(self, gt_lookup, query_start, max_pos=MAX_POS):
        self.pairs = []
        for qid, neighbors in gt_lookup.items():
            for nid in neighbors[:max_pos]:
                if qid >= query_start and nid < query_start:
                    self.pairs.append((qid, nid))
        random.shuffle(self.pairs)
        print(f"pairs={len(self.pairs):,}  steps/epoch≈{len(self.pairs)//BATCH_SIZE}", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class TripletAE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, in_dim, bias=False),
        )

    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z / z.sum(dim=1, keepdim=True).clamp(min=1e-10)

    def forward(self, x):
        z = self.encode(x)
        return z, F.relu(self.decoder(z))


def module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def make_model(in_dim: int, full_ckpt: str | None = None) -> nn.Module:
    n_gpu = torch.cuda.device_count()
    model = TripletAE(in_dim, OUT_DIM)
    if full_ckpt and Path(full_ckpt).exists():
        state = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        module(model).load_state_dict(state, strict=False)
        print(f"resumed from {full_ckpt}", flush=True)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
    return model.to(DEVICE)


@torch.no_grad()
def embed_all(model: nn.Module, qt_norm: np.ndarray, batch_size: int = 512) -> np.ndarray:
    enc = module(model)
    enc.eval()
    chunks = []
    for s in range(0, len(qt_norm), batch_size):
        x = torch.tensor(qt_norm[s : s + batch_size], dtype=torch.float32, device=DEVICE)
        chunks.append(enc.encode(x).cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


# ── train ──────────────────────────────────────────────────────────────────────
def run_train(model, vecs_gpu, gt_gpu, gt, query_start, epochs, loss_name, ckpt_path):
    dataset = IndexAnchorPositiveDataset(gt, query_start, max_pos=MAX_POS)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best = float("inf")
    t0 = time.time()
    loss_fn = wj_infonce_inbatch if loss_name == "infonce" else wj_triplet_inbatch

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = tot_main = tot_rec = tot_metric = steps = 0
        for a_ids, p_ids in tqdm(loader, desc=f"ep{epoch}", leave=False):
            a = vecs_gpu[a_ids.to(VECS_DEVICE)].to(DEVICE)
            p = vecs_gpu[p_ids.to(VECS_DEVICE)].to(DEVICE)
            b = a.shape[0]
            z, xrec = model(torch.cat([a, p]))
            za, zp = z[:b], z[b:]
            fn_mask = build_fn_mask(a_ids, p_ids, gt_gpu, query_start)
            main, metric = loss_fn(za, zp, gt_matrix=fn_mask)
            rec = F.mse_loss(xrec[:b], a)
            loss = main + LAMBDA_RECON * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item(); tot_main += main.item()
            tot_rec += rec.item(); tot_metric += metric; steps += 1
        sch.step()
        avg = tot_loss / max(steps, 1)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / epoch * (epochs - epoch)
        metric_name = "acc" if loss_name == "infonce" else "viol_frac"
        print(
            f"ep{epoch:02d}/{epochs} loss={avg:.4f} main={tot_main/steps:.4f} "
            f"rec={tot_rec/steps:.4f} {metric_name}={tot_metric/steps:.3f} "
            f"{elapsed:.1f}min eta={eta:.1f}min",
            flush=True,
        )
        if avg < best:
            best = avg
            torch.save(module(model).state_dict(), ckpt_path)
            print(f"  → saved {ckpt_path}", flush=True)
    torch.save(module(model).state_dict(), ckpt_path.replace(".pt", "_last.pt"))


# ── eval (paper protocol: nmslib WJ HNSW + exact rerank@1k/2k) ────────────────────
def run_eval(model, dataset_name, gt, query_start, corpus_qt, query_qt, corpus_sums,
             qt_norm, method_name, out_path, notebook_name, candidate_ks):
    embs = embed_all(model, qt_norm)
    corpus_embs, query_embs = embs[:query_start], embs[query_start:]
    max_k = max(max(candidate_ks), 500)
    nbrs, info = nmslib_neighbors(
        corpus_embs, query_embs, space="WeightedJaccard", k=max_k,
        threads=THREADS, query_params={"efSearch": HNSW_EF_SEARCH},
    )
    metrics = {**eval_recall(gt, nbrs, query_start, max_k), **info, "dim": OUT_DIM}
    print(f"[base] R@50={metrics.get(50,0):.4f} R@500={metrics.get(500,0):.4f} QPS={metrics['qps']:.1f}", flush=True)
    save_result(out_path, dataset_name, method_name, metrics, meta={"notebook": notebook_name})

    preload_rerank_corpus(corpus_qt, corpus_sums)
    for ck in candidate_ks:
        cand, ci = nmslib_neighbors(
            corpus_embs, query_embs, space="WeightedJaccard", k=ck,
            threads=THREADS, query_params={"efSearch": HNSW_EF_SEARCH},
        )
        t0 = time.time()
        rr = rerank_wj_gpu(query_qt, cand, corpus_qt, corpus_sums, top_k=ck, batch_size=RERANK_BATCH)
        qps_total = len(query_qt) / max(ci["query_s"] + (time.time() - t0), 1e-9)
        rr_metrics = {**eval_recall(gt, rr, query_start, ck), "qps": qps_total,
                      "qps_candidates": ci["qps"], "candidate_k": ck}
        key = f"{method_name}_rerank_{ck}"
        print(f"[rerank k={ck}] R@50={rr_metrics.get(50,0):.4f} R@500={rr_metrics.get(500,0):.4f} "
              f"QPS={qps_total:.1f}", flush=True)
        save_result(out_path, dataset_name, key, rr_metrics, meta={"notebook": notebook_name})
    release_rerank_corpus()


def main():
    ap = argparse.ArgumentParser(description="nb28_4 filter-recall trainer")
    ap.add_argument("--dataset", choices=["10k", "full"], default="full")
    ap.add_argument("--loss", choices=["infonce", "triplet"], default="infonce")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(SEED)
    method_name = f"filter_recall_{args.loss}_{OUT_DIM}"
    notebook_name = "28_4_filter_recall.py"
    ckpt_path = f"/tmp/best_{method_name}_{args.dataset}.pt"
    out_path = f"/tmp/results_{method_name}.pkl"
    candidate_ks = [500, 1000] if args.dataset == "10k" else [1000, 2000]
    print(f"dataset={args.dataset} loss={args.loss} epochs={args.epochs} max_pos={MAX_POS} "
          f"method={method_name}", flush=True)

    qt, gt, query_start, corpus_qt, query_qt, corpus_sums, qt_norm = load_dataset_normalized(args.dataset)
    print(f"Loading vectors to {VECS_DEVICE}...", flush=True)
    vecs_gpu = torch.from_numpy(np.ascontiguousarray(qt_norm, dtype=np.float32)).to(VECS_DEVICE)
    print(f"vecs_gpu: {vecs_gpu.nbytes/1024**3:.2f} GB on {VECS_DEVICE}", flush=True)

    model = make_model(qt_norm.shape[1], full_ckpt=ckpt_path if args.resume else None)

    if args.phase in ("train", "all"):
        gt_stacked = build_gt_cache(gt, len(qt_norm), query_start, args.dataset)
        gt_gpu = build_gt_gpu(gt_stacked, VECS_DEVICE)
        del gt_stacked
        run_train(model, vecs_gpu, gt_gpu, gt, query_start, args.epochs, args.loss, ckpt_path)

    if args.phase in ("eval", "all"):
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"ckpt missing: {ckpt_path}")
        module(model).load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True), strict=False)
        run_eval(model, args.dataset, gt, query_start, corpus_qt, query_qt, corpus_sums,
                 qt_norm, method_name, out_path, notebook_name, candidate_ks)
        print(f"\nDone → {out_path}", flush=True)


if __name__ == "__main__":
    main()
