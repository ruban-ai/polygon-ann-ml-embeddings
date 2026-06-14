#!/usr/bin/env python3
"""
Full-scale recall push: TripletAE (triplet + recon) + WJ HNSW hard-negative mining.

Phases:
  1. Warmup — in-batch WJ triplet + recon, max_pos=100 (init from nb28 ckpt if present)
  2. Mine   — WJ HNSW top-K candidates → explicit (q, pos, neg) triplets
  3. Finetune — explicit triplet loss + anchor recon; re-mine every REMINE_EVERY epochs

Eval: nmslib WJ HNSW (paper protocol) + hnswlib L2 fast path (QPS).

Usage:
  python train_triplet_ae_mined_full.py --phase all
  python train_triplet_ae_mined_full.py --phase warmup --warmup_epochs 20
  python train_triplet_ae_mined_full.py --phase mine
  python train_triplet_ae_mined_full.py --phase finetune --finetune_epochs 15
  python train_triplet_ae_mined_full.py --phase eval
"""
from __future__ import annotations

import argparse
import os
import pickle
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

# ── defaults (full dataset) ──────────────────────────────────────────────────
DATASET = "full"
OUT_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VECS_DEVICE = torch.device("cuda:7" if torch.cuda.device_count() > 7 else "cuda:0")
THREADS = 40
SEED = 42

WARMUP_EPOCHS = 20
FINETUNE_EPOCHS = 15
REMINE_EVERY = 5
BATCH_SIZE_WARMUP = 2048
BATCH_SIZE_MINED = 512
LR_WARMUP = 1e-3
LR_FINETUNE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_POS = 100
MARGIN = 0.3
LAMBDA_RECON = 0.1

HARD_POOL_K = 500
EXCLUDE_GT_TOP = 500
POSITIVE_PER_QUERY = 5
HARD_NEG_PER_POS = 2

CANDIDATE_KS = [1000, 2000]
RERANK_BATCH = 64
HNSW_EF_SEARCH = 200  # nmslib; sweep separately for QPS

METHOD_NAME = "triplet_ae_mined_wj_512"
CKPT_PATH = f"/tmp/best_{METHOD_NAME}_{DATASET}.pt"
NB28_CKPT = f"/tmp/best_sota_triplet_autoencoder_wj_512_{DATASET}.pt"
TRIPLETS_PATH = f"/tmp/triplets_mined_{DATASET}.pkl"
OUT_PATH = f"/tmp/results_{METHOD_NAME}.pkl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wj_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mins = torch.minimum(a, b).sum(dim=-1)
    maxs = torch.maximum(a, b).sum(dim=-1).clamp(min=1e-10)
    return mins / maxs


def wj_triplet_loss_inbatch(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    margin: float = 0.3,
    gt_matrix: torch.Tensor | None = None,
):
    sim_ap = wj_sim(anchors, positives)
    mins_c = torch.min(anchors.unsqueeze(1), positives.unsqueeze(0)).sum(2)
    maxs_c = torch.max(anchors.unsqueeze(1), positives.unsqueeze(0)).sum(2)
    sim_cross = mins_c / maxs_c.clamp(min=1e-10)
    sim_cross.fill_diagonal_(-1e9)
    if gt_matrix is not None:
        fn_mask = gt_matrix.to(sim_cross.device)
        if fn_mask.any():
            sim_cross[fn_mask] = -1e9
    sim_an = sim_cross.max(dim=1).values
    loss = F.relu(sim_an - sim_ap + margin)
    violated = loss > 0
    if violated.sum() == 0:
        z = torch.tensor(0.0, device=anchors.device, requires_grad=True)
        return z, 0
    return loss[violated].mean(), int(violated.sum().item())


def wj_triplet_loss_explicit(za, zp, zn, margin=0.3):
    sim_ap = wj_sim(za, zp)
    sim_an = wj_sim(za, zn)
    loss = F.relu(sim_an - sim_ap + margin)
    violated = loss > 0
    if violated.sum() == 0:
        z = torch.tensor(0.0, device=za.device, requires_grad=True)
        return z, 0
    return loss[violated].mean(), int(violated.sum().item())


class IndexAnchorPositiveDataset(Dataset):
    def __init__(self, gt_lookup, query_start, max_pos=100):
        self.pairs = []
        for qid, neighbors in gt_lookup.items():
            for nid in neighbors[:max_pos]:
                if qid >= query_start and nid < query_start:
                    self.pairs.append((qid, nid))
        random.shuffle(self.pairs)
        print(f"warmup pairs={len(self.pairs):,}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class HardNegTripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class TripletAE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, in_dim, bias=False),
        )

    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z / z.sum(dim=1, keepdim=True).clamp(min=1e-10)

    def forward(self, x):
        z = self.encode(x)
        return z, F.relu(self.decoder(z))


def make_model(in_dim: int, init_ckpt: str | None = None) -> nn.Module:
    n_gpu = torch.cuda.device_count()
    model = TripletAE(in_dim, OUT_DIM)
    if init_ckpt and Path(init_ckpt).exists():
        state = torch.load(init_ckpt, map_location="cpu", weights_only=True)
        # nb28 TripletEncoder: encoder-only weights
        enc_keys = {k: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(enc_keys, strict=False)
        print(f"loaded encoder from {init_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
    return model.to(DEVICE)


def module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def embed_all(model: nn.Module, qt_norm: np.ndarray, batch_size: int = 512) -> np.ndarray:
    enc = module(model)
    enc.eval()
    chunks = []
    for s in range(0, len(qt_norm), batch_size):
        x = torch.tensor(qt_norm[s : s + batch_size], dtype=torch.float32, device=DEVICE)
        chunks.append(enc.encode(x).cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def mine_triplets(
    model: nn.Module,
    qt_norm: np.ndarray,
    gt: dict,
    query_start: int,
    hard_pool_k: int = HARD_POOL_K,
) -> list[tuple[int, int, int]]:
    print(f"Mining hard negatives (WJ HNSW k={hard_pool_k})...")
    embs = embed_all(model, qt_norm)
    corpus_embs = embs[:query_start]
    query_embs = embs[query_start:]
    corpus_size = query_start

    nbrs, info = nmslib_neighbors(
        corpus_embs, query_embs, space="WeightedJaccard", k=hard_pool_k, threads=THREADS
    )
    print(f"mining query done in {info['query_s']:.1f}s")

    query_ids = [qid for qid in sorted(gt) if query_start <= qid < len(qt_norm)]
    triplets = []
    missed = 0
    for local_i, qid in enumerate(tqdm(query_ids, desc="build triplets")):
        positives = [pid for pid in gt.get(qid, []) if 0 <= pid < corpus_size]
        if not positives:
            continue
        pos_train = positives[:POSITIVE_PER_QUERY]
        exclude = set(positives[:EXCLUDE_GT_TOP])
        cand_ids, _ = nbrs[local_i]
        hard_negs = [int(cid) for cid in cand_ids if int(cid) not in exclude]
        if not hard_negs:
            missed += 1
            continue
        for pos_id in pos_train:
            for j in range(HARD_NEG_PER_POS):
                neg_id = hard_negs[min(j, len(hard_negs) - 1)]
                triplets.append((qid, pos_id, neg_id))

    random.shuffle(triplets)
    print(f"triplets={len(triplets):,} | queries w/o hard neg={missed}")
    with open(TRIPLETS_PATH, "wb") as f:
        pickle.dump(triplets, f)
    print(f"saved → {TRIPLETS_PATH}")
    return triplets


def run_warmup(
    model: nn.Module,
    vecs_gpu: torch.Tensor,
    gt_gpu: torch.Tensor,
    gt: dict,
    query_start: int,
    epochs: int,
) -> None:
    dataset = IndexAnchorPositiveDataset(gt, query_start, max_pos=MAX_POS)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_WARMUP,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR_WARMUP, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = tot_trip = tot_rec = tot_viol = steps = 0
        for a_ids, p_ids in tqdm(loader, desc=f"warmup ep{epoch}", leave=False):
            a = vecs_gpu[a_ids.to(VECS_DEVICE)].to(DEVICE)
            p = vecs_gpu[p_ids.to(VECS_DEVICE)].to(DEVICE)
            b = a.shape[0]
            z, xrec = model(torch.cat([a, p]))
            za, zp = z[:b], z[b:]
            xrec_a = xrec[:b]
            fn_mask = build_fn_mask(a_ids, p_ids, gt_gpu, query_start)
            trip, n_viol = wj_triplet_loss_inbatch(za, zp, margin=MARGIN, gt_matrix=fn_mask)
            rec = F.mse_loss(xrec_a, a)
            loss = trip + LAMBDA_RECON * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            tot_trip += trip.item()
            tot_rec += rec.item()
            tot_viol += n_viol
            steps += 1
        sch.step()
        avg = tot_loss / max(steps, 1)
        print(
            f"warmup ep{epoch:02d}/{epochs} loss={avg:.4f} trip={tot_trip/steps:.4f} "
            f"rec={tot_rec/steps:.4f} viol={tot_viol/steps:.1f}"
        )
        if avg < best:
            best = avg
            torch.save(module(model).state_dict(), CKPT_PATH)
            print(f"  → saved {CKPT_PATH}")


def _finetune_loader(triplets: list[tuple[int, int, int]]) -> DataLoader:
    val_n = max(1, int(len(triplets) * 0.1))
    train_triplets = triplets[val_n:]
    return DataLoader(
        HardNegTripletDataset(train_triplets),
        batch_size=BATCH_SIZE_MINED,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )


def run_finetune(
    model: nn.Module,
    vecs_gpu: torch.Tensor,
    triplets: list[tuple[int, int, int]],
    epochs: int,
    remine_fn,
) -> None:
    loader = _finetune_loader(triplets)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best = float("inf")

    for epoch in range(1, epochs + 1):
        if epoch > 1 and (epoch - 1) % REMINE_EVERY == 0 and remine_fn is not None:
            print(f"re-mining at finetune ep{epoch}...")
            triplets = remine_fn()
            loader = _finetune_loader(triplets)

        model.train()
        tot_loss = tot_trip = tot_rec = tot_viol = steps = 0
        for q_ids, p_ids, n_ids in tqdm(loader, desc=f"finetune ep{epoch}", leave=False):
            qa = vecs_gpu[torch.tensor(q_ids, device=VECS_DEVICE)].to(DEVICE)
            pa = vecs_gpu[torch.tensor(p_ids, device=VECS_DEVICE)].to(DEVICE)
            na = vecs_gpu[torch.tensor(n_ids, device=VECS_DEVICE)].to(DEVICE)
            z, xrec = model(torch.cat([qa, pa, na]))
            b = qa.shape[0]
            za, zp, zn = z[:b], z[b : 2 * b], z[2 * b :]
            xrec_q = xrec[:b]
            trip, n_viol = wj_triplet_loss_explicit(za, zp, zn, margin=MARGIN)
            rec = F.mse_loss(xrec_q, qa)
            loss = trip + LAMBDA_RECON * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            tot_trip += trip.item()
            tot_rec += rec.item()
            tot_viol += n_viol
            steps += 1
        sch.step()
        avg = tot_loss / max(steps, 1)
        print(
            f"finetune ep{epoch:02d}/{epochs} loss={avg:.4f} trip={tot_trip/steps:.4f} "
            f"rec={tot_rec/steps:.4f} viol={tot_viol/steps:.1f}"
        )
        if avg < best:
            best = avg
            torch.save(module(model).state_dict(), CKPT_PATH)
            print(f"  → saved {CKPT_PATH}")


def eval_nmslib(
    embs: np.ndarray,
    gt: dict,
    query_start: int,
    corpus_qt: np.ndarray,
    query_qt: np.ndarray,
    corpus_sums: np.ndarray,
    ef_search: int = HNSW_EF_SEARCH,
) -> dict:
    corpus_embs = embs[:query_start]
    query_embs = embs[query_start:]
    max_k = max(max(CANDIDATE_KS), 500)
    nbrs, info = nmslib_neighbors(
        corpus_embs,
        query_embs,
        space="WeightedJaccard",
        k=max_k,
        threads=THREADS,
        query_params={"efSearch": ef_search},
    )
    metrics = {**eval_recall(gt, nbrs, query_start, max_k), **info, "dim": OUT_DIM, "efSearch": ef_search}
    print(f"[nmslib WJ] ef={ef_search} R@50={metrics.get(50, 0):.4f} QPS={metrics['qps']:.1f}")
    save_result(OUT_PATH, DATASET, METHOD_NAME, metrics, meta={"script": "train_triplet_ae_mined_full.py"})

    preload_rerank_corpus(corpus_qt, corpus_sums)
    for ck in CANDIDATE_KS:
        cand, ci = nmslib_neighbors(
            corpus_embs,
            query_embs,
            space="WeightedJaccard",
            k=ck,
            threads=THREADS,
            query_params={"efSearch": ef_search},
        )
        t0 = time.time()
        rr = rerank_wj_gpu(query_qt, cand, corpus_qt, corpus_sums, top_k=ck, batch_size=RERANK_BATCH)
        total_s = ci["query_s"] + (time.time() - t0)
        qps_total = len(query_qt) / max(total_s, 1e-9)
        rr_metrics = {**eval_recall(gt, rr, query_start, ck), "qps": qps_total, "candidate_k": ck}
        key = f"{METHOD_NAME}_rerank_{ck}"
        print(f"[nmslib rerank k={ck}] R@50={rr_metrics.get(50, 0):.4f} QPS={qps_total:.1f}")
        save_result(OUT_PATH, DATASET, key, rr_metrics, meta={"script": "train_triplet_ae_mined_full.py"})
    release_rerank_corpus()
    return metrics


def eval_hnswlib_fast(embs: np.ndarray, gt: dict, query_start: int) -> dict:
    import hnswlib

    corpus_embs = np.ascontiguousarray(embs[:query_start], dtype=np.float32)
    query_embs = np.ascontiguousarray(embs[query_start:], dtype=np.float32)
    dim = corpus_embs.shape[1]
    max_k = max(max(CANDIDATE_KS), 500)

    p = hnswlib.Index(space="l2", dim=dim)
    p.init_index(max_elements=len(corpus_embs), ef_construction=200, M=32)
    p.add_items(corpus_embs, np.arange(len(corpus_embs)))
    p.set_ef(HNSW_EF_SEARCH)

    t0 = time.time()
    labels, _ = p.knn_query(query_embs, k=max_k, num_threads=THREADS)
    query_s = time.time() - t0
    nbrs = [(row.tolist(), None) for row in labels]
    qps = len(query_embs) / max(query_s, 1e-9)
    metrics = {**eval_recall(gt, nbrs, query_start, max_k), "qps": qps, "query_s": query_s}
    key = f"{METHOD_NAME}_hnswlib_l2"
    print(f"[hnswlib L2] R@50={metrics.get(50, 0):.4f} QPS={qps:.1f}")
    save_result(OUT_PATH, DATASET, key, metrics, meta={"script": "train_triplet_ae_mined_full.py"})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["all", "warmup", "mine", "finetune", "eval"],
        default="all",
    )
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--finetune_epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--init_ckpt", default=NB28_CKPT)
    parser.add_argument("--no_init", action="store_true", help="train from scratch")
    args = parser.parse_args()

    set_seed(SEED)
    qt, gt, query_start, corpus_qt, query_qt, corpus_sums, qt_norm = load_dataset_normalized(DATASET)

    print(f"Loading vectors to {VECS_DEVICE}...")
    vecs_gpu = torch.from_numpy(np.ascontiguousarray(qt_norm, dtype=np.float32)).to(VECS_DEVICE)
    print(f"vecs_gpu: {vecs_gpu.nbytes / 1024**3:.2f} GB")

    init = None if args.no_init else args.init_ckpt
    model = make_model(qt_norm.shape[1], init_ckpt=init)

    if Path(CKPT_PATH).exists() and args.phase in ("finetune", "eval"):
        module(model).load_state_dict(
            torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True), strict=False
        )
        print(f"resumed from {CKPT_PATH}")

    gt_stacked = None
    gt_gpu = None
    if args.phase in ("all", "warmup"):
        gt_stacked = build_gt_cache(gt, len(qt_norm), query_start, DATASET)
        gt_gpu = build_gt_gpu(gt_stacked, VECS_DEVICE)

    if args.phase in ("all", "warmup"):
        run_warmup(model, vecs_gpu, gt_gpu, gt, query_start, args.warmup_epochs)

    if args.phase in ("all", "mine", "finetune"):
        triplets = mine_triplets(model, qt_norm, gt, query_start)

    if args.phase == "finetune":
        if Path(TRIPLETS_PATH).exists():
            with open(TRIPLETS_PATH, "rb") as f:
                triplets = pickle.load(f)
            print(f"loaded {len(triplets):,} triplets from {TRIPLETS_PATH}")
        else:
            triplets = mine_triplets(model, qt_norm, gt, query_start)

    if args.phase in ("all", "finetune"):
        remine = lambda: mine_triplets(model, qt_norm, gt, query_start)
        run_finetune(model, vecs_gpu, triplets, args.finetune_epochs, remine_fn=remine)

    if args.phase in ("all", "eval"):
        if Path(CKPT_PATH).exists():
            model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True), strict=False)
        embs = embed_all(model, qt_norm)
        eval_nmslib(embs, gt, query_start, corpus_qt, query_qt, corpus_sums)
        eval_hnswlib_fast(embs, gt, query_start)
        print(f"\nDone. Results → {OUT_PATH}")


if __name__ == "__main__":
    main()
