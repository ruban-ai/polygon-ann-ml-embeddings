#!/usr/bin/env python3
"""
nb28_3 — TripletAE + recon + corpus hard-negative mining (full scale).

Phases (run via nohup or notebook cells):
  warmup   — in-batch WJ triplet + recon, max_pos=100, init encoder from nb28
  mine     — WJ HNSW top-K → explicit (q, pos, neg) triplets
  finetune — explicit triplets + recon; re-mine every REMINE_EVERY epochs
  eval     — nmslib WJ HNSW + rerank@1k/2k (paper protocol)
  all      — warmup → mine → finetune → eval

Example:
  nohup python run_28_3_phase.py --phase warmup > /tmp/nb28_3_warmup.log 2>&1 &
  python run_28_3_phase.py --phase mine
  nohup python run_28_3_phase.py --phase finetune > /tmp/nb28_3_finetune.log 2>&1 &
  python run_28_3_phase.py --phase eval
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
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

DATASET = "full"
OUT_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VECS_DEVICE = torch.device("cuda:7" if torch.cuda.device_count() > 7 else "cuda:0")
THREADS = 150
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
HNSW_EF_SEARCH = 200

METHOD_NAME = "triplet_ae_mined_wj_512"
NOTEBOOK_NAME = "28_3_triplet_ae_mined_wj_512.ipynb"
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


def wj_triplet_loss_inbatch(anchors, positives, margin=0.3, gt_matrix=None):
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
        print(f"warmup pairs={len(self.pairs):,}  steps/epoch≈{len(self.pairs)//BATCH_SIZE_WARMUP}")

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
        qid, pos_id, neg_id = self.triplets[idx]
        return qid, pos_id, neg_id


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


def module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def make_model(in_dim: int, init_ckpt: str | None = None, full_ckpt: str | None = None) -> nn.Module:
    n_gpu = torch.cuda.device_count()
    model = TripletAE(in_dim, OUT_DIM)
    if full_ckpt and Path(full_ckpt).exists():
        state = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        module(model).load_state_dict(state, strict=False)
        print(f"loaded full TripletAE from {full_ckpt}")
    elif init_ckpt and Path(init_ckpt).exists():
        state = torch.load(init_ckpt, map_location="cpu", weights_only=True)
        enc_keys = {k: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = module(model).load_state_dict(enc_keys, strict=False)
        print(f"loaded encoder from {init_ckpt} | missing={len(missing)} unexpected={len(unexpected)}")
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


def mine_triplets(model, qt_norm, gt, query_start, hard_pool_k=HARD_POOL_K):
    print(f"Mining hard negatives (WJ HNSW k={hard_pool_k})...")
    embs = embed_all(model, qt_norm)
    corpus_embs = embs[:query_start]
    query_embs = embs[query_start:]
    corpus_size = query_start

    nbrs, info = nmslib_neighbors(
        corpus_embs, query_embs, space="WeightedJaccard", k=hard_pool_k, threads=THREADS
    )
    print(f"mining queries done in {info['query_s']:.1f}s")

    query_ids = [qid for qid in sorted(gt) if query_start <= qid < len(qt_norm)]
    triplets = []
    missed = 0
    for qid in tqdm(query_ids, desc="build triplets"):
        positives = [pid for pid in gt.get(qid, []) if 0 <= pid < corpus_size]
        if not positives:
            continue
        pos_train = positives[:POSITIVE_PER_QUERY]
        exclude = set(positives[:EXCLUDE_GT_TOP])
        # nmslib_neighbors returns list[list[int]]; index by query offset, not enumerate idx
        cand_ids = nbrs[qid - query_start]
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


def run_warmup(model, vecs_gpu, gt_gpu, gt, query_start, epochs):
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
    t0_train = time.time()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            tot_trip += trip.item()
            tot_rec += rec.item()
            tot_viol += n_viol
            steps += 1
        sch.step()
        avg = tot_loss / max(steps, 1)
        elapsed = (time.time() - t0_train) / 60
        eta = elapsed / epoch * (epochs - epoch)
        print(
            f"warmup ep{epoch:02d}/{epochs} loss={avg:.4f} trip={tot_trip/steps:.4f} "
            f"rec={tot_rec/steps:.4f} viol={tot_viol/steps:.1f} "
            f"{elapsed:.1f}min eta={eta:.1f}min",
            flush=True,
        )
        if avg < best:
            best = avg
            torch.save(module(model).state_dict(), CKPT_PATH)
            print(f"  → saved {CKPT_PATH}", flush=True)


def _finetune_loader(triplets):
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


def run_finetune(model, vecs_gpu, triplets, epochs, remine_fn):
    loader = _finetune_loader(triplets)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best = float("inf")
    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        if epoch > 1 and (epoch - 1) % REMINE_EVERY == 0 and remine_fn is not None:
            print(f"re-mining at finetune ep{epoch}...", flush=True)
            triplets = remine_fn()
            loader = _finetune_loader(triplets)

        model.train()
        tot_loss = tot_trip = tot_rec = tot_viol = steps = 0
        for q_ids, p_ids, n_ids in tqdm(loader, desc=f"finetune ep{epoch}", leave=False):
            qa = vecs_gpu[q_ids.to(VECS_DEVICE)].to(DEVICE)
            pa = vecs_gpu[p_ids.to(VECS_DEVICE)].to(DEVICE)
            na = vecs_gpu[n_ids.to(VECS_DEVICE)].to(DEVICE)
            z, xrec = model(torch.cat([qa, pa, na]))
            b = qa.shape[0]
            za, zp, zn = z[:b], z[b : 2 * b], z[2 * b :]
            xrec_q = xrec[:b]
            trip, n_viol = wj_triplet_loss_explicit(za, zp, zn, margin=MARGIN)
            rec = F.mse_loss(xrec_q, qa)
            loss = trip + LAMBDA_RECON * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item()
            tot_trip += trip.item()
            tot_rec += rec.item()
            tot_viol += n_viol
            steps += 1
        sch.step()
        avg = tot_loss / max(steps, 1)
        elapsed = (time.time() - t0_train) / 60
        eta = elapsed / epoch * (epochs - epoch)
        print(
            f"finetune ep{epoch:02d}/{epochs} loss={avg:.4f} trip={tot_trip/steps:.4f} "
            f"rec={tot_rec/steps:.4f} viol={tot_viol/steps:.1f} "
            f"{elapsed:.1f}min eta={eta:.1f}min",
            flush=True,
        )
        if avg < best:
            best = avg
            torch.save(module(model).state_dict(), CKPT_PATH)
            print(f"  → saved {CKPT_PATH}", flush=True)


def eval_nmslib(embs, gt, query_start, corpus_qt, query_qt, corpus_sums):
    corpus_embs = embs[:query_start]
    query_embs = embs[query_start:]
    max_k = max(max(CANDIDATE_KS), 500)
    nbrs, info = nmslib_neighbors(
        corpus_embs,
        query_embs,
        space="WeightedJaccard",
        k=max_k,
        threads=THREADS,
        query_params={"efSearch": HNSW_EF_SEARCH},
    )
    metrics = {
        **eval_recall(gt, nbrs, query_start, max_k),
        **info,
        "dim": OUT_DIM,
        "efSearch": HNSW_EF_SEARCH,
    }
    print(f"[nmslib WJ] R@50={metrics.get(50, 0):.4f} QPS={metrics['qps']:.1f}")
    save_result(OUT_PATH, DATASET, METHOD_NAME, metrics, meta={"notebook": NOTEBOOK_NAME})

    preload_rerank_corpus(corpus_qt, corpus_sums)
    for ck in CANDIDATE_KS:
        cand, ci = nmslib_neighbors(
            corpus_embs,
            query_embs,
            space="WeightedJaccard",
            k=ck,
            threads=THREADS,
            query_params={"efSearch": HNSW_EF_SEARCH},
        )
        t0 = time.time()
        rr = rerank_wj_gpu(query_qt, cand, corpus_qt, corpus_sums, top_k=ck, batch_size=RERANK_BATCH)
        total_s = ci["query_s"] + (time.time() - t0)
        qps_total = len(query_qt) / max(total_s, 1e-9)
        rr_metrics = {**eval_recall(gt, rr, query_start, ck), "qps": qps_total, "candidate_k": ck}
        key = f"{METHOD_NAME}_rerank_{ck}"
        print(f"[nmslib rerank k={ck}] R@50={rr_metrics.get(50, 0):.4f} QPS={qps_total:.1f}")
        save_result(OUT_PATH, DATASET, key, rr_metrics, meta={"notebook": NOTEBOOK_NAME})
    release_rerank_corpus()
    return metrics


def print_status():
    paths = {
        "nb28 ckpt": NB28_CKPT,
        "28_3 ckpt": CKPT_PATH,
        "triplets": TRIPLETS_PATH,
        "results": OUT_PATH,
    }
    print("── nb28_3 artifact status ──")
    for label, p in paths.items():
        ok = Path(p).exists()
        extra = ""
        if ok and p.endswith(".pt"):
            extra = f" ({Path(p).stat().st_size / 1e6:.0f} MB)"
        elif ok and p.endswith(".pkl") and label == "triplets":
            with open(p, "rb") as f:
                extra = f" ({len(pickle.load(f)):,} triplets)"
        print(f"  {'✓' if ok else '✗'} {label}: {p}{extra}")


def main():
    parser = argparse.ArgumentParser(description="nb28_3 phase runner")
    parser.add_argument(
        "--phase",
        choices=["status", "warmup", "mine", "finetune", "eval", "all"],
        default="status",
    )
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--finetune_epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--resume", action="store_true", help="resume from 28_3 ckpt instead of nb28 encoder init")
    args = parser.parse_args()

    if args.phase == "status":
        print_status()
        return

    set_seed(SEED)
    qt, gt, query_start, corpus_qt, query_qt, corpus_sums, qt_norm = load_dataset_normalized(DATASET)

    print(f"Loading vectors to {VECS_DEVICE}...")
    vecs_gpu = torch.from_numpy(np.ascontiguousarray(qt_norm, dtype=np.float32)).to(VECS_DEVICE)
    print(f"vecs_gpu: {vecs_gpu.nbytes / 1024**3:.2f} GB on {VECS_DEVICE}")

    if args.phase in ("warmup", "all"):
        if args.resume and Path(CKPT_PATH).exists():
            model = make_model(qt_norm.shape[1], full_ckpt=CKPT_PATH)
        else:
            model = make_model(qt_norm.shape[1], init_ckpt=NB28_CKPT)
        gt_stacked = build_gt_cache(gt, len(qt_norm), query_start, DATASET)
        gt_gpu = build_gt_gpu(gt_stacked, VECS_DEVICE)
        del gt_stacked
        run_warmup(model, vecs_gpu, gt_gpu, gt, query_start, args.warmup_epochs)

    if args.phase in ("mine", "all"):
        if args.phase == "mine":
            if not Path(CKPT_PATH).exists():
                raise FileNotFoundError(f"warmup ckpt missing: {CKPT_PATH}")
            model = make_model(qt_norm.shape[1], full_ckpt=CKPT_PATH)
        mine_triplets(model, qt_norm, gt, query_start)

    if args.phase in ("finetune", "all"):
        if not Path(CKPT_PATH).exists():
            raise FileNotFoundError(f"ckpt missing: {CKPT_PATH}")
        model = make_model(qt_norm.shape[1], full_ckpt=CKPT_PATH)
        if Path(TRIPLETS_PATH).exists():
            with open(TRIPLETS_PATH, "rb") as f:
                triplets = pickle.load(f)
            print(f"loaded {len(triplets):,} triplets from {TRIPLETS_PATH}")
        else:
            triplets = mine_triplets(model, qt_norm, gt, query_start)
        remine = lambda: mine_triplets(model, qt_norm, gt, query_start)
        run_finetune(model, vecs_gpu, triplets, args.finetune_epochs, remine_fn=remine)

    if args.phase == "eval":
        if not Path(CKPT_PATH).exists():
            raise FileNotFoundError(f"ckpt missing: {CKPT_PATH}")
        model = make_model(qt_norm.shape[1], full_ckpt=CKPT_PATH)
        embs = embed_all(model, qt_norm)
        eval_nmslib(embs, gt, query_start, corpus_qt, query_qt, corpus_sums)
        print(f"\nDone. Results → {OUT_PATH}")

    if args.phase == "all":
        model = make_model(qt_norm.shape[1], full_ckpt=CKPT_PATH)
        embs = embed_all(model, qt_norm)
        eval_nmslib(embs, gt, query_start, corpus_qt, query_qt, corpus_sums)
        print(f"\nAll phases done. Results → {OUT_PATH}")


if __name__ == "__main__":
    main()
