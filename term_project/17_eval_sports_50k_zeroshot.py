#!/usr/bin/env python3
"""
Zero-shot eval: parks-trained intersection-min MLP on sports 50k.
No retraining — load encodings + GT, embed, intersection KNN, optional raw-WJ rerank.
"""
from __future__ import annotations

import glob
import os
import pickle
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ENC_DIR = "/mnt/data1/ruban/encodings/sports-50k0.002"
GT_DIR = "/mnt/data1/ruban/groundtruth/sports-query-50k"
CKPT_PATH = "/tmp/best_compressor_intersection_min_10k.pt"
CACHE_QT = "/tmp/qt_sports_50k.npy"
CACHE_GT = "/tmp/gt_lookup_sports_50k.pkl"
OUT_PATH = "/tmp/results_sports_50k_zeroshot.pkl"

PARKS_IN_DIM = 18499  # trained model input size
SPORTS_VOCAB = 18382  # weightint cell vocabulary (MLP baseline)
POLY_COUNT = 50_000
DATA_END = 40_000
QUERY_START = 40_000
QUERY_END = 50_000

device_str = "cuda:0"
seed = 42
candidate_ks = [500, 1000]
search_corpus_chunk = 2000
rerank_batch_size = 16
n_workers = 16


# ── Model (notebook 16) ───────────────────────────────────────────────────────
class QuadtreeCompressorMin(nn.Module):
    def __init__(self, in_dim, out_dim=512, use_log1p=False):
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
        out = self.net(x)
        out = F.relu(out)
        return out / out.sum(dim=1, keepdim=True).clamp(min=1e-10)


def intersection(a, b):
    return torch.minimum(a, b).sum(dim=-1)


def load_sports_encodings(enc_dir: str, vocab_size: int, total: int, n_workers: int) -> np.ndarray:
    def load_single_file(args):
        fpath, vs = args
        start_id = int(
            os.path.basename(fpath).replace("weightint_", "").replace(".txt", "")
        )
        rows = []
        with open(fpath) as f:
            for local_idx, line in enumerate(f):
                ids = [int(x) for x in line.strip().split() if x.strip()]
                ids = [i for i in ids if i < vs]
                rows.append((start_id + local_idx, ids))
        return rows

    files = sorted(
        glob.glob(os.path.join(enc_dir, "weightint_*.txt")),
        key=lambda x: int(
            os.path.basename(x).replace("weightint_", "").replace(".txt", "")
        ),
    )
    print(f"Found {len(files)} weightint files under {enc_dir}")
    matrix = np.zeros((total, vocab_size), dtype=np.float32)
    args = [(f, vocab_size) for f in files]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for file_rows in tqdm(ex.map(load_single_file, args), total=len(files), desc="Load files"):
            for poly_id, ids in file_rows:
                if poly_id >= total:
                    continue
                if ids:
                    matrix[poly_id, ids] = 1.0
    active = (matrix > 0).sum(axis=1).mean()
    print(f"Matrix {matrix.shape} | avg active cells: {active:.1f}")
    return matrix


def pad_to_parks_dim(qt: np.ndarray, target_dim: int) -> np.ndarray:
    if qt.shape[1] == target_dim:
        return qt
    if qt.shape[1] > target_dim:
        raise ValueError(f"Input dim {qt.shape[1]} > parks model {target_dim}")
    pad = np.zeros((qt.shape[0], target_dim - qt.shape[1]), dtype=np.float32)
    print(f"Padding sports {qt.shape[1]} -> {target_dim} (zeros at end)")
    return np.hstack([qt, pad])


def load_gt(gt_dir: str, query_start: int, query_end: int) -> dict[int, list[int]]:
    gt: dict[int, list[int]] = {}
    for fname in tqdm(sorted(os.listdir(gt_dir)), desc="Load GT"):
        with open(os.path.join(gt_dir, fname)) as f:
            content = f.read().strip()
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            ids = [
                int(x.strip())
                for x in line.split(",")
                if x.strip().lstrip("-").isdigit()
            ]
            if len(ids) < 2:
                continue
            qid = ids[0]
            if query_start <= qid < query_end:
                gt[qid] = ids[1:]
    return gt


@torch.no_grad()
def generate_embeddings(model, data, dev, batch_size=512):
    model.eval()
    chunks = []
    for start in tqdm(range(0, len(data), batch_size), desc="Embedding"):
        batch = torch.tensor(data[start : start + batch_size], dtype=torch.float32, device=dev)
        chunks.append(model(batch).cpu().numpy())
    return np.vstack(chunks)


@torch.no_grad()
def knn_intersection_gpu(query_embs, corpus_embs, k, dev, corpus_chunk=2000):
    q = torch.from_numpy(query_embs).to(dev, dtype=torch.float32)
    c_all = torch.from_numpy(corpus_embs).to(dev, dtype=torch.float32)
    n_q, n_c = q.shape[0], c_all.shape[0]
    top_ids = np.zeros((n_q, k), dtype=np.int64)
    top_scores = np.full((n_q, k), -1.0, dtype=np.float32)

    for qs in tqdm(range(0, n_q, 64), desc="KNN intersection"):
        qe = min(qs + 64, n_q)
        qb = q[qs:qe]
        best_scores = torch.full((qb.shape[0], k), -1.0, device=dev)
        best_ids = torch.zeros((qb.shape[0], k), dtype=torch.long, device=dev)

        for cs in range(0, n_c, corpus_chunk):
            ce = min(cs + corpus_chunk, n_c)
            cb = c_all[cs:ce]
            scores = torch.min(qb[:, None, :], cb[None, :, :]).sum(dim=2)
            cand_ids = torch.arange(cs, ce, device=dev).expand(qb.shape[0], -1)
            merged_scores = torch.cat([best_scores, scores], dim=1)
            merged_ids = torch.cat([best_ids, cand_ids], dim=1)
            new_scores, order = torch.topk(
                merged_scores, k=min(k, merged_scores.shape[1]), dim=1
            )
            best_ids = torch.gather(merged_ids, 1, order)
            best_scores = new_scores

        top_ids[qs:qe] = best_ids.cpu().numpy()
        top_scores[qs:qe] = best_scores.cpu().numpy()

    return [(top_ids[i].tolist(), top_scores[i].tolist()) for i in range(n_q)]


def recall_at_k(gt_lookup, nbrs, query_start_id, k):
    total = 0.0
    count = 0
    for i, ids in enumerate(nbrs):
        qid = query_start_id + i
        gt_set = set(gt_lookup.get(qid, [])[:k])
        if not gt_set:
            continue
        total += len(gt_set & set(ids[:k])) / len(gt_set)
        count += 1
    return total / count if count else 0.0


def eval_recall_dict(gt_lookup, nbrs, query_start_id, max_k):
    return {
        k: recall_at_k(gt_lookup, nbrs, query_start_id, k)
        for k in (10, 50, 100, 500)
        if k <= max_k
    }


@torch.no_grad()
def rerank_wj_gpu(query_qt, nbrs_ids, corpus_qt, corpus_sums, dev, batch_size=16):
    corpus_t = torch.from_numpy(corpus_qt).to(dev, dtype=torch.float32)
    corpus_sums_t = torch.from_numpy(corpus_sums).to(dev, dtype=torch.float32)
    reranked = []
    for start in tqdm(range(0, len(nbrs_ids), batch_size), desc="Raw WJ rerank"):
        batch = nbrs_ids[start : start + batch_size]
        groups: dict[int, list] = {}
        for offset, ids in enumerate(batch):
            ids_arr = np.asarray(ids, dtype=np.int64)
            groups.setdefault(len(ids_arr), []).append((start + offset, ids_arr))
        for cand_len, items in groups.items():
            if cand_len == 0:
                for _, _ in items:
                    reranked.append([])
                continue
            ids_np = np.stack([ids for _, ids in items], axis=0)
            query_np = np.stack([query_qt[abs_i] for abs_i, _ in items], axis=0)
            ids_t = torch.from_numpy(ids_np).to(dev)
            q_t = torch.from_numpy(query_np).to(dev, dtype=torch.float32)
            c_t = corpus_t[ids_t]
            mins = torch.minimum(q_t[:, None, :], c_t).sum(dim=2)
            maxs = q_t.sum(dim=1, keepdim=True) + corpus_sums_t[ids_t] - mins
            order = torch.argsort(
                mins / maxs.clamp_min(1e-10), dim=1, descending=True
            ).cpu().numpy()
            for row, (_, ids) in zip(order, items):
                reranked.append(ids[row].tolist())
    return reranked


def main():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")

    if os.path.exists(CACHE_QT):
        print(f"Loading cached quadtrees: {CACHE_QT}")
        qt = np.load(CACHE_QT)
    else:
        qt = load_sports_encodings(ENC_DIR, SPORTS_VOCAB, POLY_COUNT, n_workers)
        qt = pad_to_parks_dim(qt, PARKS_IN_DIM)
        np.save(CACHE_QT, qt)
        print(f"Cached {CACHE_QT}")

    if os.path.exists(CACHE_GT):
        with open(CACHE_GT, "rb") as f:
            gt = pickle.load(f)
        print(f"Loaded GT cache: {len(gt)} queries")
    else:
        gt = load_gt(GT_DIR, QUERY_START, QUERY_END)
        with open(CACHE_GT, "wb") as f:
            pickle.dump(gt, f)
        print(f"Cached {CACHE_GT} ({len(gt)} queries)")

    corpus_qt = qt[:DATA_END]
    query_qt = qt[QUERY_START:QUERY_END]
    print(f"corpus={corpus_qt.shape} | queries={query_qt.shape} | GT queries={len(gt)}")

    model = QuadtreeCompressorMin(PARKS_IN_DIM, out_dim=512, use_log1p=False).to(dev)
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=dev, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    print("\n" + "=" * 72)
    print("ZERO-SHOT: parks intersection-min → sports 50k")
    print("=" * 72)

    embs = generate_embeddings(model, qt, dev)
    corpus_embs = embs[:DATA_END]
    query_embs = embs[QUERY_START:QUERY_END]
    corpus_sums = corpus_qt.sum(axis=1).astype(np.float32)

    results = {}
    max_k = max(max(candidate_ks), 500)

    t0 = time.time()
    nbrs_min = knn_intersection_gpu(
        query_embs, corpus_embs, k=max_k, dev=dev, corpus_chunk=search_corpus_chunk
    )
    qps_min = len(query_embs) / (time.time() - t0)
    ids_only = [ids for ids, _ in nbrs_min]

    print(f"\n--- Stage 1: top-{max_k} by intersection on 512-D embeddings ---")
    rec = eval_recall_dict(gt, ids_only, QUERY_START, max_k)
    results["intersection_min_no_rerank"] = {**rec, "qps": qps_min}
    for k, r in rec.items():
        print(f"  R@{k:<4} = {r:.4f}")
    print(f"  QPS ≈ {qps_min:.1f}")

    for k in candidate_ks:
        print(f"\n--- Stage 2: top-{k} candidates + raw WJ ratio rerank ---")
        cand_ids = [ids[:k] for ids, _ in nbrs_min]
        t0 = time.time()
        rr_ids = rerank_wj_gpu(
            query_qt, cand_ids, corpus_qt, corpus_sums, dev, rerank_batch_size
        )
        qps = len(query_embs) / (time.time() - t0)
        rec_rr = eval_recall_dict(gt, rr_ids, QUERY_START, k)
        results[f"k{k}_raw_wj_ratio_rerank"] = {**rec_rr, "qps": qps, "k": k}
        for rk, rv in rec_rr.items():
            print(f"  R@{rk:<4} = {rv:.4f}")
        print(f"  QPS ≈ {qps:.1f}")

    payload = {
        "sports_50k_zeroshot": results,
        "_meta": {
            "train_dataset": "parks_10k",
            "eval_dataset": "sports_50k",
            "ckpt": CKPT_PATH,
            "enc": ENC_DIR,
            "gt": GT_DIR,
            "sports_vocab": SPORTS_VOCAB,
            "padded_to": PARKS_IN_DIM,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved {OUT_PATH}")
    print("\nReference (parks 10k, same model):")
    print("  intersection-min scratch: R@10 ~0.65 no rerank, ~0.99 K1000+rerank")


if __name__ == "__main__":
    main()
