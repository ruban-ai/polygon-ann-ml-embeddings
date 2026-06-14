#!/usr/bin/env python3
"""
Sweep nmslib efSearch for a saved checkpoint — find best recall/QPS tradeoff.

Usage:
  python eval_ef_sweep.py --ckpt /tmp/best_sota_triplet_autoencoder_wj_512_full.pt
  python eval_ef_sweep.py --ckpt /tmp/best_triplet_ae_mined_wj_512_full.pt --efs 50 100 200 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sota_experiment_common import eval_recall, load_dataset_normalized, nmslib_neighbors

DATASET = "full"
OUT_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
THREADS = 40
QUERY_START_FULL = 187019


class TripletEncoder(nn.Module):
    """nb28 encoder-only (for loading old ckpts)."""

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

    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z / z.sum(dim=1, keepdim=True).clamp(min=1e-10)


class TripletAE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.encoder = TripletEncoder(in_dim, out_dim).encoder
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, in_dim, bias=False),
        )

    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z / z.sum(dim=1, keepdim=True).clamp(min=1e-10)


@torch.no_grad()
def embed(model, qt_norm, batch_size=512):
    model.eval()
    chunks = []
    for s in range(0, len(qt_norm), batch_size):
        x = torch.tensor(qt_norm[s : s + batch_size], dtype=torch.float32, device=DEVICE)
        chunks.append(model.encode(x).cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def load_model(in_dim: int, ckpt: str) -> nn.Module:
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    has_decoder = any(k.startswith("decoder.") for k in state)
    model = TripletAE(in_dim, OUT_DIM) if has_decoder else TripletEncoder(in_dim, OUT_DIM)
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--efs", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    parser.add_argument("--k", type=int, default=500)
    args = parser.parse_args()

    _, gt, query_start, _, query_qt, _, qt_norm = load_dataset_normalized(DATASET)
    model = load_model(qt_norm.shape[1], args.ckpt)
    embs = embed(model, qt_norm)
    corpus_embs = embs[:query_start]
    query_embs = embs[query_start:]

    print(f"ckpt={args.ckpt} | queries={len(query_embs)} | corpus={len(corpus_embs)}")
    print(f"{'efSearch':>10} {'R@50':>8} {'QPS':>10}")
    print("-" * 32)
    for ef in args.efs:
        nbrs, info = nmslib_neighbors(
            corpus_embs,
            query_embs,
            space="WeightedJaccard",
            k=args.k,
            threads=THREADS,
            query_params={"efSearch": ef},
        )
        m = eval_recall(gt, nbrs, query_start, args.k)
        print(f"{ef:>10} {m.get(50, 0):>8.4f} {info['qps']:>10.1f}")


if __name__ == "__main__":
    main()
