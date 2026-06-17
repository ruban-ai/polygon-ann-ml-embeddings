#!/usr/bin/env python3
"""80/20 split of 47K query polygons (GT keys in gt_lookup_full.pkl).

Train/eval use existing full-corpus GT — no regeneration.
Verifies area balance between train and eval query subsets.
"""
import argparse
import math
import os
import pickle
import random
import sys

import numpy as np

QUERY_START = 187019
SPLIT_PATH = "/tmp/query_split_80_20.pkl"
AREA_CACHE = "/tmp/polygon_areas_233773.npz"
WKT_PATH = "/raid/ssEncodingData/polygonalData/osm_new/parks"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compare_polygon_areas import load_areas, print_stats  # noqa: E402


def load_query_areas(wkt_path: str, total: int, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        areas = data["areas_m2"]
        if areas.size >= total:
            print(f"areas from cache {cache_path} n={areas.size:,}", flush=True)
            return areas[:total]
    print(f"loading {total:,} polygon areas from WKT (one-time, cached to {cache_path})...", flush=True)
    areas = load_areas(wkt_path, total)
    np.savez_compressed(cache_path, areas_m2=areas)
    return areas


def ecdf_max_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Max |F_a(x) - F_b(x)| on pooled sorted unique log-areas."""
    log_a = np.log10(np.maximum(a, 1.0))
    log_b = np.log10(np.maximum(b, 1.0))
    grid = np.unique(np.concatenate([log_a, log_b]))
    fa = np.searchsorted(np.sort(log_a), grid, side="right") / log_a.size
    fb = np.searchsorted(np.sort(log_b), grid, side="right") / log_b.size
    return float(np.max(np.abs(fa - fb)))


def decade_table(train: np.ndarray, eval_: np.ndarray) -> None:
    print("\nFraction per log10 decade (% of each split):")
    print(f"  {'decade':>14s}  {'train80':>8s}  {'eval20':>8s}  {'gap':>8s}")
    for d in range(-1, 12):
        lo, hi = d, d + 1
        t_pct = 100 * ((np.log10(train) >= lo) & (np.log10(train) < hi)).sum() / train.size
        e_pct = 100 * ((np.log10(eval_) >= lo) & (np.log10(eval_) < hi)).sum() / eval_.size
        if t_pct < 0.005 and e_pct < 0.005:
            continue
        print(f"  10^{d:<2d}–10^{d+1:<2d} m²  {t_pct:7.2f}%  {e_pct:7.2f}%  {e_pct - t_pct:+7.2f}pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/tmp/gt_lookup_full.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--out", default=SPLIT_PATH)
    ap.add_argument("--wkt-path", default=WKT_PATH)
    ap.add_argument("--skip-area", action="store_true")
    args = ap.parse_args()

    with open(args.gt, "rb") as f:
        gt = pickle.load(f)
    qids = sorted(gt.keys())
    assert min(qids) >= QUERY_START, f"unexpected min qid {min(qids)}"
    print(f"GT queries with labels: {len(qids):,} (ids {min(qids)}..{max(qids)})", flush=True)

    rng = random.Random(args.seed)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * args.train_frac)
    train_qids = sorted(shuffled[:n_train])
    eval_qids = sorted(shuffled[n_train:])
    print(f"split seed={args.seed} train={len(train_qids):,} eval={len(eval_qids):,}", flush=True)

    meta = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "query_start": QUERY_START,
        "corpus_size": QUERY_START,
        "n_gt_queries": len(qids),
        "train_qids": train_qids,
        "eval_qids": eval_qids,
        "note": "Train on train_qids only; eval on eval_qids vs full 187K corpus using original GT.",
    }

    if not args.skip_area:
        total = max(qids) + 1
        areas = load_query_areas(args.wkt_path, total, AREA_CACHE)
        train_areas = areas[train_qids]
        eval_areas = areas[eval_qids]
        all_query_areas = areas[qids]
        print("\nArea summary (query polygons):")
        print_stats("all_queries", all_query_areas)
        print_stats("train80", train_areas)
        print_stats("eval20", eval_areas)
        decade_table(train_areas, eval_areas)
        gap = ecdf_max_gap(train_areas, eval_areas)
        ref_gap = ecdf_max_gap(areas[:QUERY_START], all_query_areas)
        print(f"\nECDF max gap train80 vs eval20: {100*gap:.2f}pp")
        print(f"ECDF max gap corpus vs all_queries (reference): {100*ref_gap:.2f}pp")
        meta["area_ecdf_gap_train_eval_pp"] = 100 * gap
        meta["area_ecdf_gap_corpus_queries_pp"] = 100 * ref_gap
        if gap > 0.02:
            print("WARNING: train/eval area gap >2pp — consider re-seeding.", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(meta, f)
    print(f"saved split → {args.out}", flush=True)


if __name__ == "__main__":
    main()
