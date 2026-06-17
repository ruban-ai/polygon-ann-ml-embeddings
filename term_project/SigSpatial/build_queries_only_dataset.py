#!/usr/bin/env python3
"""Build closed-world dataset: corpus + queries = query-polygon pool only (~47K).

Original full GT neighbors live in the 187K corpus (ids < 187019). Filtering
those lists to the query set yields ~0 neighbors — we must recompute GT with
exact raw WJ within the pool (80/20 split, same ratio as full).

Writes:
  /tmp/qt_queries_only.npy
  /tmp/qt_norm_queries_only.npy
  /tmp/gt_lookup_queries_only.pkl   # {qid: [corpus neighbor ids]}
  /tmp/queries_only_meta.pkl        # query_start, counts, gt_k
"""
import pickle
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, "/raid/ruban/hpmlproj/term_project/SigSpatial")
from sota_experiment_common import l1_simplex

FULL_QS = 187019
N_QUERY_POLYS = 46754
QUERY_FRAC = 0.8
GT_TOP_K = 10_000
QUERY_BATCH = 8
CORPUS_CHUNK = 2048
DEV = torch.device("cuda:0")


@torch.no_grad()
def wj_topk(q, corpus, topk):
    """q: (B,D) -> topk indices in corpus (N,D) without O(N*D) materialization."""
    best_sim = torch.full((q.shape[0], topk), -1.0, device=DEV)
    best_idx = torch.zeros((q.shape[0], topk), dtype=torch.long, device=DEV)
    for c0 in range(0, corpus.shape[0], CORPUS_CHUNK):
        cc = corpus[c0:c0 + CORPUS_CHUNK]
        mins = torch.minimum(q[:, None, :], cc[None, :, :]).sum(2)
        maxs = (q.sum(1, keepdim=True) + cc.sum(1, keepdim=True)[None, :] - mins).clamp(min=1e-10)
        sim = mins / maxs
        idx_local = torch.arange(c0, c0 + cc.shape[0], device=DEV)
        merged_sim = torch.cat([best_sim, sim], dim=1)
        merged_idx = torch.cat([best_idx, idx_local.expand(q.shape[0], -1)], dim=1)
        pick = torch.topk(merged_sim, min(topk, merged_sim.shape[1]), dim=1)
        best_sim = pick.values
        best_idx = torch.gather(merged_idx, 1, pick.indices)
    return best_idx.cpu().numpy()


def main():
    print("Loading full qt slice (query polygons only)...", flush=True)
    t0 = time.time()
    qt_full = np.load("/tmp/qtree_vectors_full.npy", mmap_mode="r")
    qt = np.array(qt_full[FULL_QS:FULL_QS + N_QUERY_POLYS], dtype=np.float32)
    del qt_full
    qtn = l1_simplex(qt.copy())
    query_start = int(N_QUERY_POLYS * QUERY_FRAC)
    corpus = qtn[:query_start]
    queries = qtn[query_start:]
    print(
        f"pool={len(qtn):,} corpus={len(corpus):,} queries={len(queries):,} "
        f"split={QUERY_FRAC:.0%}/{1-QUERY_FRAC:.0%} ({time.time()-t0:.1f}s)",
        flush=True,
    )

    corpus_t = torch.from_numpy(np.ascontiguousarray(corpus)).to(DEV)
    gt = {}
    print(f"Computing exact WJ GT (top {GT_TOP_K} per query)...", flush=True)
    for i in tqdm(range(0, len(queries), BATCH), ncols=100):
        qb = torch.from_numpy(np.ascontiguousarray(queries[i:i + BATCH])).to(DEV)
        sim = wj_scores(qb, corpus_t)
        topk = min(GT_TOP_K, corpus_t.shape[0])
        idx = torch.topk(sim, topk, dim=1).indices.cpu().numpy()
        for j, row in enumerate(idx):
            qid = query_start + i + j
            gt[qid] = row.tolist()

    lens = [len(v) for v in gt.values()]
    print(
        f"GT built: {len(gt):,} queries | neighbors/query: "
        f"mean={np.mean(lens):.0f} median={np.median(lens):.0f} min={min(lens)}",
        flush=True,
    )

    np.save("/tmp/qt_queries_only.npy", qt)
    np.save("/tmp/qt_norm_queries_only.npy", qtn)
    with open("/tmp/gt_lookup_queries_only.pkl", "wb") as f:
        pickle.dump(gt, f, protocol=4)
    meta = {
        "query_start": query_start,
        "n_total": N_QUERY_POLYS,
        "n_corpus": query_start,
        "n_queries": N_QUERY_POLYS - query_start,
        "gt_top_k": GT_TOP_K,
        "source": "query polygon pool; exact raw WJ vs corpus partition",
    }
    with open("/tmp/queries_only_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("Saved /tmp/qt_queries_only.npy, qt_norm_queries_only.npy, gt_lookup_queries_only.pkl", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
