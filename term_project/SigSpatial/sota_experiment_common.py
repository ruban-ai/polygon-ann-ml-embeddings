import gc
import os
import pickle
import time
from pathlib import Path

import numpy as np


QUERY_START_10K = 8000
QUERY_START_FULL = 187019
QUERY_START_QUERIES_ONLY = 37403  # 80% of 46754 query polygons


def load_dataset(name):
    if name == "10k":
        qt = np.load("/tmp/qt_10k.npy")
        with open("/tmp/gt_lookup_10k.pkl", "rb") as f:
            gt = pickle.load(f)
        query_start = QUERY_START_10K
    elif name == "full":
        qt = np.load("/tmp/qtree_vectors_full.npy")
        with open("/tmp/gt_lookup_full.pkl", "rb") as f:
            gt = pickle.load(f)
        query_start = QUERY_START_FULL
    elif name == "queries_only":
        qt = np.load("/tmp/qt_queries_only.npy")
        with open("/tmp/gt_lookup_queries_only.pkl", "rb") as f:
            gt = pickle.load(f)
        meta_path = Path("/tmp/queries_only_meta.pkl")
        query_start = (
            pickle.load(open(meta_path, "rb"))["query_start"]
            if meta_path.exists()
            else QUERY_START_QUERIES_ONLY
        )
    else:
        raise ValueError(f"unknown dataset: {name}")

    corpus_qt = qt[:query_start]
    query_qt = qt[query_start:]
    corpus_sums = corpus_qt.sum(axis=1).astype(np.float32)
    print(f"dataset={name} | qt={qt.shape} | corpus={corpus_qt.shape} | queries={query_qt.shape}")
    return qt, gt, query_start, corpus_qt, query_qt, corpus_sums


def load_dataset_normalized(name):
    """Like load_dataset but also returns qt_norm (L1-simplex normalized qt).
    qt_norm is cached to /tmp/qt_norm_{name}.npy after first computation."""
    qt, gt, query_start, corpus_qt, query_qt, corpus_sums = load_dataset(name)
    cache_path = f"/raid/ruban/hpmlproj/term_project/SigSpatial/qt_norm_{name}.npy"
    if Path(cache_path).exists():
        t0 = time.time()
        qt_norm = np.load(cache_path)
        print(f"qt_norm loaded from cache {qt_norm.shape} in {time.time()-t0:.1f}s")
    else:
        print("Computing qt_norm (first run — caching to /tmp)...", flush=True)
        t0 = time.time()
        qt_norm = l1_simplex(qt.copy())
        np.save(cache_path, qt_norm)
        print(f"qt_norm computed+cached {qt_norm.shape} ({qt_norm.nbytes/1024**3:.2f} GB) in {time.time()-t0:.1f}s")
    return qt, gt, query_start, corpus_qt, query_qt, corpus_sums, qt_norm


def load_10k_train8k_normalized():
    """10K benchmark: train on queries 0–7999 (gt_lookup_train_8k), eval on 8000–9999 (gt_lookup_10k).
    Uses first 10K rows of 18220-d qtree_vectors_full (same polygons as cross_eval_10k)."""
    cache_path = "/tmp/qt_norm_10k_18220.npy"
    if Path(cache_path).exists():
        t0 = time.time()
        qt_norm = np.load(cache_path)
        print(f"qt_norm loaded from cache {qt_norm.shape} in {time.time()-t0:.1f}s")
        qt = np.array(np.load("/tmp/qtree_vectors_full.npy", mmap_mode="r")[:10000], dtype=np.float32)
    else:
        print("Computing qt_norm_10k_18220 (first run — caching to /tmp)...", flush=True)
        t0 = time.time()
        qt = np.array(np.load("/tmp/qtree_vectors_full.npy", mmap_mode="r")[:10000], dtype=np.float32)
        qt_norm = l1_simplex(qt)
        np.save(cache_path, qt_norm)
        print(f"qt_norm computed+cached {qt_norm.shape} in {time.time()-t0:.1f}s")
    with open("/tmp/gt_lookup_train_8k.pkl", "rb") as f:
        train_gt = pickle.load(f)
    with open("/tmp/gt_lookup_10k.pkl", "rb") as f:
        eval_gt = pickle.load(f)
    qs = QUERY_START_10K
    corpus_qt = qt[:qs]
    query_qt = qt[qs:]
    corpus_sums = corpus_qt.sum(axis=1).astype(np.float32)
    print(
        f"dataset=10k-train8k | qt={qt.shape} | corpus={corpus_qt.shape} | queries={query_qt.shape} "
        f"| train_gt={len(train_gt)} eval_gt={len(eval_gt)}"
    )
    return qt, train_gt, eval_gt, qs, corpus_qt, query_qt, corpus_sums, qt_norm


def l1_simplex(x, eps=1e-10):
    x = np.maximum(x, 0).astype(np.float32, copy=False)
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def shifted_l1_simplex(x, eps=1e-10):
    x = x.astype(np.float32, copy=False)
    mins = x.min(axis=1, keepdims=True)
    x = x - np.minimum(mins, 0)
    return x / np.maximum(x.sum(axis=1, keepdims=True), eps)


def _ground_truth_for_query(gt_lookup, query_start_id, query_offset, k):
    """Match bulkTestingWeighted.py: select the first K GT items per query."""
    if isinstance(gt_lookup, dict):
        qid = query_start_id + query_offset
        return list(gt_lookup.get(qid, []))[:k]
    return list(gt_lookup[query_offset])[:k]


def compute_recall(correct_set, retrieved_set):
    if not correct_set:
        return 0.0
    return float(len(correct_set.intersection(retrieved_set))) / len(correct_set)


def compute_precision(correct_set, retrieved_set):
    if not retrieved_set:
        return 0.0
    return float(len(correct_set.intersection(retrieved_set))) / len(retrieved_set)


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(gt_lookup, nbrs, query_start_id, k):
    return buddhi_metrics_at_k(gt_lookup, nbrs, query_start_id, k)["recall"]


def buddhi_metrics_at_k(gt_lookup, nbrs, query_start_id, k):
    """
    Metric logic from Buddhi/bulkTestingWeighted.py.

    For each query, select the first K ground-truth ids, compare against the
    returned K-neighbor set, and average Recall/Precision/F1 over nonempty
    denominators. full_recall keeps zero-GT queries in the denominator.
    """
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    recall_count = 0
    precision_count = 0
    f1_count = 0
    query_qty = len(nbrs)

    for i in range(query_qty):
        ids = nbrs[i][0] if isinstance(nbrs[i], tuple) else nbrs[i]
        correct_set = set(_ground_truth_for_query(gt_lookup, query_start_id, i, k))
        retrieved_set = set(list(ids)[:k])

        recall = 0.0
        precision = 0.0
        f1 = 0.0

        if correct_set:
            recall = compute_recall(correct_set, retrieved_set)
            recall_count += 1
        if retrieved_set:
            precision = compute_precision(correct_set, retrieved_set)
            precision_count += 1
        if precision + recall > 0:
            f1 = compute_f1(precision, recall)
            f1_count += 1

        total_recall += recall
        total_precision += precision
        total_f1 += f1

    avg_recall = total_recall / recall_count if recall_count else 0.0
    avg_precision = total_precision / precision_count if precision_count else 0.0
    avg_f1 = total_f1 / f1_count if f1_count else 0.0
    full_recall = total_recall / query_qty if query_qty else 0.0

    return {
        "full_recall": full_recall,
        "recall": avg_recall,
        "precision": avg_precision,
        "f1": avg_f1,
    }


def eval_recall(gt_lookup, nbrs, query_start_id, max_k):
    metrics = {}
    for k in (10, 50, 100, 500):
        if k > max_k:
            continue
        m = buddhi_metrics_at_k(gt_lookup, nbrs, query_start_id, k)
        metrics[k] = m["recall"]
        metrics[f"full_recall@{k}"] = m["full_recall"]
        metrics[f"precision@{k}"] = m["precision"]
        metrics[f"f1@{k}"] = m["f1"]
    return metrics


def eval_recall_by_qids(gt_lookup, nbrs, query_ids, max_k):
    """Like eval_recall but nbrs[i] corresponds to query_ids[i] (held-out eval split)."""
    metrics = {}
    for k in (10, 50, 100, 500):
        if k > max_k:
            continue
        total_recall = total_precision = total_f1 = 0.0
        recall_count = precision_count = f1_count = 0
        for i, qid in enumerate(query_ids):
            ids = nbrs[i][0] if isinstance(nbrs[i], tuple) else nbrs[i]
            correct_set = set(gt_lookup.get(qid, [])[:k])
            retrieved_set = set(list(ids)[:k])
            if correct_set:
                total_recall += compute_recall(correct_set, retrieved_set)
                recall_count += 1
            if retrieved_set:
                total_precision += compute_precision(correct_set, retrieved_set)
                precision_count += 1
            if correct_set and retrieved_set:
                p = compute_precision(correct_set, retrieved_set)
                r = compute_recall(correct_set, retrieved_set)
                total_f1 += compute_f1(p, r)
                f1_count += 1
        n = len(query_ids)
        metrics[k] = total_recall / recall_count if recall_count else 0.0
        metrics[f"full_recall@{k}"] = total_recall / n if n else 0.0
        metrics[f"precision@{k}"] = total_precision / precision_count if precision_count else 0.0
        metrics[f"f1@{k}"] = total_f1 / f1_count if f1_count else 0.0
    return metrics


def nmslib_neighbors(corpus_embs, query_embs, space="WeightedJaccard", k=500, threads=32,
                    index_params=None, query_params=None):
    """
    Index/query logic from Buddhi/indexConstructWeighted.py and
    Buddhi/bulkTestingWeighted.py:
    HNSW + WeightedJaccard, addDataPointBatch, M=20, efConstruction=200,
    post=1, indexThreadQty=threads, and efSearch=200 by default.
    """
    import nmslib

    index_params = index_params or {
        "M": 20,
        "indexThreadQty": threads,
        "efConstruction": 200,
        "post": 1,
    }
    query_params = query_params or {"efSearch": 200}

    idx = nmslib.init(method="hnsw", space=space)
    idx.addDataPointBatch(np.asarray(corpus_embs, dtype=np.float32))
    t0 = time.time()
    idx.createIndex(index_params, print_progress=True)
    build_s = time.time() - t0
    idx.setQueryTimeParams(query_params)

    t0 = time.time()
    raw = idx.knnQueryBatch(query_embs, k=k, num_threads=threads)
    query_s = time.time() - t0
    qps = len(query_embs) / max(query_s, 1e-9)
    nbrs = [ids.tolist() if hasattr(ids, "tolist") else list(ids) for ids, _ in raw]
    return nbrs, {
        "build_s": build_s,
        "qps": qps,
        "query_s": query_s,
        "per_query_s": query_s / max(len(query_embs), 1),
        "per_query_thread_adjusted_s": threads * query_s / max(len(query_embs), 1),
        "M": index_params.get("M"),
        "efConstruction": index_params.get("efConstruction"),
        "efSearch": query_params.get("efSearch"),
        "post": index_params.get("post"),
        "indexThreadQty": index_params.get("indexThreadQty"),
    }


def rerank_raw_wj_numpy(query_qt, candidate_ids, corpus_qt, corpus_sums, top_k=500, batch_size=16):
    reranked = []
    for start in range(0, len(candidate_ids), batch_size):
        batch_ids = candidate_ids[start:start + batch_size]
        for offset, ids in enumerate(batch_ids):
            ids = np.asarray(ids, dtype=np.int64)
            if len(ids) == 0:
                reranked.append([])
                continue
            q = query_qt[start + offset]
            c = corpus_qt[ids]
            mins = np.minimum(c, q).sum(axis=1)
            maxs = corpus_sums[ids] + q.sum() - mins
            sims = mins / np.maximum(maxs, 1e-10)
            order = np.argsort(-sims)[:top_k]
            reranked.append(ids[order].tolist())
    return reranked


_rerank_corpus_cache = {}


def preload_rerank_corpus(corpus_qt, corpus_sums):
    """Load corpus onto GPU once before a batch of rerank calls. Call release_rerank_corpus() after."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
    except ImportError:
        return
    dev = torch.device("cuda:0")
    _rerank_corpus_cache["corpus_t"] = torch.from_numpy(
        np.ascontiguousarray(corpus_qt, dtype=np.float32)).to(dev)
    _rerank_corpus_cache["corpus_sums_t"] = torch.from_numpy(
        np.ascontiguousarray(corpus_sums, dtype=np.float32)).to(dev)
    print(f"Corpus pre-loaded to GPU: {corpus_qt.nbytes/1024**3:.2f} GB")


def release_rerank_corpus():
    """Free pre-loaded corpus from GPU."""
    _rerank_corpus_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def rerank_wj_gpu(query_qt, candidate_ids, corpus_qt, corpus_sums, top_k=500, batch_size=16):
    """GPU-accelerated WJ rerank. Uses pre-loaded corpus if available (call preload_rerank_corpus first).
    Falls back to numpy if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")
    except Exception:
        return rerank_raw_wj_numpy(query_qt, candidate_ids, corpus_qt, corpus_sums, top_k)

    dev = torch.device("cuda:0")
    owns_corpus = "corpus_t" not in _rerank_corpus_cache
    if owns_corpus:
        corpus_t = torch.from_numpy(np.ascontiguousarray(corpus_qt, dtype=np.float32)).to(dev)
        corpus_sums_t = torch.from_numpy(np.ascontiguousarray(corpus_sums, dtype=np.float32)).to(dev)
    else:
        corpus_t = _rerank_corpus_cache["corpus_t"]
        corpus_sums_t = _rerank_corpus_cache["corpus_sums_t"]
    reranked = []

    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start:start + batch_size]
        groups = {}
        for offset, ids in enumerate(batch):
            ids_arr = np.asarray(ids, dtype=np.int64)
            groups.setdefault(len(ids_arr), []).append((offset, ids_arr))

        slot = [None] * len(batch)
        for cand_len, items in groups.items():
            if cand_len == 0:
                for offset, _ in items:
                    slot[offset] = []
                continue
            ids_np = np.stack([ids for _, ids in items])
            q_np = np.ascontiguousarray(
                query_qt[[start + off for off, _ in items]], dtype=np.float32)
            ids_t = torch.from_numpy(ids_np).to(dev)
            q_t = torch.from_numpy(q_np).to(dev)
            c_t = corpus_t[ids_t]                                      # (B, K, D)
            mins = torch.minimum(q_t.unsqueeze(1), c_t).sum(2)        # (B, K)
            maxs = (q_t.sum(1, keepdim=True) + corpus_sums_t[ids_t] - mins).clamp_min(1e-10)
            order = torch.argsort(mins / maxs, dim=1, descending=True)[:, :top_k].cpu().numpy()
            for (offset, ids), row in zip(items, order):
                slot[offset] = ids[row].tolist()
        reranked.extend(slot)

    if owns_corpus:
        del corpus_t, corpus_sums_t
        torch.cuda.empty_cache()
    return reranked


def save_result(path, dataset_name, method_name, metrics, meta=None):
    path = Path(path)
    payload = {}
    if path.exists():
        with open(path, "rb") as f:
            payload = pickle.load(f)
    payload.setdefault(dataset_name, {})[method_name] = metrics
    payload["_meta"] = {
        **payload.get("_meta", {}),
        method_name: {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(meta or {}),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {method_name} -> {path}")


def build_gt_cache(gt, n_items, query_start, dataset_name):
    """Build (or load from disk) row-sorted GT array for FN masking.

    Rows are sorted ascending so GPU searchsorted works directly.
    -1 padding sorts to the front (smallest value) — handled by callers.

    Returns
    -------
    gt_stacked : np.int32 (n_queries, max_K), each row sorted, padded with -1.
                 Row i → query ID (query_start + i).
    """
    cache_path = f"/tmp/gt_stacked_{dataset_name}.npy"
    n_queries = n_items - query_start

    if Path(cache_path).exists():
        t0 = time.time()
        gt_stacked = np.load(cache_path)
        print(f"gt_stacked loaded from cache {gt_stacked.shape} in {time.time()-t0:.1f}s")
    else:
        print("Building gt_stacked (first run — caches to /tmp)...")
        t0 = time.time()
        max_K = max((len(gt.get(query_start + i, [])) for i in range(n_queries)), default=0)
        gt_stacked = np.full((n_queries, max_K), -1, dtype=np.int32)
        for i in range(n_queries):
            nbrs = gt.get(query_start + i, [])
            if nbrs:
                gt_stacked[i, :len(nbrs)] = nbrs
        gt_stacked.sort(axis=1)          # sort in-place; -1 padding moves to front
        np.save(cache_path, gt_stacked)
        print(f"Built+sorted+cached {gt_stacked.shape} "
              f"({gt_stacked.nbytes/1024**3:.2f} GB) in {time.time()-t0:.1f}s → {cache_path}")

    return gt_stacked


def build_gt_gpu(gt_stacked, device):
    """Move sorted GT array to GPU for fast per-step searchsorted.
    Call once after build_gt_cache; store result in training cell.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch required")
    t0 = time.time()
    gt_gpu = torch.from_numpy(gt_stacked).to(device)
    print(f"gt_gpu on {device}: {tuple(gt_gpu.shape)} "
          f"({gt_gpu.nbytes/1024**3:.2f} GB) in {time.time()-t0:.1f}s")
    return gt_gpu


def build_fn_mask(a_ids, p_ids, gt_gpu, query_start):
    """GPU-accelerated (B,B) FN mask via searchsorted — ~1.5ms vs ~440ms on CPU.

    a_ids  : query IDs  >= query_start, shape (B,) — CPU tensor or ndarray
    p_ids  : corpus IDs <  query_start, shape (B,) — CPU tensor or ndarray
    gt_gpu : (n_queries, max_K) int32 GPU tensor, each row sorted ascending,
             -1-padded (padding is at front after sort). Lives on vecs_device.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("torch required")

    dev = gt_gpu.device
    B   = len(a_ids)
    K   = gt_gpu.shape[1]

    a_idx = (a_ids if isinstance(a_ids, torch.Tensor) else torch.as_tensor(a_ids))
    p_t   = (p_ids if isinstance(p_ids, torch.Tensor) else torch.as_tensor(p_ids))

    gt_rows = gt_gpu[(a_idx - query_start).to(dev)]        # (B, K) sorted rows on GPU
    p_exp   = p_t.to(dev).unsqueeze(0).expand(B, -1)      # (B, B)

    # For each anchor row i, binary-search all B positive IDs
    pos  = torch.searchsorted(gt_rows, p_exp).clamp(0, K - 1)   # (B, B)
    mask = gt_rows.gather(1, pos) == p_exp                        # (B, B) bool
    mask.fill_diagonal_(False)
    return mask.cpu()


def cleanup():
    gc.collect()
