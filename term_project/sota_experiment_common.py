import gc
import os
import pickle
import time
from pathlib import Path

import numpy as np


QUERY_START_10K = 8000
QUERY_START_FULL = 187019


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
    else:
        raise ValueError(f"unknown dataset: {name}")

    corpus_qt = qt[:query_start]
    query_qt = qt[query_start:]
    corpus_sums = corpus_qt.sum(axis=1).astype(np.float32)
    print(f"dataset={name} | qt={qt.shape} | corpus={corpus_qt.shape} | queries={query_qt.shape}")
    return qt, gt, query_start, corpus_qt, query_qt, corpus_sums


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


def cleanup():
    gc.collect()
