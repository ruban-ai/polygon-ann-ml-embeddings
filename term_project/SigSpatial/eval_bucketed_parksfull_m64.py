#!/usr/bin/env python3
"""FULL Parks version (2026-10-03; 10K/50K are being dropped): identical method to
eval_bucketed_50k_listwise.py (area = sum of the raw query vector, complexity = nnz, quartiles over the
held-out queries; RP = same min-shift random projection), on 187,019 corpus / 46,754 queries with the
full-scale listwise checkpoint best_parksfull_m64_listwise.pt.
Original docstring: Per-polygon-class recall breakdown (Review 2 ask), re-run with the listwise
checkpoint instead of MSE -- listwise has since replaced MSE as the paper's primary
method (see LOSS_EXPLORATION_LOG.md), so this table needs to reflect the same
method the rest of the paper now reports. No new training: loads the existing
best_parksfull_m64_listwise.pt checkpoint (from run_full187k_listwise.py).

Launch: CUDA_VISIBLE_DEVICES=0 python eval_bucketed_parksfull.py
"""
import sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import nmslib_neighbors, compute_recall

DEV = torch.device('cuda:0')
QS = 187019
EMB = 4096
D_OP = 256
EF = 200
THREADS = 64
LOG = '/raid/ruban/hpmlproj/term_project/SigSpatial/logs_20261009/eval_bucketed_parksfull_m64.log'


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def norm(x):
    s = x.sum(1, keepdims=True) if isinstance(x, np.ndarray) else x.sum(1, keepdim=True)
    return x / np.clip(s, 1e-10, None) if isinstance(x, np.ndarray) else x / s.clamp(min=1e-10)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.encoder = nn.Sequential(
            nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x):
        return F.relu(s.encoder(x))


def per_query_recall(gt, nbrs, query_start_id, k):
    out = np.full(len(nbrs), np.nan)
    for i, ids in enumerate(nbrs):
        qid = query_start_id + i
        correct = set(gt.get(qid, [])[:k])
        if not correct:
            continue
        retrieved = set(list(ids)[:k])
        out[i] = compute_recall(correct, retrieved)
    return out


def bucket_report(name, values_by_bucket_r50, values_by_bucket_r500):
    log(f"\n-- {name} --")
    log(f"{'bucket':8s} {'n':>6s} {'R@50':>8s} {'R@500':>8s}")
    for b in range(4):
        r50 = values_by_bucket_r50[b]
        r500 = values_by_bucket_r500[b]
        log(f"Q{b+1:<7d} {len(r50):6d} {np.nanmean(r50):8.4f} {np.nanmean(r500):8.4f}")


def main():
    t0 = time.time()
    log("loading qtree_vectors_full.npy + gt_lookup_full.pkl ...")
    qt = np.load('/raid/ruban/hpmlproj/term_project/SigSpatial/qtree_vectors_full.npy')
    import pickle
    with open('/raid/ruban/hpmlproj/term_project/SigSpatial/gt_lookup_full.pkl', 'rb') as f:
        gt = pickle.load(f)
    log(f"qt {qt.shape} loaded in {time.time()-t0:.1f}s")

    qtn = qt / np.clip(qt.sum(1, keepdims=True), 1e-10, None)
    corpus_raw, query_raw = qt[:QS], qt[QS:]
    N_Q = query_raw.shape[0]
    log(f"corpus={corpus_raw.shape[0]} queries={N_Q}")

    area = query_raw.sum(1)
    complexity = (query_raw > 0).sum(1)
    area_bucket = np.digitize(area, np.quantile(area, [0.25, 0.5, 0.75]))
    cplx_bucket = np.digitize(complexity, np.quantile(complexity, [0.25, 0.5, 0.75]))
    log(f"area range [{area.min():.4g}, {area.max():.4g}], "
        f"complexity (nnz) range [{complexity.min()}, {complexity.max()}]")

    log("loading WJ-listwise checkpoint (256-d prefix) ...")
    IN = qt.shape[1]
    m = Net(IN).to(DEV)
    m.load_state_dict(torch.load('/raid/ruban/hpmlproj/term_project/SigSpatial/best_parksfull_m64_listwise.pt', map_location=DEV, weights_only=True))
    m.eval()
    with torch.no_grad():
        Z = torch.cat([m(torch.from_numpy(np.ascontiguousarray(qtn[i:i+1024], dtype=np.float32)).to(DEV))
                       for i in range(0, len(qtn), 1024)])
        emb_listwise = norm(Z[:, :D_OP]).cpu().numpy().astype(np.float32)
    log(f"listwise embeddings ready {emb_listwise.shape} ({time.time()-t0:.1f}s elapsed)")

    rng = np.random.RandomState(42)
    R = rng.randn(IN, D_OP).astype(np.float32) / np.sqrt(D_OP)
    proj = qtn @ R
    proj = proj - proj.min(1, keepdims=True)
    emb_rp = proj / np.clip(proj.sum(1, keepdims=True), 1e-10, None)
    log(f"random-projection embeddings ready {emb_rp.shape} ({time.time()-t0:.1f}s elapsed)")

    results = {}
    for tag, emb in [('WJ-listwise', emb_listwise), ('Rand.Proj.', emb_rp)]:
        log(f"building HNSW + querying for {tag} ...")
        ce, qe = emb[:QS], emb[QS:]
        nbrs, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=500,
                                       threads=THREADS, query_params={"efSearch": EF})
        log(f"{tag}: HNSW QPS={info['qps']:.0f}")
        r50 = per_query_recall(gt, nbrs, QS, 50)
        r500 = per_query_recall(gt, nbrs, QS, 500)
        results[tag] = (r50, r500)
        log(f"{tag} overall: R@50={np.nanmean(r50):.4f} R@500={np.nanmean(r500):.4f}")

    for tag, (r50, r500) in results.items():
        bucket_report(f"{tag} by AREA quartile (Q1=smallest .. Q4=largest)",
                       [r50[area_bucket == b] for b in range(4)],
                       [r500[area_bucket == b] for b in range(4)])
        bucket_report(f"{tag} by COMPLEXITY quartile (Q1=simplest .. Q4=most complex)",
                       [r50[cplx_bucket == b] for b in range(4)],
                       [r500[cplx_bucket == b] for b in range(4)])

    out = {
        'area_bucket': area_bucket.tolist(), 'cplx_bucket': cplx_bucket.tolist(),
        'results': {tag: {'r50': r50.tolist(), 'r500': r500.tolist()} for tag, (r50, r500) in results.items()},
    }
    with open('/raid/ruban/hpmlproj/term_project/SigSpatial/results_bucketed_parksfull_m64.json', 'w') as f:
        json.dump(out, f)
    log(f"done, saved results_bucketed_parksfull_m64.json, total {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
