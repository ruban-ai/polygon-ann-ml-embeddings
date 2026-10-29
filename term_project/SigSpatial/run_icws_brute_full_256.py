#!/usr/bin/env python3
"""ICWS (Improved Consistent Weighted Sampling) brute-force baseline at 50K --
FULL Parks version (2026-10-02, 10K/50K results are being dropped from the paper):
187,019 corpus / 46,754 held-out queries, Stage-1 only (no rerank, scope decision).
Identical algorithm and parameters to run_icws_brute_50k.py.

The original notebook (26_icws_weighted_minhash_512.ipynb) signing step is pure
NumPy with a Python loop over num_samples=512 -- at full 187K scale that alone
took 542.2 minutes (~9h). This reimplements signing as a batched GPU op
(vectorized over a chunk of samples at once via broadcasting) instead of one
NumPy op per sample; ranking (brute-force sketch collision, already broadcast-
vectorized and comparatively cheap: ~16,122s for 46,754 queries x 187,019
corpus at full scale) is kept as the original CPU/NumPy version, scaled down.

Launch: CUDA_VISIBLE_DEVICES=0 python run_icws_brute_full.py"""
import sys, time, csv, datetime
import numpy as np, torch
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (eval_recall, preload_rerank_corpus,
    release_rerank_corpus, rerank_wj_gpu)

DEV = torch.device('cuda:0')
QS = 187019  # Parks full: corpus=[0,187019), queries=[187019,233773)
NUM_SAMPLES = 256   # 2026-10-25: Buddhi asked for ICWS at the same size as our 256-d signature
SEED = 42
TOP_K = 500
SIGN_BATCH = 512
SAMPLE_CHUNK = 64
RANK_BATCH = 16
CAND_KS = [1000, 2000]
RB = 64
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"


@torch.no_grad()
def icws_signatures_gpu(x_np, num_samples, seed, sign_batch, sample_chunk, dev):
    rng = np.random.default_rng(seed)
    d = x_np.shape[1]
    r = torch.tensor(rng.gamma(2.0, 1.0, size=(num_samples, d)), dtype=torch.float32, device=dev)
    c = torch.tensor(rng.gamma(2.0, 1.0, size=(num_samples, d)), dtype=torch.float32, device=dev)
    beta = torch.tensor(rng.random(size=(num_samples, d)), dtype=torch.float32, device=dev)

    N = x_np.shape[0]
    sig_idx = np.empty((N, num_samples), dtype=np.int32)
    sig_t = np.empty((N, num_samples), dtype=np.int32)
    t0 = time.time()
    for start in range(0, N, sign_batch):
        xb = torch.tensor(np.maximum(x_np[start:start + sign_batch], 1e-30), dtype=torch.float32, device=dev)
        logx = torch.log(xb)
        b = xb.shape[0]
        idx_out = torch.empty((b, num_samples), dtype=torch.int32, device=dev)
        t_out = torch.empty((b, num_samples), dtype=torch.int32, device=dev)
        for s0 in range(0, num_samples, sample_chunk):
            s1 = min(s0 + sample_chunk, num_samples)
            rs, cs, betas = r[s0:s1], c[s0:s1], beta[s0:s1]
            t = torch.floor(logx.unsqueeze(1) / rs.unsqueeze(0) + betas.unsqueeze(0))  # (b,chunk,d)
            y = torch.exp(rs.unsqueeze(0) * (t - betas.unsqueeze(0)))
            a = cs.unsqueeze(0) / (y * torch.exp(rs.unsqueeze(0)))
            idx = torch.argmin(a, dim=2)  # (b,chunk)
            idx_out[:, s0:s1] = idx.to(torch.int32)
            t_sel = torch.gather(t, 2, idx.unsqueeze(2).long()).squeeze(2)
            t_out[:, s0:s1] = t_sel.to(torch.int32)
        sig_idx[start:start + b] = idx_out.cpu().numpy()
        sig_t[start:start + b] = t_out.cpu().numpy()
        if (start // sign_batch) % 10 == 0:
            print(f"signed {start + b:,}/{N:,} ({(time.time()-t0)/60:.1f}min elapsed)", flush=True)
    print(f"signature time={(time.time()-t0)/60:.1f} min", flush=True)
    return sig_idx, sig_t


def sketch_topk(query_idx, query_t, corpus_idx, corpus_t, k, batch_size):
    nbrs = []
    t0 = time.time()
    for start in range(0, len(query_idx), batch_size):
        qi = query_idx[start:start + batch_size]
        qt_ = query_t[start:start + batch_size]
        sim = ((qi[:, None, :] == corpus_idx[None, :, :]) &
               (qt_[:, None, :] == corpus_t[None, :, :])).mean(axis=2)
        kk = min(k, corpus_idx.shape[0])
        part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(qi))[:, None]
        order = np.argsort(-sim[rows, part], axis=1)
        top = part[rows, order]
        nbrs.extend([row.tolist() for row in top])
        if (start // batch_size) % 50 == 0:
            print(f"ranked {start + len(qi):,}/{len(query_idx):,} ({(time.time()-t0)/60:.1f}min elapsed)", flush=True)
    qps = len(query_idx) / max(time.time() - t0, 1e-9)
    return nbrs, qps


def main():
    global QS
    np.random.seed(SEED)
    import pickle
    qt = np.load('/raid/ruban/hpmlproj/term_project/SigSpatial/qtree_vectors_full.npy', mmap_mode='r')
    gt = pickle.load(open('/raid/ruban/hpmlproj/term_project/SigSpatial/gt_lookup_full.pkl', 'rb'))
    if '--smoke' in sys.argv:     # tiny plumbing check: 5K corpus, 300 queries remapped from the real split
        q0 = QS; QS = 5000
        qt = np.concatenate([np.asarray(qt[:QS]), np.asarray(qt[q0:q0 + 300])])
        gt = {QS + i: [n for n in gt.get(q0 + i, []) if n < QS] or [0] for i in range(300)}
    print(f"qt={qt.shape} corpus={QS} queries={qt.shape[0]-QS}", flush=True)

    sig_idx, sig_t = icws_signatures_gpu(qt, NUM_SAMPLES, SEED, SIGN_BATCH, SAMPLE_CHUNK, DEV)
    corpus_idx, query_idx = sig_idx[:QS], sig_idx[QS:]
    corpus_t, query_t = sig_t[:QS], sig_t[QS:]
    print(f"signature memory corpus={(corpus_idx.nbytes + corpus_t.nbytes)/1024**2:.1f} MB", flush=True)

    nbrs, qps = sketch_topk(query_idx, query_t, corpus_idx, corpus_t, k=TOP_K, batch_size=RANK_BATCH)
    metrics = eval_recall(gt, nbrs, QS, TOP_K)
    print(f"[icws-brute-parksfull base ALLq] R@10={metrics[10]:.4f} R@50={metrics[50]:.4f} R@500={metrics[500]:.4f} QPS={qps:.1f}", flush=True)

    if '--smoke' in sys.argv:
        print('SMOKE OK', flush=True); return
    today = datetime.date.today().isoformat(); rows = []
    rows.append([NUM_SAMPLES, "base", "", metrics[10], metrics[50], metrics[100], metrics[500], round(qps, 1)])


    note = ("ICWS brute-force sketch baseline, Parks FULL scale (187,019 corpus / ALL held-out queries), Stage-1, "
            "GPU-vectorized signing (reimplemented from 26_icws_weighted_minhash_512.ipynb, "
            "whose pure-NumPy signing loop took 542min at full 187K scale); num_samples=256")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for dim, stage, ck, r10, r50, r100, r500, qps_ in rows:
            w.writerow([today, "parksfull_allq", f"parksfull-ICWSbrute-d{dim}", dim, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps_, "run_icws_brute_full.py", note])
    with open("/raid/ruban/hpmlproj/term_project/SigSpatial/RESULTS_LOG.csv", "a", newline="") as f:
        w = csv.writer(f)
        for dim, stage, ck, r10, r50, r100, r500, qps_ in rows:
            w.writerow([today, "parksfull_allq", f"parksfull-ICWSbrute-d{dim}", dim, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps_, "run_icws_brute_full.py", note])
    print(f"appended {len(rows)} rows -> {CSV} and RESULTS_LOG.csv", flush=True)


if __name__ == '__main__':
    main()
