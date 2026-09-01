#!/usr/bin/env python3
"""Eval-only fix for run_matdistill_listwise_ads_10k.py: the original eval loop
built the width-k embedding as `norm_masked(Z * mask)` -- a zero-padded 4096-dim
vector with only k nonzero entries, not a genuinely compacted k-dim vector. This
gave misleading QPS numbers (nmslib was searching over full 4096-dim vectors at
every nominal width, e.g. "d=256" ran at ~976 QPS, barely above the real 4096-d
baseline's 1339 QPS) and is not representative of ADS's actual deployment story
(store/search only the selected k columns, same cost profile as a real prefix).

This reuses the already-trained checkpoint (no retraining needed -- training
itself was unaffected by this bug, only the eval's embedding constrution was)
and re-evaluates with `Z[:, selected_idx]` -- an actual (N, k) array -- per width.

Launch: CUDA_VISIBLE_DEVICES=5 python eval_listwise_ads_10k.py"""
import sys, time, csv, datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (load_dataset_normalized, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)
DEV = torch.device('cuda:0'); THREADS = 32; torch.set_num_threads(THREADS)
PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096
EF = 200; RB = 64; CAND_KS = [500, 1000]
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_matdistill_listwise_ads_10k.pt'


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))


def main():
    qt, gt, qs, cq, qq, cs, qtn = load_dataset_normalized('10k'); IN = qtn.shape[1]
    VN = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEV)
    m = Net(IN).to(DEV)
    ckpt = torch.load(CKPT, map_location=DEV, weights_only=True)
    m.load_state_dict(ckpt["model"]); m.eval(); imp_final = ckpt["imp"].to(DEV)
    with torch.no_grad():
        Z = torch.cat([m(VN[i:i + 1024]) for i in range(0, len(VN), 1024)])
    today = datetime.date.today().isoformat(); rows = []
    Cr = qt[:qs].astype(np.float32); Cr_sum = Cr.sum(1).astype(np.float32); Qr = qt[qs:].astype(np.float32)
    for k in PREFIXES:
        with torch.no_grad():
            sel_idx = torch.topk(imp_final, k).indices  # deterministic, final learned importance, no noise
            Zk = norm(Z[:, sel_idx])  # actually compacted to (N, k) -- the eval bug this script fixes
        emb = Zk.cpu().numpy().astype(np.float32); ce = emb[:qs]; qe = emb[qs:]
        mk = max(max(CAND_KS), 500)
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, qs, mk)
        print(f"[ads-fixed d={k} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
        rows.append([k, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB); e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[ads-fixed d={k} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([k, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = (f"ADS (learnable top-m dim selection, straight-through Gumbel) ablation vs MatDistillListwise10K, "
            f"EVAL FIX: original eval used a zero-padded 4096-dim vector per width (wrong QPS, e.g. d=256 ran "
            f"at ~976 QPS) instead of an actually-compacted (N,k) vector -- this re-evaluates the same trained "
            f"checkpoint with real dimensionality reduction; RAW-WJ target; tau=0.1; efSearch={EF}")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for k, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "10k", f"ListwiseADS10K-Fixed-d{k}", k, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "eval_listwise_ads_10k.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
