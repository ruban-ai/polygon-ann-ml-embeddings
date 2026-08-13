#!/usr/bin/env python3
"""Eval-only for the trained listwise (B4 ablation) checkpoint -- the training
run's own eval crashed on the rerank loop (CAND_KS=[100,500,1000] hits the same
KeyError eval_matdistill_10k.py's docstring already documents: candidate_k=100
caps eval_recall's max_k so the R@500 key doesn't exist). Redo eval only, with
the fix (CAND_KS=[500,1000]), no retraining needed.
Launch: CUDA_VISIBLE_DEVICES=1 python eval_listwise_10k.py"""
import sys, time, csv, datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (load_dataset_normalized, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)
DEV = torch.device('cuda:0'); THREADS = 32; torch.set_num_threads(THREADS)
PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096
EF = 200; RB = 64; CAND_KS = [500, 1000]
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_matdistill_listwise_10k.pt'


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))


def main():
    qt, gt, qs, cq, qq, cs, qtn = load_dataset_normalized('10k'); IN = qtn.shape[1]
    VN = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEV)
    m = Net(IN).to(DEV); m.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); m.eval()
    with torch.no_grad():
        Z = torch.cat([m(VN[i:i + 1024]) for i in range(0, len(VN), 1024)])
    today = datetime.date.today().isoformat(); rows = []
    Cr = qt[:qs].astype(np.float32); Cr_sum = Cr.sum(1).astype(np.float32); Qr = qt[qs:].astype(np.float32)
    for k in PREFIXES:
        emb = norm(Z[:, :k]).cpu().numpy().astype(np.float32); ce = emb[:qs]; qe = emb[qs:]
        mk = max(max(CAND_KS), 500)
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, qs, mk)
        print(f"[listwise d={k} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
        rows.append([k, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB); e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[listwise d={k} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([k, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = f"MatDistill-Listwise 10K prefixes={PREFIXES}; RAW-WJ target; tau=0.1; efSearch={EF}; B4 ablation vs MSE; re-eval (CAND_KS fix)"
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for k, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "10k", f"MatDistillListwise10K-d{k}", k, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "eval_listwise_10k.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
