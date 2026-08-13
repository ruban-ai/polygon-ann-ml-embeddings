#!/usr/bin/env python3
"""Eval-only for the trained water-body WJ-distillation checkpoint, against the
FIXED gt_wb50k.pkl (the training run's own end-of-job eval used a stale in-memory
copy loaded before the GT key-remap fix, so those numbers are bogus zeros -- this
redoes just the eval, no retraining needed).
Launch: CUDA_VISIBLE_DEVICES=0 python eval_wb50k.py"""
import sys, time, csv, datetime, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (l1_simplex, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)

DEV = torch.device('cuda:0')
PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096
EF = 200; RB = 64; CAND_KS = [1000, 2000]; QS = 40000
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))


def main():
    qt = np.load('/raid/ruban/hpmlproj/term_project/SigSpatial/qt_wb50k.npy')
    IN = qt.shape[1]
    qtn = l1_simplex(qt.copy())
    QTN = np.ascontiguousarray(qtn, dtype=np.float32)
    QT = np.ascontiguousarray(qt, dtype=np.float32)
    gt = pickle.load(open('/raid/ruban/hpmlproj/term_project/SigSpatial/gt_wb50k.pkl', 'rb'))
    print(f"gt: {len(gt)} queries, sample keys {sorted(gt.keys())[:3]}", flush=True)
    Cr = QT[:QS]; Qr = QT[QS:]; Cr_sum = Cr.sum(1).astype(np.float32)

    m = Net(IN).to(DEV)
    m.load_state_dict(torch.load('/raid/ruban/hpmlproj/term_project/SigSpatial/best_wb50k_distill.pt',
                                  map_location=DEV, weights_only=True))
    m.eval()
    THREADS = 120; torch.set_num_threads(THREADS)
    with torch.no_grad():
        Z = torch.cat([m(torch.from_numpy(QTN[i:i + 512]).to(DEV)) for i in range(0, len(QTN), 512)])

    today = datetime.date.today().isoformat(); rows = []
    for d in PREFIXES:
        emb = norm(Z[:, :d]).cpu().numpy().astype(np.float32); ce = emb[:QS]; qe = emb[QS:]
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=2000, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, QS, 2000)
        print(f"[distill d={d} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} QPS={info['qps']:.0f}", flush=True)
        rows.append([d, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB)
            e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, QS, ck)
            print(f"[distill d={d} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([d, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()

    note = "wb-real0.006 distill corpus-trained, eval held-out queries; raw-WJ; efSearch=200; 40k corpus/10k queries; re-eval with fixed GT keys"
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for d, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "wb50k", f"wb50k-distill-d{d}", d, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "eval_wb50k.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
