#!/usr/bin/env python3
"""QPS investigation: eval-only recheck of the existing listwise 50K checkpoint,
run immediately after the fresh MSE 50K retrain+eval (see PAPER_REVIEW_IMPROVEMENTS_LOG.md
sec 2) on the same idle GPU, for a genuinely back-to-back, same-machine-state QPS
comparison against MSE. No retraining -- loads best_50k_listwise.pt as-is.

Launch: CUDA_VISIBLE_DEVICES=1 python eval_50k_listwise_recheck.py"""
import sys, time, csv, datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (l1_simplex, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)

DEV = torch.device('cuda:0')
PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096
EF = 200; RB = 64; CAND_KS = [1000, 2000]; QS = 40000
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_50k_listwise.pt'


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))


def main():
    THREADS = 120; torch.set_num_threads(THREADS)
    qt = np.load('/raid/ruban/hpmlproj/term_project/SigSpatial/qt_50k.npy'); IN = qt.shape[1]
    qtn = l1_simplex(qt.copy())
    QTN = np.ascontiguousarray(qtn, dtype=np.float32); QT = np.ascontiguousarray(qt, dtype=np.float32)
    import pickle
    gt = pickle.load(open('/raid/ruban/hpmlproj/term_project/SigSpatial/gt_50k.pkl', 'rb'))
    print(f"recheck: listwise 50K, back-to-back with fresh MSE retrain eval, same GPU node", flush=True)

    m = Net(IN).to(DEV)
    m.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); m.eval()
    with torch.no_grad():
        Z = torch.cat([m(torch.from_numpy(QTN[i:i + 512]).to(DEV)) for i in range(0, len(QTN), 512)])

    today = datetime.date.today().isoformat(); rows = []
    Cr = QT[:QS]; Cr_sum = Cr.sum(1).astype(np.float32); Qr = QT[QS:]
    for k in PREFIXES:
        emb = norm(Z[:, :k]).cpu().numpy().astype(np.float32); ce = emb[:QS]; qe = emb[QS:]
        mk = max(max(CAND_KS), 500)
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, QS, mk)
        print(f"[listwise-recheck d={k} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} QPS={info['qps']:.0f}", flush=True)
        rows.append([k, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB)
            e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, QS, ck)
            print(f"[listwise-recheck d={k} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([k, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()

    note = ("QPS investigation: back-to-back recheck of existing listwise checkpoint (no "
            "retraining) immediately after a fresh MSE retrain+eval on the same idle GPU node, "
            "to test whether the listwise-vs-MSE QPS gap is measurement noise or real; "
            "see PAPER_REVIEW_IMPROVEMENTS_LOG.md sec 2")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for k, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "pk50k", f"50k-listwise-recheck-d{k}", k, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "eval_50k_listwise_recheck.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == '__main__':
    main()
