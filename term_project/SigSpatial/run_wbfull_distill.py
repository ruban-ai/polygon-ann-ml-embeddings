#!/usr/bin/env python3
"""Water-body FULL-SCALE MSE distillation: the comparison point for
run_wbfull_listwise.py, mirroring run_matdistill_fulleval_ddp.py's role for
Parks-full (tab:full233's "MSE row shown for comparison"). Same architecture/
optimizer/prefixes/epoch budget as the listwise run, only the loss differs
(MSE regression instead of listwise cross-entropy). Reuses the corpus kNN
cache computed by run_wbfull_listwise.py if it already ran (same corpus,
same MAX_POS=30) -- saves recomputing the O(QS^2) precompute step.

Launch: torchrun --standalone --nproc_per_node=8 run_wbfull_distill.py"""
import sys, os, time, csv, datetime, random, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (l1_simplex, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)

PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096
B = 512; EPOCHS = 5; LR = 1e-3; WD = 1e-4; MAX_POS = 30; SEED = 42; QS = 358840
BASE = '/raid/ruban/hpmlproj/term_project/SigSpatial'
KNN_CACHE = f'{BASE}/corpus_knn_wbfull.npy'
QT_PATH = f'{BASE}/qt_wbfull.npy'
QTN_CACHE = f'{BASE}/qt_norm_wbfull.npy'
GT_PATH = f'{BASE}/gt_wbfull.pkl'
CSV = f'{BASE}/NEW_RESULTS.csv'
CKPT = f'{BASE}/best_wbfull_distill.pt'
EF = 200; RB = 64; CAND_KS = [1000, 2000]


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)
def emb_wj(a, b): l1 = torch.cdist(a, b, p=1); return (2.0 - l1) / (2.0 + l1)


@torch.no_grad()
def raw_wj(R, chunk=64):
    M = R.shape[0]; out = torch.empty(M, M, device=R.device)
    for i in range(0, M, chunk):
        ri = R[i:i + chunk]
        out[i:i + chunk] = torch.minimum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2) / \
            torch.maximum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2).clamp(min=1e-10)
    return out


def matry_distill(z, tw):
    tot = 0.
    for m in PREFIXES:
        zm = norm(z[:, :m]); tot = tot + F.mse_loss(emb_wj(zm, zm), tw)
    return tot / len(PREFIXES)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))
    def embed(s, x): return F.relu(s.encoder(x))


class PairDS(Dataset):
    def __init__(s, knn):
        s.p = [(i, int(j)) for i in range(knn.shape[0]) for j in knn[i]]; random.Random(SEED).shuffle(s.p)

    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.p[i]


def evaluate(embed_fn, gt, qs, Cr, Qr, Cr_sum, qtn, DEV, tag):
    THREADS = 120; torch.set_num_threads(THREADS)
    with torch.no_grad():
        Z = torch.cat([embed_fn(torch.from_numpy(qtn[i:i + 512]).to(DEV)) for i in range(0, len(qtn), 512)])
    today = datetime.date.today().isoformat(); rows = []
    for m in PREFIXES:
        emb = norm(Z[:, :m]).cpu().numpy().astype(np.float32); ce = emb[:qs]; qe = emb[qs:]
        mk = max(max(CAND_KS), 500)
        nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
        base = eval_recall(gt, nb, qs, mk)
        print(f"[{tag} d={m} base ALLq] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
        rows.append([m, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB)
            e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[{tag} d={m} rerank K={ck} ALLq] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([m, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = (f"wb-real0.006 FULL-SCALE MSE distillation, corpus-trained on {QS} corpus, "
            f"eval ALL {len(Qr)} held-out queries; raw-WJ GT; efSearch={EF}; "
            f"comparison point for run_wbfull_listwise.py, mirrors run_matdistill_fulleval_ddp.py")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for m, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "wbfull_allq", f"wbfull-distill-d{m}", m, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_wbfull_distill.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


def main():
    dist.init_process_group('nccl'); lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV = torch.device(f'cuda:{lr}'); main = (lr == 0); world = dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    qt = np.load(QT_PATH); IN = qt.shape[1]
    if os.path.exists(QTN_CACHE):
        qtn = np.load(QTN_CACHE)
        if main: print(f"qt_norm loaded from cache {qtn.shape}", flush=True)
    else:
        qtn = l1_simplex(qt.copy())
        if main:
            np.save(QTN_CACHE, qtn)
            print(f"qt_norm computed+cached {qtn.shape}", flush=True)
        dist.barrier()
        if not main: qtn = np.load(QTN_CACHE)
    QTN = np.ascontiguousarray(qtn, dtype=np.float32); QT = np.ascontiguousarray(qt, dtype=np.float32)
    gt = pickle.load(open(GT_PATH, 'rb'))
    Cn_g = torch.from_numpy(QTN[:QS]).to(DEV)
    if main: print(f"DDP world={world} method=distill(MSE) corpus={QS} queries={qt.shape[0]-QS} dim={IN}", flush=True)
    Cr = QT[:QS]; Qr = QT[QS:]; Cr_sum = Cr.sum(1).astype(np.float32)

    if not os.path.exists(KNN_CACHE):
        t0 = time.time(); per = (QS + world - 1) // world; s0 = lr * per; s1 = min(s0 + per, QS)
        out = np.empty((max(s1 - s0, 0), MAX_POS), dtype=np.int64)
        if main: print(f"computing corpus kNN shards (QS={QS}, per-rank~{per} rows) ...", flush=True)
        for i in range(s0, s1, 2048):
            j = min(i + 2048, s1); a = Cn_g[i:j]; d = torch.cdist(a, Cn_g, p=1)
            r = torch.arange(a.shape[0], device=DEV); d[r, torch.arange(i, j, device=DEV)] = 1e9
            out[i - s0:j - s0] = torch.topk(d, MAX_POS, dim=1, largest=False).indices.cpu().numpy()
            if main and (i // 2048) % 10 == 0:
                print(f"  kNN shard progress rank0: {i-s0}/{s1-s0} rows ({(time.time()-t0)/60:.1f}min)", flush=True)
        np.save(f'{BASE}/knn_wbfull_shard_{lr}.npy', out); dist.barrier()
        if main:
            knn = np.concatenate([np.load(f'{BASE}/knn_wbfull_shard_{r}.npy') for r in range(world)], axis=0)
            np.save(KNN_CACHE, knn); print(f"corpus kNN {knn.shape} {(time.time()-t0)/60:.1f}min", flush=True)
        dist.barrier()
    knn = np.load(KNN_CACHE)
    if main: print(f"corpus kNN {knn.shape}", flush=True)

    model = DDP(Net(IN).to(DEV), device_ids=[lr])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ds = PairDS(knn); sampler = DistributedSampler(ds, shuffle=True, seed=SEED)
    loader = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()
    if main: print(f"corpus pairs={len(ds):,} epochs={EPOCHS} steps/epoch~{len(ds)//(world*B)}", flush=True)
    for ep in range(1, EPOCHS + 1):
        model.train(); sampler.set_epoch(ep); tl = st = 0
        it = tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=100, mininterval=30) if main else loader
        for ai, pi in it:
            ids = torch.cat([ai, pi]).numpy()
            xb = torch.from_numpy(QTN[ids]).to(DEV); rb = torch.from_numpy(QT[ids]).to(DEV)
            z = model(xb)
            with torch.no_grad(): tw = raw_wj(rb)
            loss = matry_distill(z, tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); st += 1
        sch.step()
        if main:
            print(f"ep{ep:02d} mse={tl/st:.6f} {(time.time()-t0)/60:.1f}min", flush=True)
            if tl / st < best: best = tl / st; torch.save(model.module.state_dict(), CKPT)
    dist.barrier()
    if not main: dist.destroy_process_group(); return
    enc = Net(IN).to(DEV); enc.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); enc.eval()
    evaluate(enc.embed, gt, QS, Cr, Qr, Cr_sum, QTN, DEV, 'wbfull-distill')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
