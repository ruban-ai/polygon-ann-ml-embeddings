#!/usr/bin/env python3
"""Independent (non-Matryoshka) 4096-d listwise training at 50K -- isolates whether
the joint multi-width Matryoshka loss (run_50k_listwise.py, 5 simultaneous prefix
terms) is costing real accuracy even at the widest width, where there is no
truncation at all. Same architecture (encoder already outputs 4096-d natively), same
data/optimizer/epoch budget/listwise loss/tau as run_50k_listwise.py -- the ONLY
difference is the loss has a single term (m=4096) instead of five.

If this reaches meaningfully higher recall than the existing Matryoshka-4096 slice
(R@50=0.929), the joint width-sharing loss has a real cost even with no truncation.
If it lands at roughly the same recall, the ~92-93% ceiling is not a Matryoshka tax
and likely reflects a more fundamental limit (encoder capacity, loss, or data).

Launch: CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 run_50k_listwise_independent4096.py"""
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

DIM = 4096; EMB = 4096
B = 512; LR = 1e-3; WD = 1e-4; MAX_POS = 30; EPOCHS = 10; TAU = 0.1
EF = 200; RB = 64; CAND_KS = [1000, 2000]; SEED = 42; QS = 40000
KNN_CACHE = '/raid/ruban/hpmlproj/term_project/SigSpatial/corpus_knn_50k.npy'
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_50k_listwise_indep4096.pt'


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


NEG_INF = -1e9


def listwise_single(z, tw):
    """Single-width listwise loss -- no Matryoshka loop, just m=4096 (the full,
    native output width; no truncation is happening here at all)."""
    N = z.shape[0]
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    target = F.softmax(tw.masked_fill(diag, NEG_INF) / TAU, dim=1)
    zm = norm(z)
    that = emb_wj(zm, zm).masked_fill(diag, NEG_INF)
    logp = F.log_softmax(that / TAU, dim=1)
    return (-(target * logp).sum(1)).mean()


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
    emb = norm(Z).cpu().numpy().astype(np.float32); ce = emb[:qs]; qe = emb[qs:]
    mk = max(max(CAND_KS), 500)
    nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=mk, threads=THREADS, query_params={"efSearch": EF})
    base = eval_recall(gt, nb, qs, mk)
    print(f"[{tag} d={DIM} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} QPS={info['qps']:.0f}", flush=True)
    rows.append([DIM, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
    preload_rerank_corpus(Cr, Cr_sum)
    for ck in CAND_KS:
        cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
        t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB)
        e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
        mm = eval_recall(gt, rr, qs, ck)
        print(f"[{tag} d={DIM} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
        rows.append([DIM, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
    release_rerank_corpus()
    note = ("Independent (non-Matryoshka) single-width listwise training at 4096-d: same "
            "arch/data/10ep/listwise loss/tau as run_50k_listwise.py, but only the m=4096 loss "
            "term (no smaller-prefix terms) -- isolates whether joint Matryoshka training costs "
            "accuracy even with no truncation. Compare to 50k-listwise-d4096 (R@50=0.929).")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for m, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "pk50k", f"50k-listwise-indep4096", m, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_50k_listwise_independent4096.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


def main():
    dist.init_process_group('nccl'); lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV = torch.device(f'cuda:{lr}'); main = (lr == 0); world = dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    qt = np.load('/raid/ruban/hpmlproj/term_project/SigSpatial/qt_50k.npy'); IN = qt.shape[1]; qtn = l1_simplex(qt.copy())
    QTN = np.ascontiguousarray(qtn, dtype=np.float32); QT = np.ascontiguousarray(qt, dtype=np.float32)
    gt = pickle.load(open('/raid/ruban/hpmlproj/term_project/SigSpatial/gt_50k.pkl', 'rb'))
    if main: print(f"DDP world={world} method=listwise-independent-4096 corpus={QS} queries={qt.shape[0]-QS} dim={IN}", flush=True)
    Cr = QT[:QS]; Qr = QT[QS:]; Cr_sum = Cr.sum(1).astype(np.float32)

    knn = np.load(KNN_CACHE)  # reuse existing cache, same corpus/MAX_POS as run_50k_listwise.py

    model = DDP(Net(IN).to(DEV), device_ids=[lr])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ds = PairDS(knn); sampler = DistributedSampler(ds, shuffle=True, seed=SEED)
    loader = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()
    if main: print(f"corpus pairs={len(ds):,} epochs={EPOCHS} steps/epoch~{len(ds)//(world*B)}", flush=True)
    for ep in range(1, EPOCHS + 1):
        model.train(); sampler.set_epoch(ep); tl = st = 0
        it = tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=100, mininterval=10) if main else loader
        for ai, pi in it:
            ids = torch.cat([ai, pi]).numpy()
            xb = torch.from_numpy(QTN[ids]).to(DEV); rb = torch.from_numpy(QT[ids]).to(DEV)
            z = model(xb)
            with torch.no_grad(): tw = raw_wj(rb)
            loss = listwise_single(z, tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); st += 1
        sch.step()
        if main:
            print(f"ep{ep:02d} listwise_ce={tl/st:.5f} {(time.time()-t0)/60:.1f}min", flush=True)
            if tl / st < best: best = tl / st; torch.save(model.module.state_dict(), CKPT)
    dist.barrier()
    if not main: dist.destroy_process_group(); return
    enc = Net(IN).to(DEV); enc.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); enc.eval()
    evaluate(enc.embed, gt, QS, Cr, Qr, Cr_sum, QTN, DEV, 'listwise-indep4096')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
