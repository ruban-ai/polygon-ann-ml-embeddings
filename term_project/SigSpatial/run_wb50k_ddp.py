#!/usr/bin/env python3
"""Water-body 50K-style benchmark: 40K corpus (0..39999) / 10K held-out queries
(358840..368839), built by build_wb.py. Same architecture/loss/procedure as
run_50k_ddp.py, just pointed at the water-body data -- a second polygon class
for Review 2's ask, methodology otherwise identical to the Parks 50K benchmark.
Methods: distill | rp.
Launch: CUDA_VISIBLE_DEVICES=0..7 torchrun --standalone --nproc_per_node=8 run_wb50k_ddp.py --method distill"""
import sys, os, time, csv, datetime, random, pickle, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (l1_simplex, shifted_l1_simplex, nmslib_neighbors,
    preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu, eval_recall)

ap = argparse.ArgumentParser(); ap.add_argument('--method', choices=['distill', 'rp'], required=True)
A = ap.parse_args()
PREFIXES = [256, 512, 1024, 2048, 4096]; EMB = 4096
B = 512; LR = 1e-3; WD = 1e-4; MAX_POS = 30
EPOCHS = {'distill': 10}.get(A.method, 1)
EF = 200; RB = 64; CAND_KS = [1000, 2000]; SEED = 42; QS = 40000
KNN_CACHE = '/raid/ruban/hpmlproj/term_project/SigSpatial/corpus_knn_wb50k.npy'
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
QT_PATH = '/raid/ruban/hpmlproj/term_project/SigSpatial/qt_wb50k.npy'
GT_PATH = '/raid/ruban/hpmlproj/term_project/SigSpatial/gt_wb50k.pkl'


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)


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
        zm = norm(z[:, :m])
        l1 = torch.cdist(zm, zm, p=1)
        emb_wj = (2.0 - l1) / (2.0 + l1)
        tot = tot + F.mse_loss(emb_wj, tw)
    return tot / len(PREFIXES)


class Net(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.encoder = nn.Sequential(nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, EMB, bias=False), nn.BatchNorm1d(EMB))

    def forward(s, x): return F.relu(s.encoder(x))
    def embed(s, x): return F.relu(s.encoder(x))


class PairDS(Dataset):
    def __init__(s, knn):
        s.p = [(i, int(j)) for i in range(knn.shape[0]) for j in knn[i]]
        random.Random(SEED).shuffle(s.p)

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
        print(f"[{tag} d={m} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} QPS={info['qps']:.0f}", flush=True)
        rows.append([m, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])])
        preload_rerank_corpus(Cr, Cr_sum)
        for ck in CAND_KS:
            cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
            t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB)
            e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
            mm = eval_recall(gt, rr, qs, ck)
            print(f"[{tag} d={m} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
            rows.append([m, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
        release_rerank_corpus()
    note = f"wb-real0.006 {tag} corpus-trained, eval held-out queries; raw-WJ; efSearch={EF}; 40k corpus/10k queries"
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for m, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "wb50k", f"wb50k-{tag}-d{m}", m, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_wb50k_ddp.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


def main():
    dist.init_process_group('nccl'); lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV = torch.device(f'cuda:{lr}'); main = (lr == 0); world = dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    qt = np.load(QT_PATH); IN = qt.shape[1]; qtn = l1_simplex(qt.copy())
    QTN = np.ascontiguousarray(qtn, dtype=np.float32); QT = np.ascontiguousarray(qt, dtype=np.float32)
    gt = pickle.load(open(GT_PATH, 'rb'))
    Cn_g = torch.from_numpy(QTN[:QS]).to(DEV)
    if main: print(f"DDP world={world} method={A.method} corpus={QS} queries={qt.shape[0]-QS} dim={IN}", flush=True)
    Cr = QT[:QS]; Qr = QT[QS:]; Cr_sum = Cr.sum(1).astype(np.float32)

    if A.method == 'rp':
        if main:
            rng = np.random.default_rng(SEED)
            for m in PREFIXES:
                W = torch.tensor((rng.standard_normal((IN, m)) / np.sqrt(m)).astype(np.float32), device=DEV)
                Z = torch.cat([(torch.from_numpy(QT[i:i + 2048]).to(DEV) @ W) for i in range(0, len(QT), 2048)]).cpu().numpy()
                Zs = torch.tensor(shifted_l1_simplex(Z), device=DEV)
                emb = norm(Zs).cpu().numpy().astype(np.float32); ce = emb[:QS]; qe = emb[QS:]
                THREADS = 120; torch.set_num_threads(THREADS)
                nb, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=2000, threads=THREADS, query_params={"efSearch": EF})
                base = eval_recall(gt, nb, QS, 2000)
                print(f"[rp d={m} base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@500={base[500]:.4f} QPS={info['qps']:.0f}", flush=True)
                preload_rerank_corpus(Cr, Cr_sum); rows = [[m, "base", "", base[10], base[50], base[100], base[500], round(info['qps'])]]
                for ck in CAND_KS:
                    cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck, threads=THREADS, query_params={"efSearch": EF})
                    t1 = time.time(); rr = rerank_wj_gpu(Qr, cand, Cr, Cr_sum, top_k=ck, batch_size=RB)
                    e2e = len(Qr) / max(ci["query_s"] + (time.time() - t1), 1e-9)
                    mm = eval_recall(gt, rr, QS, ck)
                    print(f"[rp d={m} rerank K={ck}] R@10={mm[10]:.4f} R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
                    rows.append([m, "rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
                release_rerank_corpus()
                today = datetime.date.today().isoformat()
                with open(CSV, "a", newline="") as f:
                    w = csv.writer(f)
                    for mm_, stage, ck, r10, r50, r100, r500, qps in rows:
                        w.writerow([today, "wb50k", f"wb50k-rp-d{mm_}", mm_, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_wb50k_ddp.py", "wb-real0.006 rp; raw-WJ; 40k corpus/10k queries"])
            print("rp done", flush=True)
        dist.destroy_process_group(); return

    if not os.path.exists(KNN_CACHE):
        t0 = time.time(); per = (QS + world - 1) // world; s0 = lr * per; s1 = min(s0 + per, QS)
        out = np.empty((max(s1 - s0, 0), MAX_POS), dtype=np.int64)
        for i in range(s0, s1, 2048):
            j = min(i + 2048, s1); a = Cn_g[i:j]; d = torch.cdist(a, Cn_g, p=1)
            r = torch.arange(a.shape[0], device=DEV); d[r, torch.arange(i, j, device=DEV)] = 1e9
            out[i - s0:j - s0] = torch.topk(d, MAX_POS, dim=1, largest=False).indices.cpu().numpy()
        np.save(f'/raid/ruban/hpmlproj/term_project/SigSpatial/knn_wb50k_shard_{lr}.npy', out); dist.barrier()
        if main:
            knn = np.concatenate([np.load(f'/raid/ruban/hpmlproj/term_project/SigSpatial/knn_wb50k_shard_{r}.npy') for r in range(world)], axis=0)
            np.save(KNN_CACHE, knn); print(f"corpus kNN {knn.shape} {(time.time()-t0)/60:.1f}min", flush=True)
        dist.barrier()
    knn = np.load(KNN_CACHE)

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
            loss = matry_distill(z, tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); st += 1
        sch.step()
        if main:
            print(f"ep{ep:02d} loss={tl/st:.5f} {(time.time()-t0)/60:.1f}min", flush=True)
            if tl / st < best: best = tl / st; torch.save(model.module.state_dict(), '/raid/ruban/hpmlproj/term_project/SigSpatial/best_wb50k_distill.pt')
    dist.barrier()
    if not main: dist.destroy_process_group(); return
    enc = Net(IN).to(DEV); enc.load_state_dict(torch.load('/raid/ruban/hpmlproj/term_project/SigSpatial/best_wb50k_distill.pt', map_location=DEV, weights_only=True)); enc.eval()
    evaluate(enc.embed, gt, QS, Cr, Qr, Cr_sum, QTN, DEV, A.method)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
