#!/usr/bin/env python3
"""S-XBM escalation to 50K: at 10K, S-XBM (candidate #3 from LOSS_EXPLORATION_LOG.md
§3.3) was a completely flat null result vs. plain listwise -- but with an explicit
scale-dependence caveat, since S-XBM's whole motivation (a batch is a vanishingly
small slice of a *large* corpus) barely applies when 10K's corpus (8K) is only
traversed a few thousand steps total. This resolves that caveat directly: identical
S-XBM mechanism (FIFO queue of recent ids, mined each step by true raw-WJ similarity
for extra hard/diverse comparison partners), same architecture/optimizer/listwise
loss/tau as run_50k_listwise.py -- only the batch composition differs, exactly
mirroring the 10K vs. 10K-listwise-baseline comparison.

DDP-adapted: FIFO queue and raw-corpus tensor are per-rank (each GPU mines from its
own recently-seen ids only) -- a reasonable adaptation since the queue's job is
local batch-diversity augmentation, not global corpus coverage.

Launch: torchrun --standalone --nproc_per_node=8 run_50k_listwise_sxbm.py"""
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
B = 512; LR = 1e-3; WD = 1e-4; MAX_POS = 30; EPOCHS = 10; TAU = 0.1
EF = 200; RB = 64; CAND_KS = [1000, 2000]; SEED = 42; QS = 40000
QUEUE_MAX = 2048; MINE_K = 64  # same as run_matdistill_listwise_sxbm_10k.py
KNN_CACHE = '/tmp/corpus_knn_50k.npy'  # shared cache from run_50k_ddp.py, same corpus/MAX_POS
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
CKPT = '/raid/ruban/hpmlproj/term_project/SigSpatial/best_50k_listwise_sxbm.pt'


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


@torch.no_grad()
def raw_wj_cross(A, Bm, chunk=16):
    """chunk=16 deliberately small: intermediate is chunk*len(Bm)*D floats;
    with D~18382, queue=2048, chunk=16 -> ~2.4GB (chunk=256 would be ~76GB, OOM)."""
    out = torch.empty(A.shape[0], Bm.shape[0], device=A.device)
    for i in range(0, A.shape[0], chunk):
        ai = A[i:i + chunk]
        mn = torch.minimum(ai.unsqueeze(1), Bm.unsqueeze(0)).sum(2)
        mx = torch.maximum(ai.unsqueeze(1), Bm.unsqueeze(0)).sum(2).clamp(min=1e-10)
        out[i:i + chunk] = mn / mx
    return out


NEG_INF = -1e9


def matry_listwise(z, tw):
    N = z.shape[0]
    diag = torch.eye(N, dtype=torch.bool, device=z.device)
    target = F.softmax(tw.masked_fill(diag, NEG_INF) / TAU, dim=1)
    tot = 0.
    for m in PREFIXES:
        zm = norm(z[:, :m])
        that = emb_wj(zm, zm).masked_fill(diag, NEG_INF)
        logp = F.log_softmax(that / TAU, dim=1)
        tot = tot + (-(target * logp).sum(1)).mean()
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


class FIFOQueue:
    """Ring buffer of corpus ids, on-device. mine() finds the globally top-k
    (anchor, queue-member) pairs by true raw WJ and returns the queue-side ids."""
    def __init__(s, maxlen, device):
        s.maxlen = maxlen; s.dev = device
        s.buf = torch.empty(0, dtype=torch.long, device=device)

    def push(s, ids):
        s.buf = torch.cat([s.buf, ids.to(s.dev)])[-s.maxlen:]

    def mine(s, anchor_ids, RAW, k):
        if s.buf.numel() < 8:
            return torch.empty(0, dtype=torch.long, device=s.dev)
        sims = raw_wj_cross(RAW[anchor_ids], RAW[s.buf])
        flat_k = min(k, sims.numel())
        top = torch.topk(sims.flatten(), flat_k).indices
        queue_idx = top % sims.shape[1]
        mined = s.buf[queue_idx]
        return torch.unique(mined)


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
    note = (f"pk-real50k0.002 listwise+S-XBM (50K escalation of 10K null result, see "
            f"LOSS_EXPLORATION_LOG.md {{3.3): queue_max={QUEUE_MAX} mine_k={MINE_K}, per-rank FIFO; "
            f"raw-WJ; tau={TAU}; efSearch={EF}")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for m, stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "pk50k", f"50k-listwiseSXBM-d{m}", m, stage, ck, round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4), qps, "run_50k_listwise_sxbm.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


def main():
    dist.init_process_group('nccl'); lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV = torch.device(f'cuda:{lr}'); main = (lr == 0); world = dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    qt = np.load('/tmp/qt_50k.npy'); IN = qt.shape[1]; qtn = l1_simplex(qt.copy())
    QTN = np.ascontiguousarray(qtn, dtype=np.float32); QT = np.ascontiguousarray(qt, dtype=np.float32)
    gt = pickle.load(open('/tmp/gt_50k.pkl', 'rb'))
    Cn_g = torch.from_numpy(QTN[:QS]).to(DEV)
    RAW = torch.from_numpy(QT[:QS]).to(DEV)  # full corpus raw vectors resident on-device for S-XBM mining
    if main: print(f"DDP world={world} method=listwise+S-XBM corpus={QS} queries={qt.shape[0]-QS} dim={IN} "
                    f"tau={TAU} queue_max={QUEUE_MAX} mine_k={MINE_K}", flush=True)
    Cr = QT[:QS]; Qr = QT[QS:]; Cr_sum = Cr.sum(1).astype(np.float32)

    if not os.path.exists(KNN_CACHE):
        t0 = time.time(); per = (QS + world - 1) // world; s0 = lr * per; s1 = min(s0 + per, QS)
        out = np.empty((max(s1 - s0, 0), MAX_POS), dtype=np.int64)
        for i in range(s0, s1, 2048):
            j = min(i + 2048, s1); a = Cn_g[i:j]; d = torch.cdist(a, Cn_g, p=1)
            r = torch.arange(a.shape[0], device=DEV); d[r, torch.arange(i, j, device=DEV)] = 1e9
            out[i - s0:j - s0] = torch.topk(d, MAX_POS, dim=1, largest=False).indices.cpu().numpy()
        np.save(f'/tmp/knn50k_shard_{lr}.npy', out); dist.barrier()
        if main:
            knn = np.concatenate([np.load(f'/tmp/knn50k_shard_{r}.npy') for r in range(world)], axis=0)
            np.save(KNN_CACHE, knn); print(f"corpus kNN {knn.shape} {(time.time()-t0)/60:.1f}min", flush=True)
        dist.barrier()
    knn = np.load(KNN_CACHE)

    model = DDP(Net(IN).to(DEV), device_ids=[lr])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ds = PairDS(knn); sampler = DistributedSampler(ds, shuffle=True, seed=SEED)
    loader = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()
    queue = FIFOQueue(QUEUE_MAX, DEV)
    if main: print(f"corpus pairs={len(ds):,} epochs={EPOCHS} steps/epoch~{len(ds)//(world*B)}", flush=True)
    for ep in range(1, EPOCHS + 1):
        model.train(); sampler.set_epoch(ep); tl = st = 0
        it = tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=100, mininterval=10) if main else loader
        for ai, pi in it:
            base_ids = torch.cat([ai, pi]).to(DEV)
            mined_ids = queue.mine(base_ids, RAW, MINE_K)
            if mined_ids.numel():
                mask = ~torch.isin(mined_ids, base_ids)
                mined_ids = mined_ids[mask]
            ids = torch.cat([base_ids, mined_ids]) if mined_ids.numel() else base_ids
            xb = Cn_g[ids]; rb = RAW[ids]
            z = model(xb)
            with torch.no_grad(): tw = raw_wj(rb)
            loss = matry_listwise(z, tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); st += 1
            queue.push(base_ids.detach())
        sch.step()
        if main:
            print(f"ep{ep:02d} listwise_ce={tl/st:.5f} qsize={queue.buf.numel()} {(time.time()-t0)/60:.1f}min", flush=True)
            if tl / st < best: best = tl / st; torch.save(model.module.state_dict(), CKPT)
    dist.barrier()
    if not main: dist.destroy_process_group(); return
    enc = Net(IN).to(DEV); enc.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)); enc.eval()
    evaluate(enc.embed, gt, QS, Cr, Qr, Cr_sum, QTN, DEV, 'listwise+sxbm')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
