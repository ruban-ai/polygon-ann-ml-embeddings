#!/usr/bin/env python3
"""Full-scale listwise WJ-distillation TRAINING for Overture 1M / Overture 10M /
Sports full (2026-10-01 runs). Line-for-line the protocol of
run_wbfull_listwise.py (same Net, AdamW lr/wd, cosine, grad-clip 1.0, B=512 pairs
per rank, MAX_POS=30 corpus kNN positives, listwise CE with tau=0.1 averaged over
Matryoshka prefixes, best-epoch checkpoint), with the changes listed in
fulls_common.py (memmap inputs, GPU l1-normalization, tiled corpus kNN, per-config
prefixes). Evaluation is a separate script (eval_fulls_listwise.py) so the next
training job can start while HNSW evaluation runs on the CPU.

Launch: torchrun --standalone --nproc_per_node=8 run_fulls_listwise_train.py <cfg> [--smoke]"""
import sys, os, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS, BASE, raw_matrix, rows

CFG = CONFIGS[sys.argv[1]]; SMOKE = '--smoke' in sys.argv
PREFIXES = CFG['prefixes']; EMB = CFG['emb']
B = 512; EPOCHS = 1 if SMOKE else CFG['epochs']; LR = 1e-3; WD = 1e-4; MAX_POS = 30; TAU = 0.1; SEED = 42
TRAIN_N = 20_000 if SMOKE else CFG['train_n']
KNN_CACHE = CFG['knn'].replace('.npy', '_SMOKE.npy') if SMOKE else CFG['knn']
CKPT = CFG['ckpt'].replace('.pt', '_SMOKE.pt') if SMOKE else CFG['ckpt']
MAX_STEPS = 60 if SMOKE else None
TILE = 250_000            # corpus rows per GPU tile during kNN mining
NEG_INF = -1e9


def norm(x): return x / x.sum(1, keepdim=True).clamp(min=1e-10)
def gnorm(x): return norm(x.clamp(min=0))          # == sota_experiment_common.l1_simplex, on GPU
def emb_wj(a, b): l1 = torch.cdist(a, b, p=1); return (2.0 - l1) / (2.0 + l1)


@torch.no_grad()
def raw_wj(R, chunk=64):
    M = R.shape[0]; out = torch.empty(M, M, device=R.device)
    for i in range(0, M, chunk):
        ri = R[i:i + chunk]
        out[i:i + chunk] = torch.minimum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2) / \
            torch.maximum(ri.unsqueeze(1), R.unsqueeze(0)).sum(2).clamp(min=1e-10)
    return out


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


@torch.no_grad()
def mine_knn(X, lr, world, DEV, main):
    """Exact corpus kNN (L1 on l1-normalized rows == WJ order on the simplex), tiled
    over the corpus so a 1.4M x 12K corpus never has to sit on one GPU."""
    t0 = time.time(); per = (TRAIN_N + world - 1) // world; s0 = lr * per; s1 = min(s0 + per, TRAIN_N)
    nq = max(s1 - s0, 0)
    Q = gnorm(torch.from_numpy(rows(X, np.arange(s0, s1))).to(DEV)) if nq else None
    bestd = torch.full((nq, MAX_POS), float('inf'), device=DEV)
    besti = torch.zeros((nq, MAX_POS), dtype=torch.long, device=DEV)
    ntiles = (TRAIN_N + TILE - 1) // TILE
    if main: print(f"kNN mining: train_n={TRAIN_N:,} per-rank={per:,} tiles={ntiles}", flush=True)
    for t, c0 in enumerate(range(0, TRAIN_N, TILE)):
        c1 = min(c0 + TILE, TRAIN_N)
        C = gnorm(torch.from_numpy(rows(X, np.arange(c0, c1))).to(DEV))
        for i in range(0, nq, 2048):
            j = min(i + 2048, nq); d = torch.cdist(Q[i:j], C, p=1)
            gi = torch.arange(s0 + i, s0 + j, device=DEV)            # global ids of these queries
            self_mask = (gi >= c0) & (gi < c1)
            if self_mask.any():
                r = torch.nonzero(self_mask).squeeze(1); d[r, gi[r] - c0] = float('inf')
            vd, vi = torch.topk(d, min(MAX_POS, c1 - c0), dim=1, largest=False)
            cd = torch.cat([bestd[i:j], vd], 1); ci = torch.cat([besti[i:j], vi + c0], 1)
            o = torch.topk(cd, MAX_POS, dim=1, largest=False).indices
            bestd[i:j] = torch.gather(cd, 1, o); besti[i:j] = torch.gather(ci, 1, o)
        del C; torch.cuda.empty_cache()
        if main:
            el = time.time() - t0
            print(f"  kNN tile {t+1}/{ntiles} done ({el/60:.1f}min, ETA {el/(t+1)*(ntiles-t-1)/60:.1f}min)", flush=True)
    np.save(f'{BASE}/knn_{CFG["name"]}_shard_{lr}{"_SMOKE" if SMOKE else ""}.npy', besti.cpu().numpy())
    dist.barrier()
    if main:
        knn = np.concatenate([np.load(f'{BASE}/knn_{CFG["name"]}_shard_{r}{"_SMOKE" if SMOKE else ""}.npy')
                              for r in range(world)], axis=0)
        assert knn.shape == (TRAIN_N, MAX_POS), knn.shape
        np.save(KNN_CACHE, knn)
        for r in range(world):
            os.remove(f'{BASE}/knn_{CFG["name"]}_shard_{r}{"_SMOKE" if SMOKE else ""}.npy')
        print(f"corpus kNN {knn.shape} {(time.time()-t0)/60:.1f}min -> {KNN_CACHE}", flush=True)
    dist.barrier()


def main():
    dist.init_process_group('nccl'); lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV = torch.device(f'cuda:{lr}'); main = (lr == 0); world = dist.get_world_size()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    X = raw_matrix(CFG); IN = X.shape[1]
    if main:
        print(f"[{CFG['name']}] DDP world={world} method=listwise rows={X.shape[0]:,} train_n={TRAIN_N:,} "
              f"dim={IN} prefixes={PREFIXES} emb={EMB} epochs={EPOCHS} B={B} tau={TAU} smoke={SMOKE}", flush=True)
    if not os.path.exists(KNN_CACHE):
        mine_knn(X, lr, world, DEV, main)
    knn = np.load(KNN_CACHE)
    if main: print(f"corpus kNN {knn.shape} loaded", flush=True)

    model = DDP(Net(IN).to(DEV), device_ids=[lr])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ds = PairDS(knn); sampler = DistributedSampler(ds, shuffle=True, seed=SEED)
    loader = DataLoader(ds, batch_size=B, sampler=sampler, num_workers=4, drop_last=True, pin_memory=False)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); best = 1e9; t0 = time.time()
    spe = len(loader)
    if main: print(f"corpus pairs={len(ds):,} epochs={EPOCHS} steps/epoch={spe}", flush=True)
    for ep in range(1, EPOCHS + 1):
        model.train(); sampler.set_epoch(ep); tl = st = 0; te = time.time()
        for ai, pi in loader:
            ids = torch.cat([ai, pi]).numpy()
            rb = torch.from_numpy(rows(X, ids)).to(DEV, non_blocking=True)
            xb = gnorm(rb)
            z = model(xb)
            with torch.no_grad(): tw = raw_wj(rb)
            loss = matry_listwise(z, tw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item(); st += 1
            if main and (st % 200 == 0 or st == 1 or (SMOKE and st % 20 == 0)):
                el = time.time() - te
                print(f"  ep{ep:02d} step {st}/{spe} loss={tl/st:.5f} ({el/60:.1f}min, epoch ETA "
                      f"{el/st*(spe-st)/60:.1f}min)", flush=True)
            if MAX_STEPS and st >= MAX_STEPS:
                break
        sch.step()
        if main:
            print(f"ep{ep:02d} listwise_ce={tl/st:.5f} {(time.time()-t0)/60:.1f}min", flush=True)
            if tl / st < best: best = tl / st; torch.save(model.module.state_dict(), CKPT)
    dist.barrier()
    if main: print(f"TRAIN DONE {CFG['name']} best_ce={best:.5f} ckpt={CKPT} total {(time.time()-t0)/60:.1f}min", flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
