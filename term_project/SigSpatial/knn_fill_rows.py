#!/usr/bin/env python3
"""Recompute one missing kNN shard (rows [R0, R1) of the training corpus) split across
the GPUs given to torchrun, with exactly the same tiled exact-L1-on-simplex search as
run_fulls_listwise_train.mine_knn (used when one rank of the main job is starved by
another user's process on its GPU). Output: knn_<cfg>_shard_<SHARD>.npy, the same file
the main job would have written.
Launch: torchrun --standalone --nproc_per_node=N knn_fill_rows.py <cfg> <R0> <R1> <SHARD>"""
import sys, os, time
import numpy as np, torch, torch.distributed as dist
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS, BASE, raw_matrix, rows

CFG = CONFIGS[sys.argv[1]]; R0, R1, SHARD = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
MAX_POS = 30; TILE = 250_000; TRAIN_N = CFG['train_n']


def gnorm(x): x = x.clamp(min=0); return x / x.sum(1, keepdim=True).clamp(min=1e-10)


@torch.no_grad()
def main():
    dist.init_process_group('nccl'); lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    DEV = torch.device(f'cuda:{lr}'); world = dist.get_world_size(); main_ = lr == 0; t0 = time.time()
    X = raw_matrix(CFG)
    per = (R1 - R0 + world - 1) // world; s0 = R0 + lr * per; s1 = min(s0 + per, R1); nq = max(s1 - s0, 0)
    Q = gnorm(torch.from_numpy(rows(X, np.arange(s0, s1))).to(DEV))
    bestd = torch.full((nq, MAX_POS), float('inf'), device=DEV); besti = torch.zeros((nq, MAX_POS), dtype=torch.long, device=DEV)
    ntiles = (TRAIN_N + TILE - 1) // TILE
    if main_: print(f"fill rows [{R0},{R1}) of {CFG['name']} over {world} GPUs, per-rank={per:,}, tiles={ntiles}", flush=True)
    for t, c0 in enumerate(range(0, TRAIN_N, TILE)):
        c1 = min(c0 + TILE, TRAIN_N)
        C = gnorm(torch.from_numpy(rows(X, np.arange(c0, c1))).to(DEV))
        for i in range(0, nq, 2048):
            j = min(i + 2048, nq); d = torch.cdist(Q[i:j], C, p=1)
            gi = torch.arange(s0 + i, s0 + j, device=DEV); m = (gi >= c0) & (gi < c1)
            if m.any():
                r = torch.nonzero(m).squeeze(1); d[r, gi[r] - c0] = float('inf')
            vd, vi = torch.topk(d, min(MAX_POS, c1 - c0), dim=1, largest=False)
            cd = torch.cat([bestd[i:j], vd], 1); ci = torch.cat([besti[i:j], vi + c0], 1)
            o = torch.topk(cd, MAX_POS, dim=1, largest=False).indices
            bestd[i:j] = torch.gather(cd, 1, o); besti[i:j] = torch.gather(ci, 1, o)
        del C; torch.cuda.empty_cache()
        if main_:
            el = time.time() - t0
            print(f"  tile {t+1}/{ntiles} ({el/60:.1f}min, ETA {el/(t+1)*(ntiles-t-1)/60:.1f}min)", flush=True)
    np.save(f'{BASE}/knnfill_{CFG["name"]}_{SHARD}_part{lr}.npy', besti.cpu().numpy()); dist.barrier()
    if main_:
        parts = [np.load(f'{BASE}/knnfill_{CFG["name"]}_{SHARD}_part{r}.npy') for r in range(world)]
        out = np.concatenate(parts, 0); assert out.shape == (R1 - R0, MAX_POS), out.shape
        np.save(f'{BASE}/knn_{CFG["name"]}_shard_{SHARD}.npy', out)
        for r in range(world): os.remove(f'{BASE}/knnfill_{CFG["name"]}_{SHARD}_part{r}.npy')
        print(f"FILL DONE shard {SHARD} {out.shape} {(time.time()-t0)/60:.1f}min", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
