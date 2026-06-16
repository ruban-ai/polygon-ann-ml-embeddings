#!/usr/bin/env python3
"""InfoNCE (recall-oriented candidate generator) at OUT_DIM=2048 on the full corpus.
Mirrors the InfoNCE-512 baseline (max_pos=256, temp=0.07, +0.1 recon, 18 epochs) so the
2048-vs-512 dimension comparison is fair. Uses the exact cdist WJ identity for the in-batch
loss (memory-safe at 2048-d, bit-identical to min/max). DataParallel over all visible GPUs.
Appends base + rerank rows to NEW_RESULTS.csv."""
import sys, os, time, csv, datetime, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (
    build_fn_mask, build_gt_cache, build_gt_gpu, eval_recall,
    load_dataset_normalized, nmslib_neighbors, preload_rerank_corpus,
    release_rerank_corpus, rerank_wj_gpu,
)

THREADS = 150
torch.set_num_threads(THREADS)
DEVICE = torch.device("cuda:0")

DATASET      = "full"
OUT_DIM      = 2048
EPOCHS       = 18
BATCH_SIZE   = 2048
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_POS      = 256        # InfoNCE tolerates broad positives (its design) -- NOT 30
TEMPERATURE  = 0.07
LAMBDA_RECON = 0.1
EF_SEARCH    = 200
RERANK_BATCH = 64
CAND_KS      = [1000, 2000]
SEED         = 42

METHOD   = f"InfoNCE(Recall-CG)-d{OUT_DIM}"
CKPT     = f"/tmp/best_filter_recall_infonce_{OUT_DIM}_{DATASET}.pt"
CSV_PATH = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def cross_wj_l1(a, b):
    # L1-simplex embeddings -> WJ = (2-L1)/(2+L1) exactly; cdist avoids the (B,B,dim) tensor.
    l1 = torch.cdist(a, b, p=1)
    return (2.0 - l1) / (2.0 + l1)


def wj_infonce_inbatch(anchors, positives, temperature=TEMPERATURE, gt_matrix=None):
    sim = cross_wj_l1(anchors, positives)          # (B,B), diagonal = paired positive
    logits = sim / temperature
    if gt_matrix is not None:
        mask = gt_matrix.to(logits.device).clone()
        mask.fill_diagonal_(False)                  # keep the designated positive
        logits = logits.masked_fill(mask, -1e9)     # drop a query's other true neighbors
    labels = torch.arange(anchors.shape[0], device=anchors.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
    return loss, acc


class TripletAE(nn.Module):
    def __init__(self, in_dim, out_dim=OUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, in_dim, bias=False),
        )
    def encode(self, x):
        z = F.relu(self.encoder(x))
        return z / z.sum(1, keepdim=True).clamp(min=1e-10)
    def forward(self, x):
        z = self.encode(x)
        return z, F.relu(self.decoder(z))


class IndexAnchorPositiveDataset(Dataset):
    def __init__(self, gt_lookup, query_start, max_pos=MAX_POS):
        self.pairs = [(q, n) for q, nb in gt_lookup.items()
                      for n in nb[:max_pos] if q >= query_start and n < query_start]
        random.shuffle(self.pairs)
        print(f"pairs={len(self.pairs):,} steps/epoch~{len(self.pairs)//BATCH_SIZE}", flush=True)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


def module(m): return m.module if hasattr(m, "module") else m


def main():
    qt, gt, qs, cq, qq, cs, qtn = load_dataset_normalized(DATASET)
    print(f"corpus={qs:,} queries={len(qq):,} in_dim={qtn.shape[1]} OUT_DIM={OUT_DIM}", flush=True)
    vecs = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEVICE)
    gtg = build_gt_gpu(build_gt_cache(gt, len(qtn), qs, DATASET), DEVICE)

    model = TripletAE(qtn.shape[1], OUT_DIM).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs", flush=True)

    ds = IndexAnchorPositiveDataset(gt, qs, MAX_POS)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
                        drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    best = float("inf"); t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tl = tm = tr = ta = st = 0
        for ai, pi in tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=110, mininterval=10):
            a = vecs[ai.to(DEVICE)]; p = vecs[pi.to(DEVICE)]; b = a.shape[0]
            z, xr = model(torch.cat([a, p])); za, zp = z[:b], z[b:]
            fn = build_fn_mask(ai, pi, gtg, qs)
            main_l, acc = wj_infonce_inbatch(za, zp, gt_matrix=fn)
            rec = F.mse_loss(xr[:b], a)
            loss = main_l + LAMBDA_RECON * rec
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); tm += main_l.item(); tr += rec.item(); ta += acc; st += 1
        sch.step()
        el = (time.time() - t0) / 60
        print(f"ep{ep:02d}/{EPOCHS} loss={tl/st:.4f} main={tm/st:.4f} rec={tr/st:.4f} "
              f"acc={ta/st:.3f} {el:.1f}min eta={el/ep*(EPOCHS-ep):.1f}min", flush=True)
        if tl/st < best:
            best = tl/st
            torch.save(module(model).state_dict(), CKPT)
            print(f"  -> saved {CKPT}", flush=True)

    # ---- eval ----
    enc = module(model); enc.eval()
    embs = []
    with torch.no_grad():
        for s in range(0, len(qtn), 512):
            x = torch.tensor(qtn[s:s+512], dtype=torch.float32, device=DEVICE)
            embs.append(enc.encode(x).cpu().numpy().astype(np.float32))
    embs = np.vstack(embs); ce, qe = embs[:qs], embs[qs:]
    max_k = max(max(CAND_KS), 500)
    nbrs, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=max_k,
                                  threads=THREADS, query_params={"efSearch": EF_SEARCH})
    base = eval_recall(gt, nbrs, qs, max_k)
    print(f"[base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@100={base[100]:.4f} "
          f"R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
    rows = [["base", "", base[10], base[50], base[100], base[500], round(info['qps'])]]
    preload_rerank_corpus(cq, cs)
    for ck in CAND_KS:
        cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck,
                                    threads=THREADS, query_params={"efSearch": EF_SEARCH})
        t1 = time.time()
        rr = rerank_wj_gpu(qq, cand, cq, cs, top_k=ck, batch_size=RERANK_BATCH)
        e2e = len(qq) / max(ci["query_s"] + (time.time()-t1), 1e-9)
        m = eval_recall(gt, rr, qs, ck)
        print(f"[rerank K={ck}] R@10={m[10]:.4f} R@50={m[50]:.4f} R@100={m[100]:.4f} "
              f"R@500={m[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
        rows.append(["rerank", ck, m[10], m[50], m[100], m[500], round(e2e)])
    release_rerank_corpus()

    today = datetime.date.today().isoformat()
    note = f"single-process DataParallel cdist-WJ-loss bs={BATCH_SIZE} max_pos={MAX_POS} ep={EPOCHS}; {THREADS} threads"
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        for stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, DATASET, METHOD, OUT_DIM, stage, ck,
                        round(r10,4), round(r50,4), round(r100,4), round(r500,4),
                        qps, "run_infonce_2048.py", note])
    print(f"appended {len(rows)} rows -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
