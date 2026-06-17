#!/usr/bin/env python3
"""Plain 2048-d WJ embedding (no Matryoshka): loss on full 2048-d output only.
Architecture matches Matryoshka wide trunk: 18220 -> 4096 -> 2048 (no 1024 funnel).
--loss triplet (max_pos=30) | infonce (max_pos=256)."""
import sys, time, csv, datetime, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from sota_experiment_common import (build_fn_mask, build_gt_cache, build_gt_gpu, eval_recall,
    load_dataset_normalized, nmslib_neighbors, preload_rerank_corpus, release_rerank_corpus, rerank_wj_gpu)

ap = argparse.ArgumentParser()
ap.add_argument('--loss', choices=['triplet', 'infonce'], required=True)
A = ap.parse_args()
DEV = torch.device('cuda:0')
THREADS = 120
torch.set_num_threads(THREADS)
OUT_DIM = 2048
BATCH = 1024
LR = 1e-3
WD = 1e-4
MARGIN = 0.3
TEMP = 0.07
LAM_REC = 0.1
EF = 200
RB = 64
CAND_KS = [1000, 2000]
SEED = 42
MAX_POS = 30 if A.loss == 'triplet' else 256
EPOCHS = 75 if A.loss == 'triplet' else 18
CKPT = f"/tmp/best_plain_{A.loss}_2048_full.pt"
CSV = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"plain-2048 loss={A.loss} max_pos={MAX_POS} epochs={EPOCHS} out_dim={OUT_DIM} batch={BATCH}", flush=True)


def norm(x):
    return x / x.sum(1, keepdim=True).clamp(min=1e-10)


def cross_wj(a, b):
    l1 = torch.cdist(a, b, p=1)
    return (2.0 - l1) / (2.0 + l1)


def triplet_loss(za, zb, gtm):
    if gtm is not None:
        gtm = gtm.to(za.device)
    sim = cross_wj(za, zb)
    ap = sim.diagonal()
    sc = sim.clone()
    sc.fill_diagonal_(-1e9)
    if gtm is not None and gtm.any():
        sc[gtm] = -1e9
    sa = sc.max(1).values
    loss = F.relu(sa - ap + MARGIN)
    v = loss > 0
    return (loss[v].mean() if v.any() else torch.tensor(0., device=za.device, requires_grad=True)), float(v.float().mean())


def infonce_loss(za, zb, gtm):
    logits = cross_wj(za, zb) / TEMP
    if gtm is not None:
        m = gtm.to(logits.device).clone()
        m.fill_diagonal_(False)
        logits = logits.masked_fill(m, -1e9)
    lab = torch.arange(za.shape[0], device=za.device)
    loss = F.cross_entropy(logits, lab)
    return loss, float((logits.argmax(1) == lab).float().mean())


LOSS_FN = triplet_loss if A.loss == 'triplet' else infonce_loss


class PlainAE(nn.Module):
    def __init__(self, d, out_dim=OUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096, out_dim, bias=False), nn.BatchNorm1d(out_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, d, bias=False),
        )

    def embed(self, x):
        return F.relu(self.encoder(x))

    def forward(self, x):
        z = self.embed(x)
        return z, F.relu(self.decoder(norm(z)))


def module(m):
    return m.module if hasattr(m, 'module') else m


class DS(Dataset):
    def __init__(self, gt, qsv):
        self.p = [(q, n) for q, nb in gt.items() for n in nb[:MAX_POS] if q >= qsv and n < qsv]
        random.shuffle(self.p)
        print(f"pairs={len(self.p):,} steps/epoch~{len(self.p) // BATCH}", flush=True)

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.p[i]


def main():
    qt, gt, qs, cq, qq, cs, qtn = load_dataset_normalized('full')
    vecs = torch.from_numpy(np.ascontiguousarray(qtn, dtype=np.float32)).to(DEV)
    gtg = build_gt_gpu(build_gt_cache(gt, len(qtn), qs, 'full'), DEV)
    model = PlainAE(qtn.shape[1]).to(DEV)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel {torch.cuda.device_count()} GPUs", flush=True)
    loader = DataLoader(DS(gt, qs), batch_size=BATCH, shuffle=True, num_workers=12,
                        drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    best = float('inf')
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl = tm = tr = tv = st = 0
        for ai, pi in tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=110, mininterval=10):
            a = vecs[ai.to(DEV)]
            p = vecs[pi.to(DEV)]
            b = a.shape[0]
            z, xr = model(torch.cat([a, p]))
            za, zb = norm(z[:b]), norm(z[b:])
            fn = build_fn_mask(ai, pi, gtg, qs)
            main_l, metric = LOSS_FN(za, zb, fn)
            rec = F.mse_loss(xr[:b], a)
            loss = main_l + LAM_REC * rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            tm += main_l.item()
            tr += rec.item()
            tv += metric
            st += 1
        sch.step()
        el = (time.time() - t0) / 60
        mlabel = 'viol' if A.loss == 'triplet' else 'acc'
        print(f"ep{ep:02d}/{EPOCHS} loss={tl/st:.4f} main={tm/st:.4f} rec={tr/st:.4f} {mlabel}={tv/st:.3f} {el:.1f}min eta={el/ep*(EPOCHS-ep):.1f}min", flush=True)
        if tl / st < best:
            best = tl / st
            torch.save(module(model).state_dict(), CKPT)
            print(f"  -> saved {CKPT}", flush=True)

    enc = module(model)
    enc.eval()
    with torch.no_grad():
        Z = torch.cat([
            norm(enc.embed(torch.tensor(qtn[i:i + 512], dtype=torch.float32, device=DEV)))
            for i in range(0, len(qtn), 512)
        ])
    emb = Z.cpu().numpy().astype(np.float32)
    ce, qe = emb[:qs], emb[qs:]
    max_k = max(max(CAND_KS), 500)
    nbrs, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=max_k,
                                  threads=THREADS, query_params={"efSearch": EF})
    base = eval_recall(gt, nbrs, qs, max_k)
    print(f"[base] R@50={base[50]:.4f} R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
    rows = [["base", "", base[10], base[50], base[100], base[500], round(info['qps'])]]
    preload_rerank_corpus(cq, cs)
    for ck in CAND_KS:
        cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck,
                                    threads=THREADS, query_params={"efSearch": EF})
        t1 = time.time()
        rr = rerank_wj_gpu(qq, cand, cq, cs, top_k=ck, batch_size=RB)
        e2e = len(qq) / max(ci["query_s"] + (time.time() - t1), 1e-9)
        mm = eval_recall(gt, rr, qs, ck)
        print(f"[rerank K={ck}] R@50={mm[50]:.4f} R@500={mm[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
        rows.append(["rerank", ck, mm[10], mm[50], mm[100], mm[500], round(e2e)])
    release_rerank_corpus()

    today = datetime.date.today().isoformat()
    note = (f"Plain {A.loss} max_pos={MAX_POS} 18220->4096->{OUT_DIM} single loss (no Matryoshka); "
            f"{THREADS} threads efSearch={EF}")
    method = f"Plain-{A.loss}-d{OUT_DIM}"
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        for stage, ck, r10, r50, r100, r500, qps in rows:
            w.writerow([today, "full", method, OUT_DIM, stage, ck,
                        round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4),
                        qps, "run_plain_2048.py", note])
    print(f"appended {len(rows)} rows -> {CSV}", flush=True)


if __name__ == "__main__":
    main()
