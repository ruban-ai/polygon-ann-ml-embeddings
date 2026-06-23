import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# nb28 — Triplet+Recon Autoencoder, **dim = 2048** (full corpus)

**Experiment:** Buddhi's "go bigger" hypothesis. The 512-d triplet+recon model flatlines on the
full corpus. Does pushing the embedding to **2048-d** (toward the ~3k raw range) restore Stage-1
recall into the mid-80s/90s? If the pattern holds, the bottleneck is *capacity*, not the objective.

**Setup (deliberate):**
- **All visible GPUs via DataParallel** — the 18,220-d input/output matmuls dominate cost and
  parallelize cleanly (vecs on cuda:0; the WJ loss is gathered on cuda:0). Buddhi's load is gone.
- **cdist L1-identity WJ loss** (`WJ=(2-L1)/(2+L1)` for simplex vecs) so batch=2048 @ 2048-d
  fits one GPU with no `(B,B,dim)` tensor, no checkpointing — exact, not approximate.
- **150 nmslib threads** (cores reclaimed) → eval QPS comparable to the other 150-thread rows.
- Same architecture / objective as nb28 base (margin triplet + 0.1·recon), only `OUT_DIM` changes.
- **`max_pos=30`** (NOT 256). A v1 of this run used 256 (from the InfoNCE runner) and the
  hard-margin triplet *collapsed* (loss pinned at margin, identical embeddings) — broad
  positives make the margin unsatisfiable. 30 = the original healthy nb28 setting.

**Caveat for the CSV:** QPS here is at 34 threads, so it is **NOT** comparable to the 150-thread
rows already in `NEW_RESULTS.csv`. Recall is thread-independent — that's the signal we care about.
"""))

cells.append(nbf.v4.new_code_cell(
"""import sys, os, time, csv, datetime, random
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

# ---- single-GPU, 35-core budget ----
N_CORES   = 150                   # buddhi's job done -> reclaim cores
THREADS   = N_CORES               # nmslib build/query threads; matches the 150-thread rows
torch.set_num_threads(THREADS)
DEVICE    = torch.device("cuda:0")  # single GPU: vectors, model, embeddings all live here

DATASET      = "full"
OUT_DIM      = 2048               # <-- the experiment
LOSS         = "triplet"          # margin triplet + recon (nb28 base objective)
EPOCHS       = 75     # match the original healthy 512-d recipe (max_pos=30 -> ~570 steps/ep)
BATCH_SIZE   = 2048               # same as 512-d run; the B*B*OUT_DIM cross-WJ term is
                                  # chunk+checkpointed in the loss so it still fits one GPU
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_POS      = 30     # CRITICAL: triplet hard-margin needs TIGHT positives. max_pos=256
                      # (carried over from the InfoNCE runner) made the objective
                      # unsatisfiable -> embedding collapse. Original healthy nb28 uses 30.
MARGIN       = 0.3
LAMBDA_RECON = 0.1
EF_SEARCH    = 200
RERANK_BATCH = 64
CAND_KS      = [1000, 2000]
SEED         = 42

METHOD   = f"triplet+recon(nb28-d{OUT_DIM})"
CKPT     = f"/tmp/best_triplet_autoencoder_wj_{OUT_DIM}_{DATASET}.pt"
CSV_PATH = "/raid/ruban/hpmlproj/term_project/SigSpatial/NEW_RESULTS.csv"
TTY      = sys.stdout.isatty()    # tqdm only when interactive (clean background logs)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
print(f"DEVICE={DEVICE} OUT_DIM={OUT_DIM} LOSS={LOSS} EPOCHS={EPOCHS} THREADS={THREADS}", flush=True)
"""))

cells.append(nbf.v4.new_code_cell(
"""# ---- WJ losses, model, dataset ----
def cross_wj_l1(a, b):
    # Embeddings are L1-simplex (row sum = 1), so WJ(a,b) = (2 - L1)/(2 + L1) EXACTLY.
    # torch.cdist computes pairwise L1 in a fused kernel that outputs only the (B,B) matrix
    # -- it never materializes the (B,B,OUT_DIM) tensor in the forward, so batch=2048 @
    # 2048-d fits on one GPU with no checkpointing and no bf16. Verified bit-identical to
    # the explicit min/max WJ (max abs err 0.0); fwd+bwd peak ~32GB at B=D=2048.
    l1 = torch.cdist(a, b, p=1)
    return (2.0 - l1) / (2.0 + l1)

def wj_triplet_inbatch(anchors, positives, margin=MARGIN, gt_matrix=None):
    sim = cross_wj_l1(anchors, positives)        # (B,B) exact WJ
    sim_ap = sim.diagonal()                       # WJ(anchor_i, its paired positive_i)
    sim_cross = sim.clone()
    sim_cross.fill_diagonal_(-1e9)
    if gt_matrix is not None:
        m = gt_matrix.to(sim_cross.device)
        if m.any():
            sim_cross[m] = -1e9                    # don't treat true neighbors as negatives
    sim_an = sim_cross.max(1).values              # hardest in-batch negative
    loss = F.relu(sim_an - sim_ap + margin)
    viol = loss > 0
    if viol.sum() == 0:
        return torch.tensor(0.0, device=anchors.device, requires_grad=True), 0.0
    return loss[viol].mean(), float(viol.float().mean().item())

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
        self.pairs = []
        for qid, neigh in gt_lookup.items():
            for nid in neigh[:max_pos]:
                if qid >= query_start and nid < query_start:
                    self.pairs.append((qid, nid))
        random.shuffle(self.pairs)
        print(f"pairs={len(self.pairs):,} steps/epoch~{len(self.pairs)//BATCH_SIZE}", flush=True)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]
"""))

cells.append(nbf.v4.new_code_cell(
"""# ---- load data onto the single GPU ----
qt, gt, query_start, corpus_qt, query_qt, corpus_sums, qt_norm = load_dataset_normalized(DATASET)
print(f"corpus={query_start:,} queries={len(query_qt):,} in_dim={qt_norm.shape[1]}", flush=True)

vecs_gpu = torch.from_numpy(np.ascontiguousarray(qt_norm, dtype=np.float32)).to(DEVICE)
print(f"vecs_gpu {vecs_gpu.nbytes/1024**3:.2f} GB on {DEVICE}", flush=True)

gt_stacked = build_gt_cache(gt, len(qt_norm), query_start, DATASET)
gt_gpu = build_gt_gpu(gt_stacked, DEVICE)
del gt_stacked
"""))

cells.append(nbf.v4.new_code_cell(
"""# ---- train ----
def module(m):
    return m.module if hasattr(m, "module") else m

def train():
    model = TripletAE(qt_norm.shape[1], OUT_DIM).to(DEVICE)
    if torch.cuda.device_count() > 1:                  # use ALL visible GPUs (matmuls dominate)
        model = nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
    ds = IndexAnchorPositiveDataset(gt, query_start, MAX_POS)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
                        pin_memory=False, drop_last=True, persistent_workers=True)
    n_steps = len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    best = float("inf"); t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tl = tm = tr = tv = st = 0
        pbar = tqdm(loader, desc=f"ep{ep:02d}/{EPOCHS}", ncols=110, mininterval=10)
        for a_ids, p_ids in pbar:
            a = vecs_gpu[a_ids.to(DEVICE)]
            p = vecs_gpu[p_ids.to(DEVICE)]
            b = a.shape[0]
            z, xrec = model(torch.cat([a, p]))
            za, zp = z[:b], z[b:]
            fn = build_fn_mask(a_ids, p_ids, gt_gpu, query_start)
            main, viol = wj_triplet_inbatch(za, zp, gt_matrix=fn)
            rec = F.mse_loss(xrec[:b], a)
            loss = main + LAMBDA_RECON * rec
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); tm += main.item(); tr += rec.item(); tv += viol; st += 1
            pbar.set_postfix(loss=f"{tl/st:.4f}", viol=f"{tv/st:.3f}")
        sch.step()
        avg = tl / max(st, 1); el = (time.time() - t0) / 60; eta = el / ep * (EPOCHS - ep)
        print(f"ep{ep:02d}/{EPOCHS} loss={avg:.4f} main={tm/st:.4f} rec={tr/st:.4f} "
              f"viol={tv/st:.3f} {el:.1f}min eta={eta:.1f}min", flush=True)
        if avg < best:
            best = avg
            torch.save(module(model).state_dict(), CKPT)
            print(f"  -> saved {CKPT}", flush=True)
    return model

model = train()
"""))

cells.append(nbf.v4.new_code_cell(
"""# ---- eval (paper protocol: WJ HNSW + exact GPU rerank) + log to NEW_RESULTS.csv ----
@torch.no_grad()
def embed_all(model, bs=512):
    enc = module(model); enc.eval(); out = []
    for s in range(0, len(qt_norm), bs):
        x = torch.tensor(qt_norm[s:s+bs], dtype=torch.float32, device=DEVICE)
        out.append(enc.encode(x).cpu().numpy().astype(np.float32))
    return np.vstack(out)

module(model).load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
embs = embed_all(model)
ce, qe = embs[:query_start], embs[query_start:]

max_k = max(max(CAND_KS), 500)
nbrs, info = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=max_k,
                              threads=THREADS, query_params={"efSearch": EF_SEARCH})
base = eval_recall(gt, nbrs, query_start, max_k)
print(f"[base] R@10={base[10]:.4f} R@50={base[50]:.4f} R@100={base[100]:.4f} "
      f"R@500={base[500]:.4f} HNSW_QPS={info['qps']:.0f}", flush=True)
rows = [["base", "", base[10], base[50], base[100], base[500], round(info['qps'])]]

preload_rerank_corpus(corpus_qt, corpus_sums)
for ck in CAND_KS:
    cand, ci = nmslib_neighbors(ce, qe, space="WeightedJaccard", k=ck,
                                threads=THREADS, query_params={"efSearch": EF_SEARCH})
    t0 = time.time()
    rr = rerank_wj_gpu(query_qt, cand, corpus_qt, corpus_sums, top_k=ck, batch_size=RERANK_BATCH)
    e2e = len(query_qt) / max(ci["query_s"] + (time.time() - t0), 1e-9)
    m = eval_recall(gt, rr, query_start, ck)
    print(f"[rerank K={ck}] R@10={m[10]:.4f} R@50={m[50]:.4f} R@100={m[100]:.4f} "
          f"R@500={m[500]:.4f} e2eQPS={e2e:.0f}", flush=True)
    rows.append(["rerank", ck, m[10], m[50], m[100], m[500], round(e2e)])
release_rerank_corpus()

today = datetime.date.today().isoformat()
note = f"single-GPU cdist-WJ-loss bs={BATCH_SIZE} max_pos={MAX_POS} ep={EPOCHS}; {THREADS} threads; QPS comparable to 150-thread rows"
with open(CSV_PATH, "a", newline="") as f:
    w = csv.writer(f)
    for stage, ck, r10, r50, r100, r500, qps in rows:
        w.writerow([today, DATASET, METHOD, OUT_DIM, stage, ck,
                    round(r10, 4), round(r50, 4), round(r100, 4), round(r500, 4),
                    qps, "28_triplet_autoencoder_wj_2048.ipynb", note])
print(f"appended {len(rows)} rows -> {CSV_PATH}", flush=True)
"""))

nb["cells"] = cells
out = "/raid/ruban/hpmlproj/term_project/SigSpatial/28_triplet_autoencoder_wj_2048.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
