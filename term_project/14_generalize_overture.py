# Test MLP generalization: Parks-trained model on Overture dataset
# No retraining — just inference on Overture quadtree vectors

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nmslib
import pickle
import time
import os
import glob
import re
from tqdm import tqdm

device = torch.device('cuda:0')

# ── Model (same as training) ─────────────────────────────────
class QuadtreeCompressorV1Fixed(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(),
            nn.Linear(4096,   1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024,    512, bias=False), nn.BatchNorm1d(512),
        )
    def forward(self, x):
        x = torch.log1p(x * 1e6)
        return self.net(x)

# ── Load Overture encodings ──────────────────────────────────
ENCODING_DIR = "/raid/ruban/encodings/overture-50k0.020/"
POLY_COUNT   = 50000
DATA_END     = 40000
QUERY_START  = 40000
QUERY_END    = 50000

print("Loading Overture encodings...")
t0 = time.time()

def load_encodings(encoding_dir, poly_count):
    files   = sorted(glob.glob(os.path.join(encoding_dir, "real_*.txt")),
                     key=lambda f: int(re.search(r'real_(\d+)\.txt', f).group(1)))
    vectors = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = np.array([float(x) for x in line.split()], dtype=np.float32)
                vectors.append(vals)
                if len(vectors) >= poly_count:
                    break
        if len(vectors) >= poly_count:
            break
    return np.vstack(vectors)

qt_overture = load_encodings(ENCODING_DIR, POLY_COUNT)
print(f"Loaded: {qt_overture.shape} in {time.time()-t0:.1f}s")

D = qt_overture.shape[1]
print(f"Overture D = {D}")

# ── Load Parks-trained model ─────────────────────────────────
print("\nLoading Parks-trained model...")
model = QuadtreeCompressorV1Fixed(in_dim=D, out_dim=512).to(device)

try:
    model.load_state_dict(
        torch.load('/tmp/best_compressor_full_fixed.pt', weights_only=True))
    print("Loaded best_compressor_full_fixed.pt")
except Exception as e:
    print(f"Error: {e}")
    # Try the 10k model
    model2 = QuadtreeCompressorV1Fixed(in_dim=D, out_dim=512).to(device)
    model2.load_state_dict(
        torch.load('/tmp/best_compressor_v1_clean.pt', weights_only=True))
    model = model2
    print("Loaded best_compressor_v1_clean.pt")

model.eval()
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── Generate embeddings ──────────────────────────────────────
print("\nGenerating embeddings...")
t0 = time.time()

def generate_embeddings(model, vectors, batch_size=512):
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(vectors), batch_size), desc="Embedding"):
            batch = torch.tensor(vectors[start:start+batch_size],
                                 dtype=torch.float32).to(device)
            out   = model(batch)
            emb   = F.normalize(out, dim=1)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)

embs = generate_embeddings(model, qt_overture)
print(f"Embeddings: {embs.shape} in {time.time()-t0:.1f}s")

corpus_embs = embs[:DATA_END]
query_embs  = embs[QUERY_START:QUERY_END]

# ── Build HNSW cosine index ──────────────────────────────────
print("\nBuilding HNSW cosine index...")
t0  = time.time()
idx = nmslib.init(method='hnsw', space='cosinesimil')
for i in range(len(corpus_embs)):
    idx.addDataPoint(i, corpus_embs[i])
idx.createIndex({'M': 20, 'efConstruction': 200, 'post': 1},
                print_progress=False)
idx.setQueryTimeParams({'efSearch': 200})
build_time = time.time() - t0
print(f"Index built in {build_time:.1f}s")

# ── Query ────────────────────────────────────────────────────
print("\nQuerying...")
t0   = time.time()
nbrs = idx.knnQueryBatch(query_embs, k=500, num_threads=32)
qps  = len(query_embs) / (time.time() - t0)
print(f"QPS: {qps:.1f}")

# ── Load Overture GT ─────────────────────────────────────────
print("\nLoading Overture GT...")
GT_DIR = "/raid/ruban/groundtruth/overture-query-50k/"

gt = {}
for fname in sorted(os.listdir(GT_DIR)):
    fpath = os.path.join(GT_DIR, fname)
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split(', ')
            if len(parts) < 2:
                continue
            qid = int(parts[0])
            if QUERY_START <= qid < QUERY_END:
                gt[qid] = [int(x) for x in parts[1:]]

print(f"GT loaded: {len(gt):,} queries")

# ── Compute Recall ───────────────────────────────────────────
def recall_at_k(gt, nbrs, query_start, K):
    total = 0.0
    count = 0
    for i, (ids, _) in enumerate(nbrs):
        qid    = query_start + i
        gt_set = set(gt.get(qid, [])[:K])
        if not gt_set:
            continue
        total += len(gt_set & set(ids[:K])) / len(gt_set)
        count += 1
    return total / count if count > 0 else 0.0

print("\n=== Generalization Results (Parks model → Overture data) ===")
for K in [10, 50, 500]:
    r = recall_at_k(gt, nbrs, QUERY_START, K)
    print(f"R@{K:<4} : {r:.4f}")
print(f"QPS    : {qps:.1f}")
print(f"Build  : {build_time:.1f}s")
print(f"Emb dim: 512")