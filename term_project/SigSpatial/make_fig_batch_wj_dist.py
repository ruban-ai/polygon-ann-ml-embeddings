#!/usr/bin/env python3
"""Similarity-score distribution figure for Review 2's ask: what does the WJ
distillation loss (Eq. 5) actually regress against inside a training batch?

Replicates real batch construction (anchor + its top-30 WJ nearest neighbours,
as used by run_matdistill_fulleval_ddp.py / run_50k_ddp.py) on the 10k corpus,
forms the true NxN WJ matrix T for one such batch, and plots its distribution
split into curated anchor<->positive pairs vs. the remaining (uncurated) cross-pairs.
GPU + chunked to avoid materializing huge intermediate tensors.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV = torch.device('cuda:0')
np.random.seed(42)
qt = np.load('/tmp/qt_10k.npy')
corpus = torch.from_numpy(qt[:8000]).to(DEV)  # 10k benchmark: 8k corpus / 2k queries


def wj_chunked(a, b, chunk=64):
    """Chunked pairwise WJ, a:(Na,D) b:(Nb,D) -> (Na,Nb)."""
    out = torch.empty(a.shape[0], b.shape[0], device=a.device)
    for i in range(0, a.shape[0], chunk):
        ai = a[i:i + chunk]
        mn = torch.minimum(ai.unsqueeze(1), b.unsqueeze(0)).sum(2)
        mx = torch.maximum(ai.unsqueeze(1), b.unsqueeze(0)).sum(2).clamp(min=1e-10)
        out[i:i + chunk] = mn / mx
    return out


B = 512
sample_idx = np.random.choice(corpus.shape[0], B, replace=False)
anchors = corpus[sample_idx]

sims_to_corpus = wj_chunked(anchors, corpus)  # (512, 8000), chunked -> fine on GPU
for row, aid in zip(sims_to_corpus, sample_idx):
    row[aid] = -1  # exclude self
top1 = sims_to_corpus.argmax(1).cpu().numpy()  # top-1 of the top-30 pool, one positive per anchor

batch_idx = np.concatenate([sample_idx, top1])
batch = corpus[batch_idx]
T = wj_chunked(batch, batch).cpu().numpy()
N = T.shape[0]

diag_mask = np.eye(N, dtype=bool)
curated_mask = np.zeros((N, N), dtype=bool)
for i in range(B):
    curated_mask[i, B + i] = True
    curated_mask[B + i, i] = True
other_mask = ~diag_mask & ~curated_mask

curated_vals = T[curated_mask]
other_vals = T[other_mask]

print(f"curated anchor-positive: n={curated_vals.size} mean={curated_vals.mean():.3f}")
print(f"other in-batch cross-pairs: n={other_vals.size} mean={other_vals.mean():.4f} "
      f"median={np.median(other_vals):.4f} frac<0.01={np.mean(other_vals < 0.01):.3f}")

fig, ax = plt.subplots(figsize=(4.6, 3.0))
bins = np.linspace(0, 1, 51)
ax.hist(other_vals, bins=bins, weights=np.ones_like(other_vals) / len(other_vals),
        alpha=0.75, color='#4C72B0', label=f'other cross-pairs (n={other_vals.size:,})')
ax.hist(curated_vals, bins=bins, weights=np.ones_like(curated_vals) / len(curated_vals),
        alpha=0.85, color='#DD8452', label=f'anchor--positive pairs (n={curated_vals.size})')
ax.set_yscale('log')
ax.set_xlabel('true WJ similarity $T_{ij}$')
ax.set_ylabel('fraction of pairs (log scale)')
ax.legend(fontsize=8, loc='upper right')
ax.set_title('In-batch WJ target distribution (10k, one batch)', fontsize=9)
fig.tight_layout()
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/full_paper/fig_batch_wj_dist.png', dpi=200)
print("saved fig_batch_wj_dist.png")
