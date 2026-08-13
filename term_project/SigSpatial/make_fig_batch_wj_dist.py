#!/usr/bin/env python3
"""Similarity-score distribution figure for Review 2's ask: what does the WJ
distillation loss (Eq. 5) actually regress against inside a training batch?

Replicates real batch construction (anchor + its top-30 WJ nearest neighbours,
as used by run_matdistill_fulleval_ddp.py / run_50k_ddp.py) on the 10k corpus,
forms the true NxN WJ matrix T for one such batch, and plots its distribution
split into: diagonal (self-pairs, T_ii=1, trivial), the curated anchor<->neighbour
pairs, and the remaining (uncurated) cross-pairs.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
qt = np.load('/tmp/qt_10k.npy')
corpus = qt[:8000]  # 10k benchmark: 8k corpus / 2k queries

# corpus self-kNN (top-30 by raw WJ), matching MAX_POS=30 used in training
sample_idx = np.random.choice(corpus.shape[0], 512, replace=False)  # one batch's worth of anchors
anchors = corpus[sample_idx]


def wj(a, b):
    mn = np.minimum(a[:, None, :], b[None, :, :]).sum(-1)
    mx = np.maximum(a[:, None, :], b[None, :, :]).sum(-1)
    return mn / np.clip(mx, 1e-10, None)


# find each anchor's true top-30 neighbour in the corpus (excluding itself)
sims_to_corpus = wj(anchors, corpus)  # (512, 8000) -- fine for this one-off figure
np.put_along_axis(sims_to_corpus, sample_idx[:, None], -1, axis=1)  # exclude self
top30 = np.argpartition(-sims_to_corpus, 30, axis=1)[:, :30]
positive_idx = top30[:, 0]  # one anchor-positive pair per anchor, as in training

batch_idx = np.concatenate([sample_idx, positive_idx])
batch = qt[batch_idx]
T = wj(batch, batch)
N = T.shape[0]

diag_mask = np.eye(N, dtype=bool)
# "curated" = the anchor<->its mined positive (both directions) and anchor<->itself's
# neighbour set overlap; approximate as: pairs (i, i+B) and (i+B, i) for the B anchor/positive split
B = len(sample_idx)
curated_mask = np.zeros((N, N), dtype=bool)
for i in range(B):
    curated_mask[i, B + i] = True
    curated_mask[B + i, i] = True
other_mask = ~diag_mask & ~curated_mask

diag_vals = T[diag_mask]
curated_vals = T[curated_mask]
other_vals = T[other_mask]

print(f"diag: n={diag_vals.size} mean={diag_vals.mean():.3f}")
print(f"curated anchor-positive: n={curated_vals.size} mean={curated_vals.mean():.3f}")
print(f"other in-batch cross-pairs: n={other_vals.size} mean={other_vals.mean():.4f} "
      f"median={np.median(other_vals):.4f} frac<0.01={np.mean(other_vals<0.01):.3f}")

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
