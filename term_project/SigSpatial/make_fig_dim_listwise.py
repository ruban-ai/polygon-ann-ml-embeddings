#!/usr/bin/env python3
"""Regenerates fig_dim_distill.png with listwise (now the primary method) numbers
instead of MSE. Data from NEW_RESULTS.csv, 50k-listwise-d* base rows."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dims = [256, 512, 1024, 2048, 4096]
r500 = [0.9694, 0.9713, 0.9722, 0.9728, 0.9729]
qps = [4921, 2682, 1958, 1631, 1493]

fig, ax1 = plt.subplots(figsize=(5.2, 3.2))
color1 = '#1f5fa8'
ax1.set_xlabel('Embedding dimension $d$')
ax1.set_ylabel('Stage-1 R@500', color=color1)
ax1.plot(dims, r500, 'o-', color=color1, linewidth=2, markersize=6)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log', base=2)
ax1.set_xticks(dims)
ax1.set_xticklabels([str(d) for d in dims])
ax1.set_ylim(0.95, 0.98)

ax2 = ax1.twinx()
color2 = '#c0392b'
ax2.set_ylabel('HNSW QPS', color=color2)
ax2.plot(dims, qps, 's--', color=color2, linewidth=2, markersize=6)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 5500)

fig.tight_layout()
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/full_paper/fig_dim_distill.png', dpi=200)
print("saved fig_dim_distill.png")
