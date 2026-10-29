#!/usr/bin/env python3
"""2026-10-10 signature precision (Buddhi: 8-bit is an adaptation of ShapeToVec+, add plot + discussion).
Parks, 3K-dimensional input (parksres_3k, widths 64..4096), R@50 on all held-out queries, HNSW M=20 efS=200.
float32: logs_20261009/eval_3k.log; 8-bit: logs_20261009/quant_3k.log; 4/2-bit: logs_20261010/quant_parks3k_lowbits.log.
Writes IEEE/fig_precision.{png,pdf}."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dims = [64, 128, 256]
r = {32: [0.8552, 0.8798, 0.8978], 8: [0.8529, 0.8781, 0.8967], 4: [0.6634, 0.7412, 0.8051], 2: [0.2924, 0.4212, 0.5480]}
sty = {32: ('float32', '#6a8fd1', '--o'), 8: ('8-bit', '#1f4e9c', '-o'), 4: ('4-bit', '#d98c1f', '-s'), 2: ('2-bit', '#b22222', '-^')}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for b in (32, 8, 4, 2):
    lab, c, m = sty[b]
    xs = [d * b // 8 for d in dims]
    ax.plot(xs, r[b], m, color=c, lw=2 if b == 8 else 1.5, ms=5, label=lab)
    if b in (8, 2):
        for x, y, d in zip(xs, r[b], dims):
            ax.annotate(f'{d}-d', (x, y), textcoords='offset points', xytext=(0, 6 if b == 8 else -12), ha='center', fontsize=7, color=c)
ax.axvline(64, color='black', lw=0.8, ls=':')
ax.text(66, 0.33, '64 B', fontsize=8)
ax.set_xscale('log', base=2)
ax.set_xticks([16, 32, 64, 128, 256, 512, 1024]); ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1K'], fontsize=9)
ax.set_xlabel('Bytes per polygon (log scale)', fontsize=10); ax.set_ylabel('Recall@50', fontsize=10)
ax.set_ylim(0.25, 0.95)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=8, loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_precision.png', dpi=200)
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_precision.pdf')
print('saved IEEE/fig_precision')
