#!/usr/bin/env python3
"""2026-10-10: 18K dropped, x = input dimensions, full method name (Buddhi). 2026-10-09 input-resolution study (Buddhi's 10-09 request; cf. ShapeToVec+ spatial-depth figure).
Recall@50 on Parks vs ShapeToVec input grid resolution, one encoder per resolution, identical recipe
(widths 64..4096, 5 epochs, 7-GPU DDP). 3K/6K/12K: logs_20261009/quant_{3k,6k,12k}.log (8-bit).
18K: logs_20261009/quant_18k.log (7-GPU rerun, parksres_18k; identical recipe to 3K/6K/12K).
ShapeToVec float32 at 3K/6K/12K as published (96/96/97%). Writes IEEE/fig_resolution.png."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res = [3, 6, 12]                                 # thousand dimensions (3,043 / 6,004 / 12,061 / 18,220)
ours256 = [0.8967, 0.8975, 0.8983]           # 256-d, 8-bit = 256 B
ours64 = [0.8529, 0.8546, 0.8540]            # 64-d, 8-bit = 64 B
s2v = [(3, 0.96), (6, 0.96), (12, 0.97)]             # ShapeToVec float32: 12 KB / 24 KB / 48 KB
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(res, ours256, '-o', color='#1f4e9c', lw=2, ms=6, label='Listwise WJ distillation, 256 B')
ax.plot(res, ours64, '-o', color='#6a8fd1', lw=1.5, ms=5, label='Listwise WJ distillation, 64 B')
ax.plot([r for r, _ in s2v], [v for _, v in s2v], '-^', color='#555555', lw=1.5, ms=6, label='ShapeToVec, float32')
for (r, v), t in zip(s2v, ['12 KB', '24 KB', '48 KB']):
    ax.annotate(t, (r, v), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8, color='#555555')
ax.set_xticks(res); ax.set_xticklabels(['3K', '6K', '12K'], fontsize=9)
ax.set_xlabel('Input dimensions $D$', fontsize=10)
ax.set_ylabel('Recall@50', fontsize=10); ax.set_ylim(0.80, 1.0)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=8, loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_resolution.png', dpi=200)
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_resolution.pdf')   # 2026-10-10: vector PDF for the paper (Buddhi)
print('saved IEEE/fig_resolution.png')
