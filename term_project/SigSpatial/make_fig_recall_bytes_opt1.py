#!/usr/bin/env python3
"""2026-10-19 (Prasad): extend the 8-bit line to 4 KB and label the width of every point on both lines
(his confusion: at 256 B the 8-bit point is 256-d but the float32 point is 64-d).
Parks, 3K input (parksres_3k): 64-256 from logs_20261009/eval_3k.log + quant_3k.log;
512-4096 (float32 and 8-bit) from logs_20261019/quant_parks3k_wide.log. ShapeToVec+/ShapeToVec as published."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
dims = [64, 128, 256, 512, 1024, 4096]
r8  = [0.8529, 0.8781, 0.8967, 0.9084, 0.9148, 0.9200]
r32 = [0.8552, 0.8798, 0.8978, 0.9091, 0.9154, 0.9203]
s2vp = [(384, 0.76, '1-bit'), (768, 0.89, '2-bit'), (1536, 0.95, '4-bit'), (3072, 0.96, '8-bit')]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(dims, r8, '-o', color='#1f4e9c', lw=2, ms=5, label='Listwise WJ distillation, 8-bit')
ax.plot([b for b, _, _ in s2vp], [r for _, r, _ in s2vp], '-s', color='#c0392b', lw=1.5, ms=5, label='ShapeToVec+ (3K dims)')
ax.axhline(0.96, ls=':', color='#555555', lw=1.6, label='ShapeToVec, float32 (12 KB)')
for d, r in zip(dims, r8):
    dx = -14 if d == 1024 else 0
    ax.annotate(f'{d}-d', (d, r), textcoords='offset points', xytext=(dx, 5), ha='center', fontsize=7, color='#1f4e9c')
off = {'1-bit': (6, -9), '2-bit': (6, -10), '4-bit': (6, -9), '8-bit': (6, -9)}
for b, r, t in s2vp:
    ax.annotate(t, (b, r), textcoords='offset points', xytext=off[t], fontsize=7.5, color='#c0392b')
ax.set_xscale('log', base=2)
ax.set_xticks([64, 256, 1024, 4096, 16384]); ax.set_xticklabels(['64 B', '256 B', '1 KB', '4 KB', '16 KB'], fontsize=9)
ax.set_xlabel('Bytes per polygon (log scale)', fontsize=10)
ax.set_ylabel('Recall@50', fontsize=10); ax.set_ylim(0.72, 1.0)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=7, loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_recall_bytes_opt1.png', dpi=200)
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_recall_bytes_opt1.pdf')
print('saved IEEE/fig_recall_bytes.{png,pdf}')
