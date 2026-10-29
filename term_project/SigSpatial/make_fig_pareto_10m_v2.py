#!/usr/bin/env python3
"""2026-10-27: QPS = median of 20 runs after warm-up (bench_are_v2.py, Prasad item 7); old figure kept as fig_pareto_10m_mean5_20261020.pdf.
2026-10-20 (Prasad: accuracy-memory-throughput Pareto view). Overture 10M, 64 query threads, HNSW M=20 efS=200.
Ours: results_bench_are_ov10m_d{64,128,256,512,1024}.json (8-bit codes in ShapeToVec+'s own SIMD ARE kernel).
ShapeToVec+ (3K dims): R@50 and QPS read from its Fig. 3 at 10M; index memory scaled linearly from its 90M numbers
(1-bit 52 GB, 2-bit/2-level 91 GB, 4-bit 167 GB, ARE 320 GB). Writes IEEE/fig_pareto_10m.{png,pdf}."""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B = '/raid/ruban/hpmlproj/term_project/SigSpatial'
ours = []
for d in [64, 128, 256, 512, 1024]:
    r = json.load(open(f'{B}/results_bench_are_v2_ov10m_d{d}.json'))
    ours.append((d, r['index_bytes'] / 1e9, r['r50'], r['qps_median'] / 1e3))
s2vp = [('1-bit', 52/9, 0.55, 7.0), ('2-level', 91/9, 0.69, 4.7), ('2-bit', 91/9, 0.75, 2.9),
        ('4-bit', 167/9, 0.90, 1.2), ('ARE', 320/9, 0.91, 2.3)]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot([o[1] for o in ours], [o[2] for o in ours], '-o', color='#1f4e9c', lw=2, ms=5, label='Listwise WJ distillation (8-bit)')
for d, gb, r, q in ours:
    ax.annotate(f'{d}-d\n{q:.1f}K', (gb, r), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=6.5, color='#1f4e9c', linespacing=0.9)
ax.plot([s[1] for s in s2vp], [s[2] for s in s2vp], 's', color='#c0392b', ms=6, label='ShapeToVec+ (3K dims)')
off = {'1-bit': (7, -2), '2-level': (7, -10), '2-bit': (7, -12), '4-bit': (0, -16), 'ARE': (-6, -16)}
for t, gb, r, q in s2vp:
    ax.annotate(f'{t}\n{q:.1f}K', (gb, r), textcoords='offset points', xytext=off[t], ha='left' if off[t][0] > 0 else 'center',
                fontsize=6.5, color='#c0392b', linespacing=0.9)
ax.set_xscale('log')
ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(['2 GB', '4 GB', '8 GB', '16 GB', '32 GB'], fontsize=8.5)
ax.set_xlabel('Index memory at 10M polygons (log scale); point labels: thousands of QPS', fontsize=8)
ax.set_ylabel('Recall@50', fontsize=10); ax.set_ylim(0.5, 0.97); ax.set_xlim(1.7, 48)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig(f'{B}/IEEE/fig_pareto_10m.png', dpi=200); fig.savefig(f'{B}/IEEE/fig_pareto_10m.pdf')
print('saved IEEE/fig_pareto_10m.{png,pdf}')
