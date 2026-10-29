#!/usr/bin/env python3
"""2026-10-25: ICWS now 256 samples, same width as the others (Buddhi; RESULTS_LOG parksfull-ICWSbrute-d256); old figure kept as fig_baselines_icws512_20261010.pdf.
2026-10-10: wrapped y labels + thicker bars (Buddhi). 2026-10-09: our row = the 64-4096 Matryoshka encoder now reported for Parks (results_parksfull_m64_quant.json,
float32 256-d), label = method name; values printed with 2 decimals. Baselines unchanged.
Baselines on Parks as a plot (2026-10-07; replaces the baselines table at Buddhi's request).
Numbers are exactly those of the table (RESULTS_LOG.csv, parksfull_allq rows; 256-d; ICWS 512 samples).
Writes IEEE/fig_baselines.png."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows = [  # method, R@10, R@50, R@500  (sorted by R@50, ascending)
    ('PCA', 0.053, 0.082, 0.151),
    ('InfoNCE (WJ)', 0.433, 0.513, 0.623),
    ('Triplet\n+ recon.', 0.455, 0.544, 0.653),
    ('NMF', 0.540, 0.581, 0.664),
    ('Random\nprojection', 0.591, 0.626, 0.690),
    ('ICWS\n(exhaustive)', 0.6738, 0.7436, 0.8311),
    ('MSE\ndistillation', 0.720, 0.797, 0.890),
    ('Listwise WJ\ndistillation', 0.8494, 0.8976, 0.9419),
]
names = [r[0] for r in rows]; y = np.arange(len(rows)); h = 0.29
cols = ['#9ecae1', '#3182bd', '#08519c']           # R@10, R@50, R@500 (light to dark blue)
fig, ax = plt.subplots(figsize=(4.6, 3.8))
for k, (lab, c) in enumerate(zip(['R@10', 'R@50', 'R@500'], cols)):
    v = [r[k + 1] for r in rows]
    ax.barh(y + (k - 1) * h, v, height=h, color=c, label=lab)
for i, r in enumerate(rows):                       # label R@50 values
    ax.text(r[2] + 0.01, i, f'{r[2]:.2f}', va='center', fontsize=7.5, color='#08306b')
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5, linespacing=0.95)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.set_xlim(0, 1.0); ax.set_xlabel('Recall', fontsize=9.5)
ax.tick_params(axis='x', labelsize=8.5)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=8, loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_baselines.png', dpi=200)
fig.savefig('/raid/ruban/hpmlproj/term_project/SigSpatial/IEEE/fig_baselines.pdf')   # 2026-10-10: vector PDF for the paper (Buddhi)
print('saved IEEE/fig_baselines.png')
