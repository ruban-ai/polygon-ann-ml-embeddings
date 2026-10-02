#!/usr/bin/env python3
"""Build GT pickles for Overture 1M / 10M in the same format every listwise eval
uses: {row_index_of_query: [corpus_row_ids ranked closest-first]}.
Source: /raid/ruban/groundtruth/ov-shapely-{1M,10M}/similarityMap_<a>-<b>.npy,
int32 (n,500) right-padded with -1; row r of file <a>-<b> is query id a+r in the
10M id space (9,999,000..9,999,999). In both encodings the 1,000 queries are the
last 1,000 rows (manifest test_start = 999,000 / 9,999,000), so
row_index = corpus_n + (qid - 9,999,000).
Usage: python build_gt_overture.py ov1m|ov10m"""
import glob, os, pickle, sys, time
import numpy as np
sys.path.insert(0, '/raid/ruban/hpmlproj/term_project/SigSpatial')
from fulls_common import CONFIGS

SRC = {'ov1m': '/raid/ruban/groundtruth/ov-shapely-1M', 'ov10m': '/raid/ruban/groundtruth/ov-shapely-10M'}
Q0 = 9_999_000

cfg = CONFIGS[sys.argv[1]]; t0 = time.time()
files = sorted(glob.glob(f"{SRC[cfg['name']]}/similarityMap_*.npy"),
               key=lambda f: int(os.path.basename(f).split('_')[1].split('-')[0]))
gt = {}
for f in files:
    a = int(os.path.basename(f).split('_')[1].split('-')[0]); arr = np.load(f)
    for r in range(arr.shape[0]):
        nb = [int(x) for x in arr[r] if 0 <= x < cfg['corpus_n']]
        if nb:
            gt[cfg['corpus_n'] + (a + r - Q0)] = nb
assert len(gt) == cfg['total'] - cfg['corpus_n'], (len(gt), cfg['total'] - cfg['corpus_n'])
assert min(gt) == cfg['corpus_n'] and max(gt) == cfg['total'] - 1
d = [len(v) for v in gt.values()]
pickle.dump(gt, open(cfg['gt'], 'wb'))
print(f"{cfg['name']}: {len(files)} files -> {len(gt)} queries, GT depth mean={np.mean(d):.1f} "
      f"min={min(d)} max={max(d)} -> {cfg['gt']} ({time.time()-t0:.1f}s)", flush=True)
