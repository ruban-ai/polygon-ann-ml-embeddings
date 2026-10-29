#!/usr/bin/env python3
"""2026-10-10 generalization (Buddhi 10-10: comparisons across methods must use the same input resolution ->
Water bodies and Sports at 3K like ShapeToVec+). Usage: python build_res.py <dataset> <tag>, dataset in {parks,wb,sports}.
Verified 2026-10-10: wb-real0.024 / sp-real0.008 shards share id order + per-polygon area with the 12K encodings.
2026-10-09 input-resolution sweep (Buddhi, 10-09 meeting; mirrors ShapeToVec+'s spatial-depth study).
Builds Parks ShapeToVec encodings at a coarser quadtree resolution into one float32 .npy, id order.
Verified: pk-real* text shards share the id order of qtree_vectors_full.npy (pk-real0.002, 18,220-d;
rows match exactly) and per-polygon total area is identical across resolutions, so the Parks GT
(gt_lookup_full.pkl, geometry-based) and the 80/20 split are reused unchanged.
Usage: python build_parks_res.py <tag>   tag in {3k,6k,12k}. Outputs on /raid, never /tmp."""
import glob, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

DS = {'parks': (233_773, 'qtree_vectors_full.npy', {'3k': 'pk-real0.012', '6k': 'pk-real0.006', '12k': 'pk-real0.003'}),
      'wb': (448_550, 'qt_wbfull.npy', {'3k': 'papers-data/wb-real0.024', '6k': 'papers-data/wb-real0.0105'}),
      'sports': (1_753_989, 'qt_sportsfull.npy', {'3k': 'papers-data/sp-real0.008', '6k': 'papers-data/sp-real0.0041'})}
NAME, TAG = sys.argv[1], sys.argv[2]
TOTAL_N, REF, RES = DS[NAME]
ENC = f'/raid/ssEncodingData/encoding/{RES[TAG]}'
OUT = f'/raid/ruban/hpmlproj/term_project/SigSpatial/qt_{NAME}res_{TAG}.npy'


def kf(f): return int(f.split('real_')[-1][:-4])


def job(a):
    f, s = a
    return s, pd.read_csv(f, sep=r'\s+', header=None, dtype=np.float32, engine='c').values


if __name__ == '__main__':
    t0 = time.time()
    fs = sorted(glob.glob(ENC + '/real_*.txt'), key=kf); st = [kf(f) for f in fs]
    assert st[0] == 0
    D = pd.read_csv(fs[0], sep=r'\s+', header=None, dtype=np.float32, nrows=1).shape[1]
    print(f"{TAG}: {len(fs)} shards from {ENC}, D={D}", flush=True)
    X = np.lib.format.open_memmap(OUT, mode='w+', dtype=np.float32, shape=(TOTAL_N, D))
    done = 0
    with Pool(32) as p:
        for i, (s, a) in enumerate(p.imap_unordered(job, zip(fs, st))):
            assert a.shape[1] == D, (s, a.shape)
            X[s:s + len(a)] = a; done += len(a)
            if i % 40 == 0:
                print(f"  {i+1}/{len(fs)} shards, {done:,} rows, {(time.time()-t0)/60:.1f}min", flush=True)
    assert done == TOTAL_N, done
    X.flush()
    ref = np.load(f'/raid/ruban/hpmlproj/term_project/SigSpatial/{REF}', mmap_mode='r')
    idx = np.random.default_rng(0).choice(TOTAL_N, 2000, replace=False)
    rel = np.abs(X[idx].sum(1) - ref[idx].sum(1)) / np.maximum(ref[idx].sum(1), 1e-30)
    print(f"total-area check vs reference encoding: max rel diff {rel.max():.2e}, zero rows {(X[idx].sum(1)==0).sum()}", flush=True)
    print(f"BUILD DONE {TAG} {X.shape} -> {OUT} ({X.nbytes/1024**3:.1f} GB) {(time.time()-t0)/60:.1f}min", flush=True)
