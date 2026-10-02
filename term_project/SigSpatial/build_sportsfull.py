#!/usr/bin/env python3
"""Build the FULL Sports benchmark (mirrors build_wbfull.py, parallelized for
1.75M x 12,430): ShapeToVec encodings at the highest available resolution
(sp-real0.002, D=12,430 -- ShapeToVec's own 12K configuration), id order, written
to one float32 memmap (qt_sportsfull.npy, ~87 GB), plus the GT pickle.

Split = the existing sports_full convention (No Encoding/.../run_rawvertex_distill_
sports_full_v4.py): corpus ids [0, 1,403,192), queries [1,403,192, 1,753,989)
(80/20, 350,797 queries). GT: warehouse/sports_all-query similarityMap_* text,
'qid, n1, n2, ...' ranked closest-first; neighbours restricted to the corpus and
capped at the first 1000 (same as build_wbfull.py).
Every output goes to durable /raid storage, never /tmp."""
import glob, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

ENC = '/raid/ssEncodingData/encoding/papers-data/sp-real0.002'
GT_DIR = '/raid/ssEncodingData/warehouse/sports_all-query'
BASE = '/raid/ruban/hpmlproj/term_project/SigSpatial'
OUT_QT = f'{BASE}/qt_sportsfull.npy'
OUT_GT = f'{BASE}/gt_sportsfull.pkl'
OUT_META = f'{BASE}/sportsfull_meta.pkl'
CORPUS_N = 1_403_192; TOTAL_N = 1_753_989; D = 12_430; GT_CAP = 1000
SMOKE = '--smoke' in sys.argv


def kf(f): return int(os.path.basename(f).split('real_')[1].split('.txt')[0])


def enc_job(args):
    f, start, n_expected = args
    a = pd.read_csv(f, sep=r'\s+', header=None, dtype=np.float32, engine='c').values
    if a.shape != (n_expected, D):
        return (f, a.shape, 'BAD')
    X = np.load(OUT_QT, mmap_mode='r+'); X[start:start + n_expected] = a; X.flush(); del X
    return (f, a.shape, 'ok')


def gt_job(f):
    out = {}
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = line.split(',', 4 * GT_CAP + 1)          # parse only a prefix of very long lines
            qid = int(p[0])
            if not (CORPUS_N <= qid < TOTAL_N):
                continue
            nb = []
            for x in p[1:4 * GT_CAP + 1]:
                x = x.strip()
                if x:
                    v = int(x)
                    if v < CORPUS_N:
                        nb.append(v)
                        if len(nb) == GT_CAP:
                            break
            if nb:
                out[qid] = nb
    return out


def main():
    t0 = time.time()
    files = sorted(glob.glob(ENC + '/real_*.txt'), key=kf)
    starts = [kf(f) for f in files]
    sizes = [b - a for a, b in zip(starts, starts[1:] + [TOTAL_N])]
    assert starts[0] == 0 and all(s > 0 for s in sizes), 'shard ids not contiguous'
    print(f"{len(files)} encoding shards, ids [0,{TOTAL_N}), D={D}", flush=True)
    jobs = list(zip(files, starts, sizes))
    if SMOKE:
        jobs = jobs[:3] + jobs[-2:]
    if not os.path.exists(OUT_QT) or SMOKE:
        np.lib.format.open_memmap(OUT_QT, mode='w+', dtype=np.float32, shape=(TOTAL_N, D)).flush()
        print(f"allocated {OUT_QT} ({TOTAL_N*D*4/1024**3:.1f} GB)", flush=True)
        bad = []
        with Pool(40) as pool:
            for k, (f, shp, st) in enumerate(pool.imap_unordered(enc_job, jobs), 1):
                if st != 'ok':
                    bad.append((f, shp))
                if k % 50 == 0 or k == len(jobs):
                    el = time.time() - t0
                    print(f"  encodings {k}/{len(jobs)} shards ({el/60:.1f}min, "
                          f"ETA {el/k*(len(jobs)-k)/60:.1f}min)", flush=True)
        if bad:
            print(f"FATAL: {len(bad)} shards with unexpected shape, e.g. {bad[:3]}", flush=True); sys.exit(1)
    else:
        print(f"{OUT_QT} exists, skipping encoding step", flush=True)

    X = np.load(OUT_QT, mmap_mode='r')
    for s in [0, CORPUS_N - 1, CORPUS_N, TOTAL_N - 1]:
        if SMOKE and s not in (0, TOTAL_N - 1):
            continue
        r = np.asarray(X[s]); print(f"  sanity row {s}: sum={r.sum():.3e} nnz={int((r>0).sum())}", flush=True)
        assert r.sum() > 0, f'row {s} is all-zero (shard not written?)'

    gfiles = sorted(glob.glob(GT_DIR + '/similarityMap_*'))
    gfiles = [f for f in gfiles if 'overlapPercent' not in f]
    if SMOKE:
        gfiles = gfiles[:3]
    gt = {}; tg = time.time()
    with Pool(32) as pool:
        for k, part in enumerate(pool.imap_unordered(gt_job, gfiles), 1):
            gt.update(part)
            if k % 50 == 0 or k == len(gfiles):
                el = time.time() - tg
                print(f"  GT {k}/{len(gfiles)} files, {len(gt):,} queries ({el/60:.1f}min, "
                      f"ETA {el/k*(len(gfiles)-k)/60:.1f}min)", flush=True)
    d = [len(v) for v in gt.values()]
    print(f"GT queries with >=1 corpus neighbour: {len(gt):,}/{TOTAL_N-CORPUS_N:,}; depth mean={np.mean(d):.1f} "
          f"min={min(d)} max={max(d)}", flush=True)
    if SMOKE:
        print("SMOKE OK (outputs not final)", flush=True); return
    pickle.dump(gt, open(OUT_GT, 'wb'))
    pickle.dump({'corpus_n': CORPUS_N, 'total': TOTAL_N, 'dim': D, 'source': ENC, 'gt': GT_DIR},
                open(OUT_META, 'wb'))
    print(f"done, total {(time.time()-t0)/60:.1f}min -> {OUT_QT}, {OUT_GT}", flush=True)


if __name__ == '__main__':
    main()
