#!/usr/bin/env python3
"""Build the FULL-SCALE water-body benchmark, mirroring build_50k.py -> the full
187K Parks pipeline (run_full187k_listwise.py / run_matdistill_fulleval_ddp.py):
train on the corpus-internal neighbour structure so ALL held-out queries are
genuinely unseen, then evaluate ALL of them against the full corpus.

Water-body source: /raid/ssEncodingData/encoding/papers-data/wb-real0.006/ (448,550
polygons total, D=12,013 quadtree dims, same real_*.txt format as Parks).
GT source: /raid/ssEncodingData/warehouse/wb-query-358840/ (similarityMap_* files,
query ids start at 358840 -- exactly an 80/20 corpus/query split, 358840/448550=0.800,
matching the Parks 233K convention).

Corpus = ids [0, 358840); queries = ALL ids [358840, 448550) (89,710 queries).
Every output goes to durable /raid storage, never /tmp.
"""
import glob, pickle, time, bisect
import numpy as np, pandas as pd

ENC = '/raid/ssEncodingData/encoding/papers-data/wb-real0.006'
GT_DIR = '/raid/ssEncodingData/warehouse/wb-query-358840'
CORPUS_N = 358840
QUERY_START = 358840
TOTAL_N = 448550
QUERY_N = TOTAL_N - QUERY_START
OUT_QT = '/raid/ruban/hpmlproj/term_project/SigSpatial/qt_wbfull.npy'
OUT_GT = '/raid/ruban/hpmlproj/term_project/SigSpatial/gt_wbfull.pkl'
OUT_META = '/raid/ruban/hpmlproj/term_project/SigSpatial/wbfull_meta.pkl'


def kf(f):
    return int(f.split('real_')[1].split('.txt')[0])


def load_range(files_sorted, starts, lo, hi):
    """Load only the files covering absolute row range [lo, hi), return rows sliced
    to exactly that range, indexed so output row 0 == absolute row id `lo`."""
    i0 = bisect.bisect_right(starts, lo) - 1
    i0 = max(i0, 0)
    i1 = bisect.bisect_right(starts, hi - 1)
    parts, part_starts = [], []
    for i in range(i0, i1):
        df = pd.read_csv(files_sorted[i], sep=r'\s+', header=None, dtype=np.float32)
        parts.append(df.values)
        part_starts.append(starts[i])
    block = np.vstack(parts)
    block_start = part_starts[0]
    return block[lo - block_start: hi - block_start]


def main():
    t0 = time.time()
    files = sorted(glob.glob(ENC + '/real_*.txt'), key=kf)
    starts = [kf(f) for f in files]
    print(f"{len(files)} encoding files, id range [{starts[0]}, {starts[-1]}]", flush=True)

    print(f"loading corpus rows [0, {CORPUS_N}) ...", flush=True)
    corpus = load_range(files, starts, 0, CORPUS_N)
    print(f"corpus {corpus.shape} loaded in {(time.time()-t0)/60:.1f}min", flush=True)

    print(f"loading query rows [{QUERY_START}, {QUERY_START+QUERY_N}) (ALL {QUERY_N}) ...", flush=True)
    queries = load_range(files, starts, QUERY_START, QUERY_START + QUERY_N)
    print(f"queries {queries.shape} loaded in {(time.time()-t0)/60:.1f}min", flush=True)

    qt = np.ascontiguousarray(np.vstack([corpus, queries]), dtype=np.float32)
    np.save(OUT_QT, qt)
    print(f"qt {qt.shape} saved -> {OUT_QT} ({qt.nbytes/1024**3:.2f} GB)", flush=True)

    print("parsing GT ...", flush=True)
    gt = {}
    for f in sorted(glob.glob(GT_DIR + '/similarityMap_*')):
        for line in open(f):
            p = line.strip().split(',')
            if len(p) < 2:
                continue
            qid = int(p[0])
            if not (QUERY_START <= qid < QUERY_START + QUERY_N):
                continue
            nbrs = [int(x) for x in p[1:] if x.strip() != '']
            nbrs = [n for n in nbrs if n < CORPUS_N][:1000]
            if nbrs:
                gt[CORPUS_N + (qid - QUERY_START)] = nbrs
    pickle.dump(gt, open(OUT_GT, 'wb'))
    print(f"GT queries with >=1 in-corpus neighbour: {len(gt)}/{QUERY_N}", flush=True)
    if gt:
        depths = [len(v) for v in gt.values()]
        print(f"GT depth: mean={np.mean(depths):.1f} min={min(depths)} max={max(depths)}", flush=True)

    pickle.dump({'corpus_n': CORPUS_N, 'query_start': QUERY_START, 'query_n': QUERY_N,
                 'dim': qt.shape[1], 'source': ENC}, open(OUT_META, 'wb'))
    print(f"done, total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
