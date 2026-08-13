#!/usr/bin/env python3
"""Build a 50K-style water-body benchmark (Review 2 ask: a genuine second polygon
class, not just a Parks re-slice), mirroring build_50k.py's structure exactly so the
existing run_50k_ddp.py-style pipeline works unchanged.

Water-body source: /raid/ssEncodingData/encoding/papers-data/wb-real0.006/ (448,550
polygons total, D=12,013 quadtree dims, same real_*.txt format as Parks).
GT source: /raid/ssEncodingData/warehouse/wb-query-358840/ (similarityMap_* files,
query ids start at 358840).

We take a 40K corpus (ids 0..39999) + 10K held-out queries (ids 358840..368839) --
the same scale as the paper's Parks 50K benchmark, for an apples-to-apples comparison
-- WITHOUT materializing the full 448K-row corpus (avoids ~21GB of unneeded I/O).
"""
import glob, pickle, time, bisect
import numpy as np, pandas as pd

ENC = '/raid/ssEncodingData/encoding/papers-data/wb-real0.006'
GT_DIR = '/raid/ssEncodingData/warehouse/wb-query-358840'
CORPUS_N = 40000
QUERY_START = 358840
QUERY_N = 10000
OUT_QT = '/raid/ruban/hpmlproj/term_project/SigSpatial/qt_wb50k.npy'
OUT_GT = '/raid/ruban/hpmlproj/term_project/SigSpatial/gt_wb50k.pkl'
OUT_META = '/raid/ruban/hpmlproj/term_project/SigSpatial/wb50k_meta.pkl'


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

    print(f"loading query rows [{QUERY_START}, {QUERY_START+QUERY_N}) ...", flush=True)
    queries = load_range(files, starts, QUERY_START, QUERY_START + QUERY_N)
    print(f"queries {queries.shape} loaded in {(time.time()-t0)/60:.1f}min", flush=True)

    qt = np.ascontiguousarray(np.vstack([corpus, queries]), dtype=np.float32)
    np.save(OUT_QT, qt)
    print(f"qt {qt.shape} saved -> {OUT_QT} ({qt.nbytes/1024**3:.2f} GB)", flush=True)

    # GT: parse similarityMap_* files, keep only queries in our range, remap
    # neighbour ids to corpus-local ids (drop any neighbour >= CORPUS_N, since our
    # mini-corpus only contains ids [0, CORPUS_N)).
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
                gt[qid] = nbrs
    pickle.dump(gt, open(OUT_GT, 'wb'))
    print(f"GT queries with >=1 in-corpus neighbour: {len(gt)}/{QUERY_N} "
          f"(some queries' true neighbours may fall outside the 40K mini-corpus "
          f"and are naturally excluded, same caveat as any corpus subsampling)", flush=True)
    if gt:
        depths = [len(v) for v in gt.values()]
        print(f"GT depth: mean={np.mean(depths):.1f} min={min(depths)} max={max(depths)}", flush=True)

    pickle.dump({'corpus_n': CORPUS_N, 'query_start': QUERY_START, 'query_n': QUERY_N,
                 'dim': qt.shape[1], 'source': ENC}, open(OUT_META, 'wb'))
    print(f"done, total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
