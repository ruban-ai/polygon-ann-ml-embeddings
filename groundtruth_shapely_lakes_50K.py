# Shapely exact Jaccard GT for 50K OSM lakes dataset.
# Corpus: 40K (0-39999), Test: 10K (40000-49999).
# Output: /raid/ruban/hpmlproj/groundtruth/lakes-shapely-50K/ (.npy, top-500)
#
# Run: nohup time python groundtruth_shapely_lakes_50K.py \
#        > /raid/ruban/hpmlproj/gt_shapely_lakes_50K.log 2>&1 &

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely import affinity
import shapely.wkt
import math, os, time, gc
from multiprocessing import Process, Array
from threading import Thread

# ── Config ────────────────────────────────────────────────────────────────────
LAKES_PATH   = "/raid/ssEncodingData/polygonalData/osm_new/lakes"
OUT_DIR      = "/raid/ruban/hpmlproj/groundtruth/lakes-shapely-50K/"
TOTAL        = 50_000
CORPUS_END   = 40_000
QUERY_START  = 40_000
QUERY_END    = 50_000
TOP_K        = 500
FILE_SIZE    = 125
pCount       = 75
tCount       = 2
# ─────────────────────────────────────────────────────────────────────────────

wktList = []


def read_lake_polygons(path, limit):
    polygons = []; skipped = 0; rows_seen = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            rows_seen += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2: skipped += 1; continue
            try:
                geom = shapely.wkt.loads(parts[1])
            except Exception:
                skipped += 1; continue
            if isinstance(geom, Polygon) and not geom.is_empty:
                polygons.append(geom)
            elif hasattr(geom, 'geoms'):
                polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
                if polys: polygons.append(max(polys, key=lambda p: p.area))
                else: skipped += 1; continue
            else:
                skipped += 1; continue
            if len(polygons) == limit: break
    print(f"Loaded {len(polygons):,} polygons from {rows_seen:,} rows  (skipped {skipped:,})")
    return polygons


def center(p):
    c = p.centroid
    return affinity.translate(p, -c.x, -c.y)


def shapely_jaccard(a, b):
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def FindSimilarPolygons(tid, bid, q_poly, tCount, start, end, results):
    local_size = math.floor((end - start) / tCount)
    s = start + tid * local_size
    e = start + local_size if tid < tCount - 1 else end
    for oid in range(s, e):
        results[oid - start] = shapely_jaccard(q_poly, wktList[oid])


def shapely_brute_force(bid, start, end, tCount):
    n = end - start
    results = Array("d", n)
    q_poly = wktList[bid]
    threads = [Thread(target=FindSimilarPolygons,
                      args=(t, bid, q_poly, tCount, start, end, results))
               for t in range(tCount)]
    for th in threads: th.start()
    for th in threads: th.join()
    return results


def create_gt_process(pid, out_dir, file_size, pCount, tCount,
                      data_start, data_end, query_start, query_end, top_k):
    local_size = math.floor((query_end - query_start) / pCount)
    s = query_start + pid * local_size
    e = s + local_size if pid < pCount - 1 else query_end

    n_queries = e - s
    print(f"[P{pid}] {n_queries} queries [{s}-{e-1}]", flush=True)
    t_start = time.time()

    batch = []; file_q_start = s

    for count, bid in enumerate(range(s, e)):
        similarity = shapely_brute_force(bid, data_start, data_end, tCount)
        row = [(i, similarity[i]) for i in range(data_end - data_start) if similarity[i] > 0]
        row.sort(key=lambda x: x[1], reverse=True)
        top_ids = np.array([x[0] for x in row[:top_k]], dtype=np.int32)
        if len(top_ids) < top_k:
            top_ids = np.pad(top_ids, (0, top_k - len(top_ids)), constant_values=-1)
        batch.append(top_ids)

        if len(batch) == file_size or bid == e - 1:
            file_end = file_q_start + len(batch) - 1
            fname = os.path.join(out_dir, f"similarityMap_{file_q_start}-{file_end}.npy")
            np.save(fname, np.array(batch, dtype=np.int32))
            batch = []; file_q_start = file_end + 1

        if (count + 1) % 10 == 0 or bid == e - 1:
            elapsed = time.time() - t_start
            rate = (count + 1) / elapsed if elapsed > 0 else 0
            eta = (n_queries - count - 1) / rate if rate > 0 else 0
            print(f"[P{pid}] {count+1}/{n_queries}  ({elapsed/60:.1f}min, ETA {eta/60:.1f}min)", flush=True)

    print(f"[P{pid}] DONE in {(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    t0 = time.time()

    print(f"Reading {TOTAL:,} lake polygons...", flush=True)
    raw = read_lake_polygons(LAKES_PATH, TOTAL)

    print(f"Centering {len(raw):,} polygons...", flush=True)
    for p in raw:
        wktList.append(center(p))
    del raw; gc.collect()
    print(f"wktList ready: {len(wktList):,}  ({(time.time()-t0)/60:.1f}min)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    n_queries = QUERY_END - QUERY_START
    if pCount > n_queries: pCount = n_queries

    print(f"\npCount={pCount}  tCount={tCount}  corpus=[{0},{CORPUS_END})  queries=[{QUERY_START},{QUERY_END})  top_k={TOP_K}", flush=True)
    print(f"Output: {OUT_DIR}\n", flush=True)

    processes = [
        Process(target=create_gt_process,
                args=(i, OUT_DIR, FILE_SIZE, pCount, tCount,
                      0, CORPUS_END, QUERY_START, QUERY_END, TOP_K))
        for i in range(pCount)
    ]
    for p in processes: p.start()
    for p in processes: p.join()

    print(f"\nTotal time: {(time.time()-t0)/3600:.2f}h", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)
