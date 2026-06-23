# Encoding for 50K OSM lakes dataset.
# Copy of Buddhi's encodeWeightedFileWriting.py with lakes() function added.
# Corpus: 40K, Test: 10K (80/20 split).
# Output: /raid/ruban/hpmlproj/encoding/lakes-real50k0.012/
#
# Run: nohup time python encode_lakes_50k_ruban.py \
#        > /raid/ruban/hpmlproj/encode_lakes_50k.log 2>&1 &

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon, GeometryCollection
from shapely import affinity, make_valid
import shapely.wkt
import math
from shapely.strtree import STRtree
from multiprocessing import Process, Array
import os, sys, time

sys.path.insert(1, '/raid/buddhi/ssQuant/lib/')
import wkthelper
from quadtree import quadtree

# ── Config ────────────────────────────────────────────────────────────────────
LAKES_PATH       = "/raid/ssEncodingData/polygonalData/osm_new/lakes"
OUT_DIR          = "/raid/ruban/hpmlproj/encoding/lakes-real50k0.022/"
TOTAL_POLYGONS   = 50_000
CORPUS_POLYGONS  = 40_000
QUERY_POLYGONS   = 10_000
nodeCapacityPerc = 0.022   # calibrated: gives 18,778 nodes (~18K dim)
pCount           = 200
fileSize         = 1000
th_area          = 0
foreground       = "area"
# ─────────────────────────────────────────────────────────────────────────────


def read_lake_polygons(path, limit):
    """Read up to `limit` valid polygons from the tab-separated WKT file."""
    polygons = []; rows_seen = 0; skipped = 0
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
                # take largest polygon from MultiPolygon / GeometryCollection
                polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
                if polys:
                    polygons.append(max(polys, key=lambda p: p.area))
                else:
                    skipped += 1; continue
            else:
                skipped += 1; continue
            if len(polygons) == limit:
                break
    print(f"Loaded {len(polygons):,} polygons from {rows_seen:,} rows  (skipped {skipped:,})")
    return polygons


def findSetMBR(inputWKTs, end, start=0):
    mbr = list(inputWKTs[start].bounds)
    for i in range(start+1, end):
        poly_mbr = list(inputWKTs[i].bounds)
        if poly_mbr[0] < mbr[0]: mbr[0] = poly_mbr[0]
        if poly_mbr[1] < mbr[1]: mbr[1] = poly_mbr[1]
        if poly_mbr[2] > mbr[2]: mbr[2] = poly_mbr[2]
        if poly_mbr[3] > mbr[3]: mbr[3] = poly_mbr[3]
    print(f"Global MBR [{start}-{end}] = {mbr}")
    return mbr


def initialPolygonCentering(inputWkts, end, start=0):
    centered = []; fixed = 0
    for i in range(start, end):
        c = inputWkts[i].centroid
        p = affinity.translate(inputWkts[i], -c.x, -c.y)
        if not p.is_valid:
            p = make_valid(p)
            fixed += 1
            # make_valid may return MultiPolygon — take largest
            if hasattr(p, 'geoms'):
                polys = [g for g in p.geoms if isinstance(g, Polygon)]
                p = max(polys, key=lambda x: x.area) if polys else inputWkts[i]
        centered.append(p)
    print(f"{len(centered)} polygons centered ({fixed} fixed with make_valid)")
    return centered


def preProcessQuadTree(wktList, data_start, data_end, data_percet, nodeCapacityPerc):
    node_capacity = math.ceil((data_end - data_start) * nodeCapacityPerc)
    global_mbr = findSetMBR(wktList, start=0, end=len(wktList))
    qt = quadtree(global_mbr, node_capacity)
    for i in range(data_start, data_end):
        xx, yy = wktList[i].exterior.xy
        for x, y in zip(xx, yy):
            qt.insert(x, y, f"Point({x}, {y})")
    qt.get_all_bounding_boxes()
    print(f"Quad tree: {len(qt.bounding_boxes)} nodes, height {qt.levels}, built from {data_end-data_start} polygons")
    return wktList, qt, global_mbr


def writeEncodeRealToStrToFile(filename, vectors, start):
    filename += "_" + str(start) + ".txt"
    with open(filename, 'w+') as f:
        for row in vectors:
            f.write(" ".join(str(float(v)) for v in row) + "\n")


def encodePolygonsRtreeVal(filename, fileSize, rtree, qt, wktList, start, end, pCount, pid, th_area, foreground, weighted):
    qtree_boxes = qt.bounding_boxes
    qtree_box_levels = qt.bounding_box_levels
    levels = qt.levels
    localSize = math.floor((end - start) / pCount)
    if (end - start) < pCount: localSize = 1
    s = start + pid * localSize
    e = s + localSize if pid < pCount - 1 else end
    csize = len(qtree_boxes)
    vectors = []; count = 1; ls = s

    for j in range(s, e):
        candidates = rtree.query(wktList[j]); candidates.sort()
        row = np.zeros(csize)
        for candidate in candidates:
            box = qtree_boxes[candidate]
            box_level = qtree_box_levels[candidate]
            box_geom = Polygon([(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])])
            if box_geom.intersects(wktList[j]):
                intersectArea = box_geom.intersection(wktList[j]).area
                if foreground == "area":
                    row[candidate] = intersectArea
        vectors.append(row)
        if (s != j and count % fileSize == 0) or j == e - 1:
            le = j + 1
            writeEncodeRealToStrToFile(filename, vectors, ls)
            vectors = []; ls = le
        count += 1
    print(f"Process {pid} finished.", flush=True)


def multiProcessEncodingWriting(filename, fileSize, rtree, qt, wktList, start, end, arg, foreground, weighted, pCount):
    if pCount > (end - start): pCount = end - start
    processes = [Process(target=encodePolygonsRtreeVal,
                         args=(filename, fileSize, rtree, qt, wktList, start, end, pCount, i, arg, foreground, weighted))
                 for i in range(pCount)]
    for p in processes: p.start()
    for p in processes: p.join()


def writeWKTListToSparseMatrix(filename, fileSize, wktList, data_start, data_end, query_start, query_end, data_percet, nodeCapacityPerc, th_area, foreground, pCount):
    print(f"Using {pCount} processors")
    t0 = time.time()
    wktList, qt, global_mbr = preProcessQuadTree(wktList, data_start, data_end, data_percet, nodeCapacityPerc)
    t1 = time.time()
    wktListQtBoxes = qt.convertQTToPolys()
    rtree = STRtree(wktListQtBoxes)
    t2 = time.time()
    multiProcessEncodingWriting(filename, fileSize, rtree, qt, wktList, data_start, query_end, th_area, foreground, True, pCount)
    t3 = time.time()
    print(f"Quadtree construction: {t1-t0:.1f}s")
    print(f"RTtree construction: {t2-t1:.1f}s")
    print(f"Encoding: {t3-t2:.1f}s")


def lakes():
    st = time.time()
    print(f"Reading {TOTAL_POLYGONS:,} lake polygons...")
    wktAll = read_lake_polygons(LAKES_PATH, TOTAL_POLYGONS)
    if len(wktAll) < TOTAL_POLYGONS:
        raise RuntimeError(f"Only found {len(wktAll)} polygons, expected {TOTAL_POLYGONS}")

    wktList = initialPolygonCentering(wktAll, end=len(wktAll), start=0)
    del wktAll

    data_start  = 0
    data_end    = CORPUS_POLYGONS
    query_start = CORPUS_POLYGONS
    query_end   = TOTAL_POLYGONS
    data_percet = CORPUS_POLYGONS / TOTAL_POLYGONS

    os.makedirs(OUT_DIR, exist_ok=True)
    filename = os.path.join(OUT_DIR, "real")

    print(f"Corpus: [{data_start}, {data_end})  Query: [{query_start}, {query_end})")
    print(f"nodeCapacityPerc={nodeCapacityPerc}  Output: {OUT_DIR}")

    writeWKTListToSparseMatrix(filename, fileSize, wktList, data_start, data_end,
                               query_start, query_end, data_percet,
                               nodeCapacityPerc, th_area, foreground, pCount)

    print(f"Total time: {time.time()-st:.1f}s")


def main():
    lakes()


if __name__ == "__main__":
    main()
