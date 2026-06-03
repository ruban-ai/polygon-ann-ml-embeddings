#!/usr/bin/env python3
"""
20_cache_raster.py — Cache rasterized polygon embeddings for Track B (CNN raster model)

Key design decisions:
  - Polygons are CENTERED (centroid → origin) before rasterizing, matching quadtree encoding
  - All polygons rasterized at a GLOBAL consistent scale (95th percentile of extents)
    so that large/small polygons appear proportionally — preserving size information
    that WJ on centered quadtree also captures
  - Global scale saved to /tmp/raster_global_scale.npy for consistent inference

Outputs (all in /tmp/):
  rasters_parks_50k.npy    (50000, 64, 64) uint8
  rasters_sports_50k.npy   (50000, 64, 64) uint8
  rasters_google_20k.npy   (20000, 64, 64) uint8
  gt_lookup_parks_50k.pkl  {qid: [neighbor_ids]}  — neighbors only (no query ID)
  gt_lookup_sports_50k.pkl {qid: [neighbor_ids]}
  qt_parks_50k.npy         (50000, 18382) float32 — for WJ rerank on parks
  raster_global_scale.npy  scalar — saved for consistent inference
"""

import glob
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
GRID_SIZE   = 64
SUPERSAMPLE = 4      # render at 4x resolution, then avg-pool → fractional coverage
N_WORKERS   = 16
BATCH_SIZE  = 1000

PARKS_WKT      = "/raid/ssEncodingData/polygonalData/osm_new/parks"
SPORTS_WKT     = "/raid/ruban/data/sports"
LAKES_WKT      = "/raid/ssEncodingData/polygonalData/osm_new/lakes"
PARKS_ENC_DIR  = "/raid/ssEncodingData/encoding/pk-real50k0.002"
PARKS_GT_DIR   = "/raid/ruban/groundtruth/pk-query-50k"
SPORTS_GT_DIR  = "/raid/ssEncodingData/warehouse/sports-query-50k"

OUT_DIR        = "/tmp"
N_PARKS        = 50000
N_SPORTS       = 50000
N_LAKES        = 20000   # zero-shot eval — model never trained on lakes



# ── Geometry helpers ──────────────────────────────────────────────────────────
def _extract_polygon(geom):
    t = geom.geom_type
    if t == "Polygon":
        return geom
    if t == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    if t == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return _extract_polygon(max(polys, key=lambda g: g.area))
    return None


# ── Rasterization ─────────────────────────────────────────────────────────────
def _rasterize_one(wkt_str):
    """
    1. Center polygon (centroid → origin), matching Buddhi's quadtree preprocessing.
    2. Normalize to fill the grid (per-polygon), so every polygon has good coverage.
    3. Rasterize at 4× supersampling then avg-pool → fractional pixel coverage [0,1].

    Centering removes geographic location dependency.
    Per-polygon normalization ensures all polygons (parks, sports, lakes) are visible.
    Size information comes implicitly from the WJ GT training signal.
    """
    from PIL import Image, ImageDraw
    from shapely import wkt as shapely_wkt
    from shapely.affinity import translate

    blank = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if not wkt_str or not wkt_str.strip():
        return blank
    try:
        geom = shapely_wkt.loads(wkt_str)
        poly = _extract_polygon(geom)
        if poly is None or poly.is_empty:
            return blank

        # ── Center polygon (matches Buddhi's quadtree preprocessing) ─────────
        cx, cy = poly.centroid.x, poly.centroid.y
        poly   = translate(poly, -cx, -cy)

        minx, miny, maxx, maxy = poly.bounds
        extent = max(maxx - minx, maxy - miny)
        if extent < 1e-12:
            return blank

        # ── Per-polygon normalization: scale to fill the grid ─────────────────
        HIGH_RES   = GRID_SIZE * SUPERSAMPLE          # 256
        margin     = 2 * SUPERSAMPLE
        pixel_size = (HIGH_RES - 2 * margin) / extent  # each polygon fills its grid
        half       = HIGH_RES / 2.0

        coords = np.array(poly.exterior.coords)
        px = coords[:, 0] * pixel_size + half
        py = half - coords[:, 1] * pixel_size          # flip Y axis

        px = np.clip(px, 0, HIGH_RES - 1)
        py = np.clip(py, 0, HIGH_RES - 1)

        img = Image.new("L", (HIGH_RES, HIGH_RES), 0)
        ImageDraw.Draw(img).polygon(list(zip(px.tolist(), py.tolist())), fill=1)

        # ── 4× avg-pool → fractional pixel coverage ───────────────────────────
        arr = np.array(img, dtype=np.float32)
        arr = arr.reshape(GRID_SIZE, SUPERSAMPLE,
                          GRID_SIZE, SUPERSAMPLE).mean(axis=(1, 3))
        return arr.astype(np.float32)
    except Exception:
        return blank


def _rasterize_batch(wkts):
    return [_rasterize_one(w) for w in wkts]


def rasterize_parallel(wkts, desc="Rasterizing"):
    batches = [wkts[i:i + BATCH_SIZE] for i in range(0, len(wkts), BATCH_SIZE)]
    results = []
    with Pool(N_WORKERS) as pool:
        for batch_out in tqdm(
            pool.imap(_rasterize_batch, batches), total=len(batches), desc=desc
        ):
            results.extend(batch_out)
    arr    = np.stack(results, axis=0)
    filled = int((arr.sum(axis=(1, 2)) > 0).sum())
    print(f"  shape={arr.shape} | non-empty={filled}/{len(arr)}")
    return arr


# ── Global scale estimation ───────────────────────────────────────────────────


# ── WKT loaders ──────────────────────────────────────────────────────────────
def load_tsv_wkt(path, n):
    """Load first n WKT strings from tab-separated file: <id>\\t<WKT>\\t<tags...>"""
    wkts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.rstrip("\n").split("\t")
            wkts.append(parts[1] if len(parts) >= 2 else "")
    return wkts


def load_lakes_wkt(path, n):
    """Load first n WKT strings from lakes file (same format as parks: id\\tWKT\\t...)."""
    return load_tsv_wkt(path, n)


# ── GT loader (Buddhi-compatible format) ─────────────────────────────────────
def load_gt(gt_dir):
    """
    Load similarity map files → {qid: [neighbor_ids]}.
    Format per line: qid, n1, n2, ...  (comma + space separated)
    Stores ONLY neighbor IDs (not query ID) — query ID added in eval to match
    Buddhi's denominator convention.
    """
    gt = {}
    for fpath in sorted(glob.glob(os.path.join(gt_dir, "similarityMap_*"))):
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split(",")
                if not parts or not parts[0].strip():
                    continue
                qid       = int(parts[0])
                neighbors = [int(x) for x in parts[1:] if x.strip()]
                gt[qid]   = neighbors
    return gt


# ── Parks quadtree vectors for WJ rerank ─────────────────────────────────────
def load_parks_qt(enc_dir, n=50000):
    import re
    files = sorted(
        glob.glob(os.path.join(enc_dir, "real_*.txt")),
        key=lambda p: int(re.search(r"real_(\d+)\.txt", p).group(1)),
    )
    chunks, total = [], 0
    for fpath in tqdm(files, desc="Loading parks qt"):
        mat = np.loadtxt(fpath, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        remaining = n - total
        if remaining <= 0:
            break
        mat    = mat[:remaining]
        chunks.append(mat)
        total += mat.shape[0]
        if total >= n:
            break
    qt = np.vstack(chunks)
    print(f"Parks qt shape: {qt.shape}")
    return qt


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    # ── Load all WKTs first (needed for scale estimation) ────────────────────
    print(f"\n{'='*60}")
    print("Loading WKT files...")
    parks_wkts  = load_tsv_wkt(PARKS_WKT, N_PARKS)
    sports_wkts = load_tsv_wkt(SPORTS_WKT, N_SPORTS)
    lakes_wkts  = load_lakes_wkt(LAKES_WKT, N_LAKES)
    print(f"parks={len(parks_wkts)} sports={len(sports_wkts)} lakes={len(lakes_wkts)}")

    # ── Parks rasters ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Parks {N_PARKS} → {GRID_SIZE}×{GRID_SIZE} rasters (global scale)")
    t0 = time.time()
    parks_rasters = rasterize_parallel(parks_wkts, desc="Parks")
    np.save(os.path.join(OUT_DIR, "rasters_parks_50k.npy"), parks_rasters)
    print(f"Saved rasters_parks_50k.npy in {time.time()-t0:.1f}s")
    del parks_wkts, parks_rasters

    # ── Sports rasters ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Sports {N_SPORTS} → {GRID_SIZE}×{GRID_SIZE} rasters (global scale)")
    t0 = time.time()
    sports_rasters = rasterize_parallel(sports_wkts, desc="Sports")
    np.save(os.path.join(OUT_DIR, "rasters_sports_50k.npy"), sports_rasters)
    print(f"Saved rasters_sports_50k.npy in {time.time()-t0:.1f}s")
    del sports_wkts, sports_rasters

    # ── Lakes rasters (zero-shot eval — not used in training) ─────────────────
    print(f"\n{'='*60}")
    print(f"Lakes {len(lakes_wkts)} → {GRID_SIZE}×{GRID_SIZE} rasters (global scale)")
    t0 = time.time()
    lakes_rasters = rasterize_parallel(lakes_wkts, desc="Lakes")
    np.save(os.path.join(OUT_DIR, "rasters_lakes_20k.npy"), lakes_rasters)
    print(f"Saved rasters_lakes_20k.npy in {time.time()-t0:.1f}s")
    del lakes_wkts, lakes_rasters

    # ── GT lookups ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading GT lookups...")
    parks_gt = load_gt(PARKS_GT_DIR)
    with open(os.path.join(OUT_DIR, "gt_lookup_parks_50k.pkl"), "wb") as f:
        pickle.dump(parks_gt, f)
    print(f"Parks GT: {len(parks_gt)} queries")

    sports_gt = load_gt(SPORTS_GT_DIR)
    with open(os.path.join(OUT_DIR, "gt_lookup_sports_50k.pkl"), "wb") as f:
        pickle.dump(sports_gt, f)
    print(f"Sports GT: {len(sports_gt)} queries")

    # ── Parks quadtree for WJ rerank ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading parks quadtree vectors...")
    t0 = time.time()
    qt_parks = load_parks_qt(PARKS_ENC_DIR, N_PARKS)
    np.save(os.path.join(OUT_DIR, "qt_parks_50k.npy"), qt_parks)
    print(f"Saved qt_parks_50k.npy in {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("All caches ready in /tmp/:")
    for fname in [
        "rasters_parks_50k.npy", "rasters_sports_50k.npy", "rasters_lakes_20k.npy",
        "gt_lookup_parks_50k.pkl", "gt_lookup_sports_50k.pkl",
        "qt_parks_50k.npy",
    ]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            mb = os.path.getsize(fpath) / 1e6
            print(f"  {fname}: {mb:.1f} MB")
