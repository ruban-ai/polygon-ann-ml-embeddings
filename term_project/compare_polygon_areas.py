#!/usr/bin/env python3
"""
Compare polygon area distributions: corpus (80%) vs GT queries (20%).

The encoding and warehouse directories hold vectors / neighbor lists, not
geometry. This script recovers the train/query split from the GT filenames
(e.g. similarityMap_187019-...) and reads polygons from the Parks WKT file.

Usage:
    python compare_polygon_areas.py
    python compare_polygon_areas.py --out-dir ./my_output --bins 60
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt
from pyproj import Geod
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

WKT_PATH = "/raid/ssEncodingData/polygonalData/osm_new/parks"
ENCODING_DIR = "/raid/ssEncodingData/encoding/pk-real0.002"
GT_DIR = "/raid/ssEncodingData/warehouse/pk-query-187019"
OUT_DIR = os.path.join(os.path.dirname(__file__), "area_distribution")

GT_RE = re.compile(r"similarityMap_(\d+)-(\d+)")
GEOD = Geod(ellps="WGS84")


def discover_split(gt_dir: str) -> tuple[int, int]:
    """Return (query_start, total_polygon_count) from GT similarityMap filenames."""
    starts, ends = [], []
    for path in glob.glob(os.path.join(gt_dir, "similarityMap_*")):
        m = GT_RE.search(os.path.basename(path))
        if m:
            starts.append(int(m.group(1)))
            ends.append(int(m.group(2)))
    if not starts:
        sys.exit(f"No similarityMap_* files found in {gt_dir}")
    return min(starts), max(ends) + 1


def polygon_parts(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            yield from polygon_parts(g)


def area_m2(geom) -> float:
    total = 0.0
    for poly in polygon_parts(geom):
        try:
            a, _ = GEOD.geometry_area_perimeter(poly)
            total += abs(a)
        except Exception:
            pass
    return total


def load_areas(wkt_path: str, limit: int, progress_every: int = 50_000) -> np.ndarray:
    """Read polygons from WKT until `limit` valid positive areas are collected."""
    areas: list[float] = []
    rows_seen = 0
    with open(wkt_path, encoding="utf-8") as f:
        for line in f:
            rows_seen += 1
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 2:
                continue
            try:
                geom = shapely.wkt.loads(parts[1])
                a = area_m2(geom)
            except Exception:
                continue
            if not math.isfinite(a) or a <= 0:
                continue
            areas.append(a)
            if progress_every and len(areas) % progress_every == 0:
                print(f"  loaded {len(areas):,} / {limit:,} areas", flush=True)
            if len(areas) >= limit:
                break
    if len(areas) < limit:
        sys.exit(
            f"Only collected {len(areas):,} valid areas after {rows_seen:,} rows "
            f"(needed {limit:,})."
        )
    return np.array(areas, dtype=np.float64)


def print_stats(name: str, areas: np.ndarray) -> None:
    p = np.percentile(areas, [5, 25, 50, 75, 95])
    print(
        f"  {name:12s}  n={areas.size:>7,}  "
        f"median={p[2]:>12,.1f} m²  p95={p[4]:>14,.1f} m²  max={areas.max():>14,.1f} m²"
    )


def plot_histogram(corpus: np.ndarray, queries: np.ndarray, out_path: str, bins: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    combined = np.concatenate([corpus, queries])
    x_lo = np.percentile(combined, 1)
    x_hi = np.percentile(combined, 99)
    log_lo = math.floor(np.log10(x_lo))
    log_hi = math.ceil(np.log10(x_hi))
    zoom_bins = np.logspace(log_lo, log_hi, bins)

    ax = axes[0]
    corpus_hist, _ = np.histogram(corpus, bins=zoom_bins, density=True)
    query_hist, _ = np.histogram(queries, bins=zoom_bins, density=True)
    y_hi = max(corpus_hist.max(), query_hist.max()) * 1.08

    ax.hist(
        corpus,
        bins=zoom_bins,
        alpha=0.55,
        density=True,
        label=f"Corpus (n={corpus.size:,})",
        color="C0",
    )
    ax.hist(
        queries,
        bins=zoom_bins,
        alpha=0.55,
        density=True,
        label=f"GT queries (n={queries.size:,})",
        color="C1",
    )
    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_hi)
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Density")
    ax.set_title(f"Area distribution (p1–p99: {x_lo:,.0f} – {x_hi:,.0f} m²)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    log_corpus = np.log10(corpus)
    log_queries = np.log10(queries)
    log_combined = np.concatenate([log_corpus, log_queries])
    log_x_lo, log_x_hi = np.percentile(log_combined, [1, 99])
    log_bin_edges = np.linspace(log_x_lo, log_x_hi, bins + 1)

    ax = axes[1]
    ax.hist(
        log_corpus,
        bins=log_bin_edges,
        alpha=0.45,
        density=True,
        label="Corpus",
        color="C0",
    )
    ax.hist(
        log_queries,
        bins=log_bin_edges,
        histtype="step",
        linewidth=2.0,
        density=True,
        label="GT queries (outline)",
        color="C1",
    )
    ax.set_xlim(log_x_lo, log_x_hi)
    ax.set_xlabel("log₁₀(area m²)")
    ax.set_ylabel("Density")
    ax.set_title("Log-area distribution (corpus filled, GT outline)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.suptitle("Parks polygon area: corpus vs GT query split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved histogram → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wkt-path", default=WKT_PATH)
    parser.add_argument("--encoding-dir", default=ENCODING_DIR, help="For reference only; geometry comes from WKT.")
    parser.add_argument("--gt-dir", default=GT_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max-polygons", type=int, default=None, help="Cap for quick tests.")
    parser.add_argument(
        "--from-npz",
        default=None,
        help="Skip WKT load; plot from saved .npz (e.g. area_distribution/polygon_areas.npz).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hist_path = os.path.join(args.out_dir, "polygon_area_histogram.png")

    if args.from_npz:
        data = np.load(args.from_npz)
        corpus = data["corpus_m2"]
        query_start = corpus.size
        queries = data["query_m2"]
        print(f"Loaded from {args.from_npz}")
        print(f"  corpus n={corpus.size:,}, queries n={queries.size:,}")
        print("\nSummary:")
        print_stats("corpus", corpus)
        print_stats("queries", queries)
        plot_histogram(corpus, queries, hist_path, args.bins)
        return

    query_start, total = discover_split(args.gt_dir)
    limit = args.max_polygons if args.max_polygons else total
    if limit < query_start + 1:
        sys.exit(f"--max-polygons must be ≥ {query_start + 1} to include query polygons.")

    print(f"Encoding dir : {args.encoding_dir}")
    print(f"GT dir       : {args.gt_dir}")
    print(f"Split        : corpus [0, {query_start})  |  queries [{query_start}, {total})  ({query_start/total:.0%} / {1-query_start/total:.0%})")
    print(f"WKT source   : {args.wkt_path}")
    print(f"Loading {limit:,} polygon areas...")

    areas = load_areas(args.wkt_path, limit)
    corpus = areas[:query_start]
    queries = areas[query_start:limit]

    print("\nSummary:")
    print_stats("corpus", corpus)
    print_stats("queries", queries)

    plot_histogram(corpus, queries, hist_path, args.bins)

    npz_path = os.path.join(args.out_dir, "polygon_areas.npz")
    np.savez_compressed(npz_path, corpus_m2=corpus, query_m2=queries)
    print(f"  saved areas      → {npz_path}")


if __name__ == "__main__":
    main()
