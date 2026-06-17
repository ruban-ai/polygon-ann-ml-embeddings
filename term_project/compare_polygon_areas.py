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


def decade_table(corpus: np.ndarray, queries: np.ndarray) -> None:
    """Print % of polygons in each log10 area decade."""
    print("\nFraction per log10 decade (% of each split):")
    print(f"  {'decade':>14s}  {'corpus':>8s}  {'queries':>8s}  {'gap':>8s}")
    for d in range(-1, 12):
        lo, hi = d, d + 1
        c_pct = 100 * ((np.log10(corpus) >= lo) & (np.log10(corpus) < hi)).sum() / corpus.size
        q_pct = 100 * ((np.log10(queries) >= lo) & (np.log10(queries) < hi)).sum() / queries.size
        if c_pct < 0.005 and q_pct < 0.005:
            continue
        print(f"  10^{d:<2d}–10^{d+1:<2d} m²  {c_pct:7.2f}%  {q_pct:7.2f}%  {q_pct - c_pct:+7.2f}pp")


def shared_log_bins(corpus: np.ndarray, queries: np.ndarray, bins: int) -> np.ndarray:
    """Log10-uniform bin edges shared by both panels (p1–p99)."""
    combined = np.concatenate([corpus, queries])
    log_lo = np.percentile(np.log10(combined), 1)
    log_hi = np.percentile(np.log10(combined), 99)
    return np.linspace(log_lo, log_hi, bins + 1)


def frac_per_bin(values: np.ndarray, log_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin centers in m², fraction in each bin)."""
    counts, _ = np.histogram(np.log10(values), bins=log_edges)
    centers = 10 ** ((log_edges[:-1] + log_edges[1:]) / 2)
    return centers, counts / values.size


def plot_histogram(corpus: np.ndarray, queries: np.ndarray, out_path: str, bins: int) -> None:
    log_edges = shared_log_bins(corpus, queries, bins)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    area_centers = 10**log_centers
    log_step = log_edges[1] - log_edges[0]

    c_frac, _ = np.histogram(np.log10(corpus), bins=log_edges)
    q_frac, _ = np.histogram(np.log10(queries), bins=log_edges)
    c_pct = 100 * c_frac / corpus.size
    q_pct = 100 * q_frac / queries.size

    # density per unit log10(area) — same quantity both panels show
    c_dens = c_frac / (corpus.size * log_step)
    q_dens = q_frac / (queries.size * log_step)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: area (m²) log-x, log-y, % per bin (grouped bars) ---
    ax = axes[0]
    half = log_step * 0.18
    c_x = 10 ** (log_centers - half)
    q_x = 10 ** (log_centers + half)
    bar_w = area_centers * (10**half - 10**(-half))
    y_floor = 0.01  # floor for log-y (%)

    ax.bar(c_x, np.maximum(c_pct, y_floor), width=bar_w, color="C0", alpha=0.85, label="Corpus")
    ax.bar(q_x, np.maximum(q_pct, y_floor), width=bar_w, color="C1", alpha=0.85, label="GT queries")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10**log_edges[0], 10**log_edges[-1])
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("% of polygons in bin (log scale)")
    ax.set_title("Panel A: % per bin\n(same log10 bins as Panel B)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    # --- Panel 2: log10(area), density per log10 unit (aligned with A) ---
    ax = axes[1]
    ax.bar(
        log_centers - half,
        c_dens,
        width=log_step * 0.36,
        alpha=0.45,
        color="C0",
        label="Corpus",
    )
    ax.step(
        log_edges,
        np.concatenate([q_dens, [q_dens[-1]]]),
        where="post",
        linewidth=2.0,
        color="C1",
        label="GT queries (outline)",
    )
    ax.set_xlim(log_edges[0], log_edges[-1])
    ax.set_xlabel("log₁₀(area m²)")
    ax.set_ylabel("Density (per unit log₁₀ area)")
    ax.set_title("Panel B: density in log10 space\n(corpus bars + GT outline)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # --- Panel 3: ECDF — clearest "many small, few large" view ---
    ax = axes[2]
    for arr, label, color in [
        (corpus, f"Corpus (n={corpus.size:,})", "C0"),
        (queries, f"GT queries (n={queries.size:,})", "C1"),
    ]:
        sorted_a = np.sort(arr)
        y = np.arange(1, sorted_a.size + 1) / sorted_a.size
        ax.plot(sorted_a, y, color=color, linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Fraction of polygons ≤ this area")
    ax.set_title("Panel C: cumulative distribution\n(how many are smaller than X)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Parks polygon areas: corpus vs GT queries\n"
        "A & B use identical log10 bins — A shows %/bin, B shows density/log10 unit",
        fontsize=11,
    )
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
