#!/usr/bin/env python3
"""Compare polygon area distributions for the full Parks corpus/query split.

The encoding and GT directories contain derived vectors / neighbor ids, not WKT
geometry. This script uses the GT filename range to recover the query split and
then reads the corresponding polygons from the raw Parks WKT file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shapely.wkt
from pyproj import Geod
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


DEFAULT_WKT = "/raid/ssEncodingData/polygonalData/osm_new/parks"
DEFAULT_ENCODING_DIR = "/raid/ssEncodingData/encoding/pk-real0.002"
DEFAULT_GT_DIR = "/raid/ssEncodingData/warehouse/pk-query-187019"
DEFAULT_OUT_DIR = "/raid/ruban/hpmlproj/term_project/area_distribution"

GT_RE = re.compile(r"similarityMap_(\d+)-(\d+)(?:\.npy)?$")
GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Split:
    corpus_start: int
    corpus_end: int
    query_start: int
    query_end: int

    @property
    def total(self) -> int:
        return self.query_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot area distributions for corpus vs GT query polygons."
    )
    parser.add_argument("--wkt-path", default=DEFAULT_WKT)
    parser.add_argument("--encoding-dir", default=DEFAULT_ENCODING_DIR)
    parser.add_argument("--gt-dir", default=DEFAULT_GT_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-polygons",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--sample-per-split",
        type=int,
        default=None,
        help="Randomly sample this many areas per split for plotting only.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress after this many valid polygons; 0 disables it.",
    )
    return parser.parse_args()


def discover_split(gt_dir: str) -> Split:
    ranges: list[tuple[int, int]] = []
    for path in glob.glob(os.path.join(gt_dir, "similarityMap_*")):
        match = GT_RE.search(os.path.basename(path))
        if not match:
            continue
        ranges.append((int(match.group(1)), int(match.group(2))))

    if not ranges:
        raise FileNotFoundError(f"No similarityMap_* files found in {gt_dir}")

    query_start = min(start for start, _ in ranges)
    query_end = max(end for _, end in ranges) + 1
    return Split(
        corpus_start=0,
        corpus_end=query_start,
        query_start=query_start,
        query_end=query_end,
    )


def polygonal_parts(geom) -> Iterable[Polygon]:
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from polygonal_parts(part)


def area_square_meters(geom) -> float:
    area = 0.0
    for poly in polygonal_parts(geom):
        try:
            part_area, _ = GEOD.geometry_area_perimeter(poly)
            area += abs(part_area)
        except Exception:
            area += 0.0
    return area


def read_areas(
    wkt_path: str, limit: int, progress_every: int
) -> tuple[np.ndarray, int, int]:
    areas: list[float] = []
    rows_seen = 0
    skipped = 0

    with open(wkt_path, encoding="utf-8") as infile:
        for line in infile:
            rows_seen += 1
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 2:
                skipped += 1
                continue

            try:
                geom = shapely.wkt.loads(parts[1])
                area = area_square_meters(geom)
            except Exception:
                skipped += 1
                continue

            if not math.isfinite(area) or area <= 0.0:
                skipped += 1
                continue

            areas.append(area)
            if progress_every and len(areas) % progress_every == 0:
                print(
                    f"Loaded {len(areas):,}/{limit:,} valid polygons "
                    f"(rows seen={rows_seen:,}, skipped={skipped:,})",
                    flush=True,
                )
            if len(areas) >= limit:
                break

    return np.asarray(areas, dtype=np.float64), rows_seen, skipped


def summarize(name: str, values: np.ndarray) -> dict[str, float | int | str]:
    pct = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "split": name,
        "count": int(values.size),
        "min_m2": float(np.min(values)),
        "p01_m2": float(pct[0]),
        "p05_m2": float(pct[1]),
        "p25_m2": float(pct[2]),
        "median_m2": float(pct[3]),
        "p75_m2": float(pct[4]),
        "p95_m2": float(pct[5]),
        "p99_m2": float(pct[6]),
        "max_m2": float(np.max(values)),
        "mean_m2": float(np.mean(values)),
        "std_m2": float(np.std(values)),
    }


def write_summary(path: str, summaries: list[dict[str, float | int | str]]) -> None:
    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def maybe_sample(values: np.ndarray, n: int | None, rng: np.random.Generator) -> np.ndarray:
    if n is None or values.size <= n:
        return values
    idx = rng.choice(values.size, size=n, replace=False)
    return values[idx]


def plot_histograms(
    corpus: np.ndarray,
    query: np.ndarray,
    out_path: str,
    bins: int,
    sample_per_split: int | None,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    corpus_plot = maybe_sample(corpus, sample_per_split, rng)
    query_plot = maybe_sample(query, sample_per_split, rng)

    all_values = np.concatenate([corpus_plot, query_plot])
    log_min = math.floor(np.log10(np.min(all_values)))
    log_max = math.ceil(np.log10(np.max(all_values)))
    log_bins = np.logspace(log_min, log_max, bins)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    axes[0].hist(
        corpus_plot,
        bins=log_bins,
        alpha=0.55,
        density=True,
        label=f"Corpus (n={corpus.size:,})",
    )
    axes[0].hist(
        query_plot,
        bins=log_bins,
        alpha=0.55,
        density=True,
        label=f"GT queries (n={query.size:,})",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Area (m^2, log scale)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Area distribution")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    corpus_log = np.log10(corpus_plot)
    query_log = np.log10(query_plot)
    axes[1].hist(
        corpus_log,
        bins=bins,
        alpha=0.55,
        density=True,
        label="Corpus",
    )
    axes[1].hist(
        query_log,
        bins=bins,
        alpha=0.55,
        density=True,
        label="GT queries",
    )
    axes[1].set_xlabel("log10(area m^2)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log-area distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Parks polygon area distribution: corpus vs GT query split")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    split = discover_split(args.gt_dir)
    limit = split.total if args.max_polygons is None else min(split.total, args.max_polygons)

    print(f"Encoding dir: {args.encoding_dir}")
    print(f"GT dir:       {args.gt_dir}")
    print(
        "Derived split: corpus=[{}, {}) query=[{}, {}) total={:,}".format(
            split.corpus_start,
            split.corpus_end,
            split.query_start,
            split.query_end,
            split.total,
        )
    )
    print(f"Reading {limit:,} polygons from {args.wkt_path}")

    areas, rows_seen, skipped = read_areas(
        args.wkt_path, limit, args.progress_every
    )
    if areas.size < limit:
        raise RuntimeError(
            f"Only loaded {areas.size:,} valid polygons from {rows_seen:,} rows; "
            f"needed {limit:,}. Skipped {skipped:,} rows."
        )

    if limit <= split.query_start:
        raise RuntimeError(
            "--max-polygons stopped before query polygons. Use at least "
            f"{split.query_start + 1:,}."
        )

    corpus = areas[split.corpus_start : min(split.corpus_end, limit)]
    query = areas[split.query_start : limit]

    summary_path = os.path.join(args.out_dir, "area_distribution_summary.csv")
    hist_path = os.path.join(args.out_dir, "area_distribution_hist.png")
    npz_path = os.path.join(args.out_dir, "area_distribution_areas.npz")

    summaries = [summarize("corpus", corpus), summarize("gt_queries", query)]
    write_summary(summary_path, summaries)
    np.savez_compressed(npz_path, corpus_m2=corpus, gt_query_m2=query)
    plot_histograms(
        corpus,
        query,
        hist_path,
        args.bins,
        args.sample_per_split,
        args.seed,
    )

    print(f"Rows seen: {rows_seen:,}; skipped before limit: {skipped:,}")
    for row in summaries:
        print(
            "{split}: n={count:,} median={median_m2:,.2f} m^2 "
            "p95={p95_m2:,.2f} m^2 max={max_m2:,.2f} m^2".format(**row)
        )
    print(f"Summary CSV: {summary_path}")
    print(f"Area arrays:  {npz_path}")
    print(f"Histogram:    {hist_path}")


if __name__ == "__main__":
    main()
