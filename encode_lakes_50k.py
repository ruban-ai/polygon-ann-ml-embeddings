#!/usr/bin/env python3
"""Encode the first 50K polygon records from the OSM lakes dataset.

This is a thin runner around /raid/buddhi/ssQuant/encodeWeightedFileWriting.py.
It leaves the encoder's core logic unchanged and only adapts the input source.
"""

import importlib.util
import os
import sys
import time

import shapely.wkt


ENCODER_DIR = "/raid/buddhi/ssQuant"
ENCODER_PATH = os.path.join(ENCODER_DIR, "encodeWeightedFileWriting.py")
ENCODER_LIB_DIR = os.path.join(ENCODER_DIR, "lib")
LAKES_PATH = "/raid/ssEncodingData/polygonalData/osm_new/lakes"
OUTPUT_DIR = "/raid/ruban/hpmlproj/encoding/lakes-real50k0.012"

TOTAL_POLYGONS = 50_000
CORPUS_POLYGONS = 40_000
QUERY_POLYGONS = 10_000


def load_encoder():
    old_cwd = os.getcwd()
    try:
        os.chdir(ENCODER_DIR)
        sys.path.insert(0, ENCODER_DIR)
        sys.path.insert(0, ENCODER_LIB_DIR)
        spec = importlib.util.spec_from_file_location(
            "encodeWeightedFileWriting", ENCODER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def read_lake_polygons(path, limit):
    polygons = []
    skipped = 0
    rows_seen = 0

    with open(path, encoding="utf-8") as infile:
        for line in infile:
            rows_seen += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                skipped += 1
                continue

            try:
                geom = shapely.wkt.loads(parts[1])
            except Exception:
                skipped += 1
                continue

            if geom.geom_type != "Polygon" or geom.is_empty:
                skipped += 1
                continue

            polygons.append(geom)
            if len(polygons) == limit:
                break

    print(
        "Loaded {} polygons from {} rows; skipped {} non-polygon/bad rows.".format(
            len(polygons), rows_seen, skipped
        )
    )
    return polygons


def main():
    start_time = time.time()
    encoder = load_encoder()

    wkt_list = read_lake_polygons(LAKES_PATH, TOTAL_POLYGONS)
    if len(wkt_list) < TOTAL_POLYGONS:
        raise RuntimeError(
            "Only found {} polygon records, expected {}.".format(
                len(wkt_list), TOTAL_POLYGONS
            )
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, "real")

    data_start = 0
    data_end = CORPUS_POLYGONS
    query_start = data_end
    query_end = data_end + QUERY_POLYGONS
    data_percent = CORPUS_POLYGONS / TOTAL_POLYGONS
    node_capacity_percent = 0.012
    th_area = 0
    foreground = "area"
    process_count = 200
    file_size = 1000

    print("Data: [{} to {}]. Query: [{} to {}]".format(
        data_start, data_end, query_start, query_end
    ))
    print("Saving to {}.".format(OUTPUT_DIR))

    encoder.writeWKTListToSparseMatrix(
        filename,
        file_size,
        wkt_list,
        data_start,
        data_end,
        query_start,
        query_end,
        data_percent,
        node_capacity_percent,
        th_area,
        foreground,
        process_count,
    )

    print("Total time: {} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    main()
