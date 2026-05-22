"""
01_yield_cleaning/clean_yield_points.py

Cleans raw combine harvester GeoJSON point files before calibration.

Two filtering passes are applied:
  1. Logical filters: remove zero/negative yields, stopped-machine points,
     and zero-distance records that would inflate mass calculations.
  2. Statistical filter: IQR-based outlier removal to drop extreme spikes
     caused by sensor glitches (e.g. grain flow sensor clearing).

Output is a cleaned GeoJSON written to the same directory with a
'_clean' suffix. Run this once per field before stage 1 calibration.

Usage:
    python clean_yield_points.py
    (edit IN_FILES at the bottom to match your paths)
"""

import os
import geopandas as gpd
import numpy as np


# ── Speed threshold for detecting stopped-machine records ────────────────
# Points with vehicle speed below this (km/h) are boundary/turn artefacts.
MIN_SPEED_KMH = 2.0

# ── Yield plausibility range (t/ha) ──────────────────────────────────────
# Values outside this are clearly sensor errors (negative, or >20 t/ha for wheat).
MIN_YIELD = 0.5
MAX_YIELD = 20.0


def clean_field(input_path, output_path):
    print(f"\n--- {os.path.basename(input_path)} ---")

    try:
        gdf = gpd.read_file(input_path)
    except Exception as e:
        print(f"  ERROR: could not read file — {e}")
        return

    n_start = len(gdf)

    # --- Pass 1: logical filters ---
    mask = (
        (gdf["DISTANCE"] > 0) &
        (gdf["VRYIELDMAS"] > 0) &
        (gdf["VEHICLSPEE"] > MIN_SPEED_KMH)
    )
    gdf = gdf[mask].copy()
    n_logic = n_start - len(gdf)

    # --- Pass 2: IQR outlier filter on yield ---
    n_stat = 0
    if len(gdf) > 20:
        q1 = gdf["VRYIELDMAS"].quantile(0.25)
        q3 = gdf["VRYIELDMAS"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        before = len(gdf)
        gdf = gdf[
            (gdf["VRYIELDMAS"] >= lower) & (gdf["VRYIELDMAS"] <= upper)
        ].copy()
        n_stat = before - len(gdf)

    # --- Report ---
    print(f"  Input:   {n_start:,} points")
    print(f"  Removed (logical):     {n_logic:,}")
    print(f"  Removed (IQR outlier): {n_stat:,}")
    print(f"  Remaining:             {len(gdf):,}")

    if len(gdf) == 0:
        print("  WARN: no points survived — check input data")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"  Saved → {output_path}")


# ── Field list ─────────────────────────────────────────────────────────────
# Edit these paths to match your local file layout.

BASE = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\yield_raw"
OUT  = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\yield_cleaned"

IN_FILES = {
    "7":    os.path.join(BASE, "7_raw.geojson"),
    "9_ce": os.path.join(BASE, "9_ce_raw.geojson"),
    "9_lg": os.path.join(BASE, "9_lg_raw.geojson"),
    "9_pr": os.path.join(BASE, "9_pr_raw.geojson"),
    "9_sy": os.path.join(BASE, "9_sy_raw.geojson"),
    "12":   os.path.join(BASE, "12_raw.geojson"),
    "25":   os.path.join(BASE, "25_raw.geojson"),
    "44":   os.path.join(BASE, "44_raw.geojson"),
    "59":   os.path.join(BASE, "59_raw.geojson"),
    "63":   os.path.join(BASE, "63_raw.geojson"),
    "71":   os.path.join(BASE, "71_raw.geojson"),
    "79":   os.path.join(BASE, "79_raw.geojson"),
    "84":   os.path.join(BASE, "84_raw.geojson"),
}


if __name__ == "__main__":
    print("=" * 50)
    print("  YIELD POINT CLEANING")
    print("=" * 50)

    for fid, in_path in IN_FILES.items():
        out_path = os.path.join(OUT, f"{fid}_clean.geojson")
        clean_field(in_path, out_path)

    print("\nDone.")
