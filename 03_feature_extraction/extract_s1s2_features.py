"""
03_feature_extraction/extract_s1s2_features.py

Samples Sentinel-1 and Sentinel-2 spectral index rasters at the centroid
coordinates of each yield pixel. Output is one GeoPackage per field with
all S1/S2 features appended as columns.

Expected raster naming convention:
    {sensor}_{YYYYMMDD}_Mezohegyes_Stacked_{index_name}.tif
    e.g. S2_20250614_Mezohegyes_Stacked_NDVI.tif

The script auto-discovers all matching TIF files in S1S2_DIR.
An optional ALLOWED_DATES filter restricts to the heading-to-harvest window
used for the main model comparison (330-feature input). Set ALLOWED_DATES
to None to include all dates (feature comparison experiment).

Usage:
    python extract_s1s2_features.py
    (edit paths and ALLOWED_DATES at the bottom)
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, FIELD_GPKG_CALIB
from utils.raster_sampling import sample_single_band, get_xy


# ── Raster filename pattern ───────────────────────────────────────────────
FNAME_PATTERN = re.compile(
    r"(S[12])_(\d{8})_Mezohegyes_Stacked_([A-Za-z0-9_]+)\.tif$"
)

# ── Heading-to-harvest dates (used for 330-feature model comparison) ─────
# Set to None to use all available dates (feature set comparison).
HEADING_TO_HARVEST_DATES = {
    "20250614", "20250621", "20250622",
    "20250714", "20250721", "20250724",
}


def discover_s1s2(s1s2_dir, allowed_dates=None):
    """
    Scan the raster directory and return a dict {feature_key: filepath}.
    Optionally filtered to allowed_dates (set of 'YYYYMMDD' strings).
    """
    found = {}
    for fname in sorted(os.listdir(s1s2_dir)):
        if not fname.endswith(".tif"):
            continue
        m = FNAME_PATTERN.match(fname)
        if m is None:
            continue
        sensor, date, index = m.group(1), m.group(2), m.group(3)
        if allowed_dates is not None and date not in allowed_dates:
            continue
        key = f"{sensor}_{date}_{index}"
        found[key] = os.path.join(s1s2_dir, fname)
    return found


def extract_features_for_field(gdf, s1s2_files):
    """
    Sample all S1/S2 rasters at field pixel centroids.
    Returns (X array of shape [N, n_features], feature_names list).
    """
    xs, ys = get_xy(gdf)
    blocks, names = [], []

    for key, fpath in s1s2_files.items():
        v = sample_single_band(fpath, xs, ys)
        blocks.append(v[:, None])
        names.append(key)

    if len(blocks) == 0:
        return np.empty((len(gdf), 0), dtype=np.float32), []

    X = np.concatenate(blocks, axis=1)
    return X, names


def run_extraction(yield_dir, gpkg_map, s1s2_dir, out_dir, allowed_dates=None):
    os.makedirs(out_dir, exist_ok=True)

    s1s2_files = discover_s1s2(s1s2_dir, allowed_dates)
    n_dates = len(set(k.split("_")[1] for k in s1s2_files))
    print(f"S1/S2 rasters found: {len(s1s2_files)} features across {n_dates} dates")

    for fid, gpkg_name in gpkg_map.items():
        fpath = os.path.join(yield_dir, gpkg_name)
        if not os.path.exists(fpath):
            print(f"  [skip] {fid} — calibrated GeoPackage not found")
            continue

        print(f"\n  {fid} ...")
        gdf = gpd.read_file(fpath)

        X, feat_names = extract_features_for_field(gdf, s1s2_files)

        # Attach features as columns
        feat_df = pd.DataFrame(X, columns=feat_names, index=gdf.index)
        gdf_out = pd.concat([gdf, feat_df], axis=1)

        out_path = os.path.join(out_dir, f"{fid}_s1s2_features.gpkg")
        gdf_out.to_file(out_path, driver="GPKG")
        print(f"    {len(gdf):,} pts | {X.shape[1]} features → {out_path}")

    print("\nS1/S2 feature extraction done.")


# ── Field GeoPackage map (calibrated yield files from Stage 2) ───────────
FIELD_GPKG_CALIB = {
    "7":    "7_yield_10px_calib.gpkg",
    "9_ce": "9_ce_yield_10px_calib.gpkg",
    "9_lg": "9_lg_yield_10px_calib.gpkg",
    "9_pr": "9_pr_yield_10px_calib.gpkg",
    "9_sy": "9_sy_yield_10px_calib.gpkg",
    "12":   "12_yield_10px_calib.gpkg",
    "25":   "25_yield_10px_calib.gpkg",
    "44":   "44_yield_10px_calib.gpkg",
    "59":   "59_yield_10px_calib.gpkg",
    "63":   "63_yield_10px_calib.gpkg",
    "71":   "71_yield_10px_calib.gpkg",
    "79":   "79_yield_10px_calib.gpkg",
}

# ── Paths ─────────────────────────────────────────────────────────────────
YIELD_DIR = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
S1S2_DIR  = r"D:\STUDI\Thesis\mezohegyes\VIs\s1+s2"
OUT_DIR   = r"D:\STUDI\Thesis\mezohegyes\features\s1s2"


if __name__ == "__main__":
    # Change HEADING_TO_HARVEST_DATES to None for the full feature comparison.
    run_extraction(
        yield_dir     = YIELD_DIR,
        gpkg_map      = FIELD_GPKG_CALIB,
        s1s2_dir      = S1S2_DIR,
        out_dir       = OUT_DIR,
        allowed_dates = HEADING_TO_HARVEST_DATES,
    )
