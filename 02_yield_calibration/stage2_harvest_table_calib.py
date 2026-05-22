"""
02_yield_calibration/stage2_harvest_table_calib.py

Stage 2 of the two-stage yield monitor calibration pipeline.

Applies a per-field multiplicative correction factor so that the mean of
the Stage 1 yield map exactly matches the harvest-table mean:

    calib_factor = HT_mean / map_mean
    y_calibrated = y_stage1 * calib_factor

Fields with a calib_factor outside [0.85, 1.05] are flagged and excluded
from modelling. The acceptance check matches the thesis QC criteria.

Input:  Stage 1 corrected GeoPackages + harvest table CSV
Output: Final calibrated GeoPackages with 'yield_calib_tha' column

Usage:
    python stage2_harvest_table_calib.py
    (edit STAGE1_DIR and HARVEST_CSV at the bottom)
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

# Allow imports from parent directory (utils)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, FIELD_TO_TABLA


# ── Acceptance window for calibration factors ─────────────────────────────
FACTOR_MIN = 0.85
FACTOR_MAX = 1.05

# ── Yield column in Stage 1 output ───────────────────────────────────────
STAGE1_YIELD_COL = "yield_s1"

# ── Per-field GeoPackage names in Stage 1 output ─────────────────────────
FIELD_GPKG = {
    "7":    "7_yield_s1.gpkg",
    "9_ce": "9_ce_yield_s1.gpkg",
    "9_lg": "9_lg_yield_s1.gpkg",
    "9_pr": "9_pr_yield_s1.gpkg",
    "9_sy": "9_sy_yield_s1.gpkg",
    "12":   "12_yield_s1.gpkg",
    "25":   "25_yield_s1.gpkg",
    "44":   "44_yield_s1.gpkg",
    "59":   "59_yield_s1.gpkg",
    "63":   "63_yield_s1.gpkg",
    "71":   "71_yield_s1.gpkg",
    "79":   "79_yield_s1.gpkg",
    "84":   "84_yield_s1.gpkg",
}


def calibrate_stage2(stage1_dir, harvest_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    harvest_means = load_harvest_means(harvest_csv)

    print("=" * 60)
    print("  STAGE 2 — HARVEST TABLE CALIBRATION")
    print("=" * 60)
    print(f"\n{'Field':<8} {'Map mean':>10} {'HT mean':>10} "
          f"{'Factor':>8} {'Calib mean':>12} {'Status':>12}")
    print("-" * 60)

    summary = []

    for fid, gpkg_name in FIELD_GPKG.items():
        fpath = os.path.join(stage1_dir, gpkg_name)
        if not os.path.exists(fpath):
            print(f"  {fid:<6} — file not found, skipped")
            continue

        gdf = gpd.read_file(fpath)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:23700")

        if STAGE1_YIELD_COL not in gdf.columns:
            # Fall back to any numeric yield-looking column
            from utils.data_io import detect_yield_col
            col = detect_yield_col(gdf)
            if col is None:
                print(f"  {fid:<6} — no yield column found, skipped")
                continue
        else:
            col = STAGE1_YIELD_COL

        y_raw = pd.to_numeric(gdf[col], errors="coerce").values.astype(float)
        if np.nanmean(y_raw) > 50:
            y_raw = y_raw / 1000.0

        valid = np.isfinite(y_raw) & (y_raw > 0.5) & (y_raw < 20.0)
        map_mean = float(np.mean(y_raw[valid]))

        tabla   = FIELD_TO_TABLA.get(fid)
        ht_mean = harvest_means.get(fid) or harvest_means.get(tabla)
        if ht_mean is None:
            print(f"  {fid:<6} — not in harvest table, skipped")
            continue

        calib_factor = ht_mean / map_mean
        y_calib = y_raw * calib_factor
        calib_mean = float(np.mean(y_calib[valid]))

        # Status flag
        if FACTOR_MIN <= calib_factor <= FACTOR_MAX:
            status = "OK"
        elif 0.75 <= calib_factor < FACTOR_MIN:
            status = "LARGE GAP"
        elif calib_factor > FACTOR_MAX:
            status = "ANOMALY"
        else:
            status = "VERY LARGE"

        print(f"  {fid:<8} {map_mean:>10.3f} {ht_mean:>10.3f} "
              f"{calib_factor:>8.4f} {calib_mean:>12.3f} {status:>12}")

        # Save output (include original + calibrated columns)
        gdf_out = gdf.copy()
        gdf_out["yield_raw_tha"]   = y_raw
        gdf_out["yield_calib_tha"] = y_calib
        gdf_out["calib_factor"]    = calib_factor
        gdf_out["ht_mean_tha"]     = ht_mean

        out_path = os.path.join(out_dir, f"{fid}_yield_10px_calib.gpkg")
        gdf_out.to_file(out_path, driver="GPKG")

        summary.append({
            "field":        fid,
            "tabla":        tabla,
            "map_mean":     map_mean,
            "ht_mean":      ht_mean,
            "calib_factor": calib_factor,
            "calib_mean":   calib_mean,
            "n_pts":        int(valid.sum()),
            "status":       status,
        })

    # Save summary table
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("Field quality flags:")
    for _, row in df.iterrows():
        f = row["calib_factor"]
        flag = (
            "use in training"    if FACTOR_MIN <= f <= FACTOR_MAX else
            "check carefully"    if 0.75 <= f < FACTOR_MIN else
            "EXCLUDE — anomaly"
        )
        print(f"  {row['field']:<8}  factor={f:.4f}  {flag}")

    accepted = df[df["calib_factor"].between(FACTOR_MIN, FACTOR_MAX)]["field"].tolist()
    print(f"\nAccepted fields ({len(accepted)}): {accepted}")
    print(f"Summary saved → {os.path.join(out_dir, 'calibration_summary.csv')}")


# ── Paths ─────────────────────────────────────────────────────────────────
# Edit these to match your local directory layout.

STAGE1_DIR  = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\stage1_corrected"
HARVEST_CSV = r"D:\STUDI\Thesis\mezohegyes\obuza_napi_aratas_2025_fix.csv"
OUT_DIR     = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"


if __name__ == "__main__":
    calibrate_stage2(STAGE1_DIR, HARVEST_CSV, OUT_DIR)
