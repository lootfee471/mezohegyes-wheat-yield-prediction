"""
03_feature_extraction/extract_enmap_terrain.py

Samples EnMAP hyperspectral bands, DEM, TWI, and irrigation flag at
yield pixel centroid coordinates. Appends these to the S1/S2 feature
GeoPackages produced by extract_s1s2_features.py, or creates a separate
ancillary feature file that can be merged at modelling time.

EnMAP: 219 usable bands (noisy bands already removed in the Level-2A product)
        + 8 vegetation indices computed in EnMAP-Box
DEM:   single band (elevation in metres)
TWI:   single band (topographic wetness index)
Irrigation: binary flag from estate irrigation polygon layer

Usage:
    python extract_enmap_terrain.py
    (edit paths at the bottom)
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.raster_sampling import sample_single_band, sample_all_bands, get_xy


# ── EnMAP index TIF naming pattern ───────────────────────────────────────
# Expected filenames: e.g. GVMI.tif, MSI.tif, NDWI.tif, etc.
ENMAP_INDEX_NAMES = ["ARI1", "gNDVI", "hNDVI", "TCARI", "GVMI", "MSI", "NDWI", "SWIRVI"]


def extract_irrigation_flag(gdf, irrig_gpkg):
    """
    Assign a binary irrigation flag (1 = irrigated, 0 = not irrigated)
    by checking whether each pixel centroid falls inside the irrigation
    polygon layer.
    """
    if not os.path.exists(irrig_gpkg):
        print("  [warn] irrigation polygon not found, setting flag to 0")
        return np.zeros(len(gdf), dtype=np.float32)

    irrig = gpd.read_file(irrig_gpkg)
    if gdf.crs != irrig.crs:
        irrig = irrig.to_crs(gdf.crs)

    # Spatial join: points that fall inside irrigation polygons get flag=1
    pts = gdf[["geometry"]].copy()
    pts["__idx"] = np.arange(len(pts))

    joined = gpd.sjoin(pts, irrig[["geometry"]], how="left", predicate="within")
    flag = np.zeros(len(gdf), dtype=np.float32)
    irrigated_idx = joined.dropna(subset=["index_right"])["__idx"].values
    flag[irrigated_idx.astype(int)] = 1.0

    n_irr = int(flag.sum())
    print(f"    Irrigation: {n_irr}/{len(gdf)} pixels flagged as irrigated")
    return flag


def extract_ancillary(gdf, enmap_path, enmap_index_dir, dem_path, twi_path, irrig_gpkg):
    """
    Build ancillary feature matrix for one field.
    Returns (X, feature_names).
    """
    xs, ys = get_xy(gdf)
    blocks, names = [], []

    # EnMAP all raw bands
    if os.path.exists(enmap_path):
        ev = sample_all_bands(enmap_path, xs, ys)
        if ev is not None:
            n_bands = ev.shape[1]
            blocks.append(ev)
            names.extend([f"ENMAP_b{b+1}" for b in range(n_bands)])
            print(f"    EnMAP raw: {n_bands} bands")
        else:
            print("    [warn] EnMAP sampling failed")
    else:
        print("    [warn] EnMAP file not found")

    # EnMAP vegetation indices
    for idx_name in ENMAP_INDEX_NAMES:
        idx_path = os.path.join(enmap_index_dir, f"{idx_name}.tif")
        v = sample_single_band(idx_path, xs, ys)
        blocks.append(v[:, None])
        names.append(f"ENMAP_{idx_name}")
    print(f"    EnMAP indices: {len(ENMAP_INDEX_NAMES)}")

    # DEM
    v = sample_single_band(dem_path, xs, ys)
    blocks.append(v[:, None])
    names.append("DEM")
    if os.path.exists(dem_path):
        print(f"    DEM: mean={np.nanmean(v):.2f} m")

    # TWI
    v = sample_single_band(twi_path, xs, ys)
    blocks.append(v[:, None])
    names.append("TWI")
    if os.path.exists(twi_path):
        print(f"    TWI: mean={np.nanmean(v):.2f}")

    # Irrigation flag
    irr = extract_irrigation_flag(gdf, irrig_gpkg)
    blocks.append(irr[:, None])
    names.append("Irrigation")

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, names


def run_extraction(yield_dir, gpkg_map, enmap_path, enmap_index_dir,
                   dem_path, twi_path, irrig_gpkg, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    for fid, gpkg_name in gpkg_map.items():
        fpath = os.path.join(yield_dir, gpkg_name)
        if not os.path.exists(fpath):
            print(f"  [skip] {fid} — calibrated GeoPackage not found")
            continue

        print(f"\n  {fid} ...")
        gdf = gpd.read_file(fpath)

        X, feat_names = extract_ancillary(
            gdf, enmap_path, enmap_index_dir, dem_path, twi_path, irrig_gpkg
        )

        feat_df = pd.DataFrame(X, columns=feat_names, index=gdf.index)
        gdf_out = pd.concat([gdf, feat_df], axis=1)

        out_path = os.path.join(out_dir, f"{fid}_ancillary_features.gpkg")
        gdf_out.to_file(out_path, driver="GPKG")
        print(f"    {len(gdf):,} pts | {X.shape[1]} features → {out_path}")

    print("\nAncillary feature extraction done.")


# ── Field GeoPackage map ───────────────────────────────────────────────────
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

# ── Paths ──────────────────────────────────────────────────────────────────
YIELD_DIR       = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
ENMAP_PATH      = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_kepek\2025_03_13.tif"
ENMAP_INDEX_DIR = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_indices"
DEM_PATH        = r"D:\STUDI\Thesis\mezohegyes\dem10m_reproject_s2.tif"
TWI_PATH        = r"D:\STUDI\Thesis\mezohegyes\twi_from_modeller_reproject_s2.tif"
IRRIG_GPKG      = r"D:\STUDI\Thesis\mezohegyes\irrigated_fields.gpkg"
OUT_DIR         = r"D:\STUDI\Thesis\mezohegyes\features\ancillary"


if __name__ == "__main__":
    run_extraction(
        yield_dir       = YIELD_DIR,
        gpkg_map        = FIELD_GPKG_CALIB,
        enmap_path      = ENMAP_PATH,
        enmap_index_dir = ENMAP_INDEX_DIR,
        dem_path        = DEM_PATH,
        twi_path        = TWI_PATH,
        irrig_gpkg      = IRRIG_GPKG,
        out_dir         = OUT_DIR,
    )
