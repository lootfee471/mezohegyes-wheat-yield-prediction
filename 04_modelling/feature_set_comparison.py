"""
04_modelling/feature_set_comparison.py

RF-LOFO evaluation across four feature set combinations:
  - S1 only        (48 features: 7 dates × 6 backscatter indices + 6 coherence)
  - S2 only        (234 features: 13 dates × 18 indices)
  - S1 + S2        (282 features)
  - S1 + S2 + EnMAP (509 features)

No terrain or irrigation features are included here so that the sensor
contribution can be isolated cleanly.

Feature importance by date and by index is computed and saved for each fold.

Output: CSV result tables + per-fold feature importance in OUT_DIR.

Usage:
    python feature_set_comparison.py
    (edit paths and group definitions at the bottom)
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, detect_yield_col, FIELD_TO_TABLA
from utils.raster_sampling import sample_single_band, sample_all_bands, get_xy
from utils.metrics import compute_metrics, print_metrics

warnings.filterwarnings("ignore")


# ── Feature set definitions ───────────────────────────────────────────────
# S1 coherence date pairs (consecutive acquisitions)
S1_COHERENCE_PAIRS = [
    ("20250328", "20250409"),
    ("20250409", "20250421"),
    ("20250421", "20250503"),
    ("20250503", "20250515"),
    ("20250515", "20250527"),
    ("20250527", "20250608"),
]

FNAME_PATTERN = re.compile(
    r"(S[12])_(\d{8})_Mezohegyes_Stacked_([A-Za-z0-9_]+)\.tif$"
)


# ── RF hyperparameters ────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators  = 300,
    max_features  = "sqrt",
    min_samples_leaf = 5,
    max_samples   = 0.8,
    n_jobs        = -1,
    random_state  = 42,
)


def discover_rasters(s1s2_dir, sensor_filter=None):
    """Return dict {key: path} for all matching rasters, optionally filtered by sensor."""
    out = {}
    for fname in sorted(os.listdir(s1s2_dir)):
        if not fname.endswith(".tif"):
            continue
        m = FNAME_PATTERN.match(fname)
        if m is None:
            continue
        sensor = m.group(1)
        if sensor_filter and sensor not in sensor_filter:
            continue
        key = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        out[key] = os.path.join(s1s2_dir, fname)
    return out


def build_feature_matrix(gdf, s1_files, s2_files, enmap_path, include_coherence=True,
                          include_enmap=False):
    """
    Assemble feature matrix for one field from the specified sensor sources.
    Returns (X, feature_names).
    """
    xs, ys = get_xy(gdf)
    blocks, names = [], []

    for key, fpath in {**s1_files, **s2_files}.items():
        v = sample_single_band(fpath, xs, ys)
        blocks.append(v[:, None])
        names.append(key)

    if include_enmap and os.path.exists(enmap_path):
        ev = sample_all_bands(enmap_path, xs, ys)
        if ev is not None:
            blocks.append(ev)
            names.extend([f"ENMAP_b{b+1}" for b in range(ev.shape[1])])

    if len(blocks) == 0:
        return np.empty((len(gdf), 0)), []

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, names


def load_all_fields(fields, yield_dir, gpkg_map, harvest_csv,
                    s1_files, s2_files, enmap_path, feature_set):

    harvest_means = load_harvest_means(harvest_csv)

    include_s1     = "S1" in feature_set
    include_s2     = "S2" in feature_set
    include_enmap  = "EnMAP" in feature_set

    s1 = s1_files if include_s1 else {}
    s2 = s2_files if include_s2 else {}

    data = {}
    for fid in fields:
        gpkg_name = gpkg_map.get(fid)
        if gpkg_name is None:
            continue
        fpath = os.path.join(yield_dir, gpkg_name)
        if not os.path.exists(fpath):
            print(f"  [skip] {fid}")
            continue

        gdf = gpd.read_file(fpath)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:23700")

        yc = detect_yield_col(gdf)
        y  = pd.to_numeric(gdf[yc], errors="coerce").values.astype(float)
        if np.nanmean(y) > 50:
            y /= 1000.0

        valid = np.isfinite(y) & (y > 0.5) & (y < 20)
        gdf, y = gdf[valid].reset_index(drop=True), y[valid]

        tabla   = FIELD_TO_TABLA.get(fid)
        ht_mean = harvest_means.get(fid) or harvest_means.get(tabla, np.mean(y))

        X, feat_names = build_feature_matrix(
            gdf, s1, s2, enmap_path,
            include_enmap=include_enmap,
        )

        data[fid] = {"gdf": gdf, "y": y, "ht_mean": ht_mean, "X": X,
                     "feat_names": feat_names}
        print(f"  {fid}: {len(y):,} pts | {X.shape[1]} features | HT={ht_mean:.3f}")

    return data


def run_rf_lofo(fields, feature_set_name, data, out_dir):
    """Run RF LOFO for one feature set across all fields."""
    fid_list = [f for f in fields if f in data]
    results  = []
    all_importances = []

    for test_fid in fid_list:
        train_fids = [f for f in fid_list if f != test_fid]

        X_tr  = np.concatenate([data[f]["X"] for f in train_fids])
        y_res = np.concatenate([data[f]["y"] - data[f]["ht_mean"] for f in train_fids])

        X_te  = data[test_fid]["X"]
        y_te  = data[test_fid]["y"]
        ht_te = data[test_fid]["ht_mean"]

        # Impute and scale
        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        X_tr_sc = sc.fit_transform(imp.fit_transform(X_tr))
        X_te_sc = sc.transform(imp.transform(X_te))

        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr_sc, y_res)

        y_pred = rf.predict(X_te_sc) + ht_te
        m = compute_metrics(y_te, y_pred)
        print_metrics(test_fid, m, prefix=f"  [{feature_set_name}] ")

        results.append({"field": test_fid, "feature_set": feature_set_name, **m})

        # Feature importances
        feat_names = data[test_fid]["feat_names"]
        imps = rf.feature_importances_
        for name, imp_val in zip(feat_names, imps):
            all_importances.append({
                "test_fold":   test_fid,
                "feature_set": feature_set_name,
                "feature":     name,
                "importance":  imp_val,
            })

    # Save importances
    os.makedirs(out_dir, exist_ok=True)
    df_imp = pd.DataFrame(all_importances)
    df_imp.to_csv(
        os.path.join(out_dir, f"importance_{feature_set_name}.csv"),
        index=False,
    )

    return results


def main(yield_dir, gpkg_map, harvest_csv, s1s2_dir, enmap_path, out_dir,
         high_fields, low_fields):

    os.makedirs(out_dir, exist_ok=True)

    all_fields = high_fields + low_fields
    s1_rasters = discover_rasters(s1s2_dir, sensor_filter={"S1"})
    s2_rasters = discover_rasters(s1s2_dir, sensor_filter={"S2"})

    feature_sets = {
        "S1":            ("S1",),
        "S2":            ("S2",),
        "S1+S2":         ("S1", "S2"),
        "S1+S2+EnMAP":   ("S1", "S2", "EnMAP"),
    }

    all_results = []

    for fs_name, fs_tags in feature_sets.items():
        print(f"\n{'='*55}")
        print(f"  Feature set: {fs_name}")
        print(f"{'='*55}")

        feature_set_str = "+".join(fs_tags)

        # Load data for this feature set
        for group_name, fields in [("HIGH", high_fields), ("LOW", low_fields)]:
            print(f"\n  --- {group_name} group ---")
            data = load_all_fields(
                fields, yield_dir, gpkg_map, harvest_csv,
                s1_rasters, s2_rasters, enmap_path, feature_set_str
            )
            results = run_rf_lofo(fields, fs_name, data, out_dir)
            for r in results:
                r["group"] = group_name
            all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "feature_set_comparison_results.csv"), index=False)

    # Print summary table
    print(f"\n{'='*55}")
    print("SUMMARY — Mean R² per feature set")
    print(f"{'='*55}")
    pivot = df.groupby(["feature_set", "group"])["R2"].mean().unstack()
    pivot["Overall"] = df.groupby("feature_set")["R2"].mean()
    print(pivot.round(3).to_string())

    print(f"\nResults saved → {out_dir}")


# ── Configuration ──────────────────────────────────────────────────────────
YIELD_DIR   = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
HARVEST_CSV = r"D:\STUDI\Thesis\mezohegyes\obuza_napi_aratas_2025_fix.csv"
S1S2_DIR    = r"D:\STUDI\Thesis\mezohegyes\VIs\s1+s2"
ENMAP_PATH  = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_kepek\2025_03_13.tif"
OUT_DIR     = r"D:\STUDI\Thesis\mezohegyes\results\feature_comparison"

# Feature comparison uses 9 fields (4 high + 4 low + field 63 for S1-only context)
HIGH_FIELDS = ["9_ce", "9_pr", "9_sy", "12"]
LOW_FIELDS  = ["7", "25", "44", "59"]

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
}


if __name__ == "__main__":
    main(
        yield_dir   = YIELD_DIR,
        gpkg_map    = FIELD_GPKG_CALIB,
        harvest_csv = HARVEST_CSV,
        s1s2_dir    = S1S2_DIR,
        enmap_path  = ENMAP_PATH,
        out_dir     = OUT_DIR,
        high_fields = HIGH_FIELDS,
        low_fields  = LOW_FIELDS,
    )
