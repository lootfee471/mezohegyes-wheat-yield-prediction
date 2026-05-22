"""
04_modelling/extended_experiment.py

Extended experiment: adds SAR interferometric coherence (6 date pairs) and
a pre-season bare soil composite (10 features, September 22, 2024) to the
330-feature baseline to produce a 345-feature input.

All five models (RF, XGB, LGB, ANN, DANN) are evaluated under the same
LOFO design as the main model comparison.

Baresoil composite features:
    Indices: BSI, NSMI, NDTI, CI, NDVI_soil, SWIR_ratio
    Raw bands: B4, B8, B11, B12

SAR coherence: 6 consecutive pairs from Mar 28 – May 27, 2025

Usage:
    python extended_experiment.py
    (edit paths and COHERENCE_FILES / BARESOIL_PATH at the bottom)
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, detect_yield_col, FIELD_TO_TABLA
from utils.raster_sampling import sample_single_band, sample_all_bands, get_xy
from utils.metrics import compute_metrics, print_metrics

# Re-use model definitions from model_comparison.py
from model_comparison import (
    RF_PARAMS, XGB_PARAMS, LGB_PARAMS, ANN_CFG, MLP,
    train_ann, predict_ann, DEVICE, ANN_SEEDS,
)
from dann_lofo import (
    DANN_CFG, DANNModel, GradRev, train_dann, SEEDS as DANN_SEEDS,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

HEADING_DATES = {
    "20250614", "20250621", "20250622",
    "20250714", "20250721", "20250724",
}
FNAME_PATTERN = re.compile(
    r"(S[12])_(\d{8})_Mezohegyes_Stacked_([A-Za-z0-9_]+)\.tif$"
)

BARESOIL_INDEX_NAMES = ["BSI", "NSMI", "NDTI", "CI", "NDVI_soil", "SWIR_ratio",
                         "B4", "B8", "B11", "B12"]


def build_X_extended(gdf, s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg,
                     coherence_files, baresoil_dir):
    """
    330-feature baseline + 6 coherence layers + 10 baresoil bands = 345 features.
    """
    xs, ys = get_xy(gdf)
    blocks, names = [], []

    # S1/S2 heading-to-harvest (same as 330-feature input)
    for fname in sorted(os.listdir(s1s2_dir)):
        m = FNAME_PATTERN.match(fname)
        if m is None or m.group(2) not in HEADING_DATES:
            continue
        v = sample_single_band(os.path.join(s1s2_dir, fname), xs, ys)
        blocks.append(v[:, None])
        names.append(f"{m.group(1)}_{m.group(2)}_{m.group(3)}")

    # EnMAP
    if os.path.exists(enmap_path):
        ev = sample_all_bands(enmap_path, xs, ys)
        if ev is not None:
            blocks.append(ev)
            names.extend([f"ENMAP_b{b+1}" for b in range(ev.shape[1])])

    # Terrain + irrigation
    for path, name in [(dem_path, "DEM"), (twi_path, "TWI")]:
        v = sample_single_band(path, xs, ys)
        blocks.append(v[:, None])
        names.append(name)

    irr = np.zeros(len(gdf), dtype=np.float32)
    if os.path.exists(irrig_gpkg):
        irrig = gpd.read_file(irrig_gpkg)
        if gdf.crs != irrig.crs:
            irrig = irrig.to_crs(gdf.crs)
        pts = gdf[["geometry"]].copy()
        pts["__i"] = np.arange(len(pts))
        joined = gpd.sjoin(pts, irrig[["geometry"]], how="left", predicate="within")
        irr[joined.dropna(subset=["index_right"])["__i"].values.astype(int)] = 1.0
    blocks.append(irr[:, None])
    names.append("Irrigation")

    # SAR coherence (6 date pairs)
    for pair_label, fpath in coherence_files.items():
        v = sample_single_band(fpath, xs, ys)
        blocks.append(v[:, None])
        names.append(f"COH_{pair_label}")

    # Pre-season baresoil composite
    for idx_name in BARESOIL_INDEX_NAMES:
        fpath = os.path.join(baresoil_dir, f"{idx_name}.tif")
        v = sample_single_band(fpath, xs, ys)
        blocks.append(v[:, None])
        names.append(f"SOIL_{idx_name}")

    X = np.nan_to_num(
        np.concatenate(blocks, axis=1).astype(np.float32), nan=0.0
    )
    return X, names


def run_extended(fields, group_label, lambda_dann,
                 yield_dir, gpkg_map, harvest_csv,
                 s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg,
                 coherence_files, baresoil_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    harvest_means = load_harvest_means(harvest_csv)

    print(f"\n{'█'*55}")
    print(f"  Extended experiment — {group_label} (345 features)")
    print(f"{'█'*55}")

    field_data = {}
    for fid in fields:
        fpath = os.path.join(yield_dir, gpkg_map.get(fid, ""))
        if not os.path.exists(fpath):
            continue
        gdf = gpd.read_file(fpath)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:23700")
        yc  = detect_yield_col(gdf)
        y   = pd.to_numeric(gdf[yc], errors="coerce").values.astype(float)
        if np.nanmean(y) > 50:
            y /= 1000.0
        valid = np.isfinite(y) & (y > 0.5) & (y < 20)
        gdf, y = gdf[valid].reset_index(drop=True), y[valid]
        tabla   = FIELD_TO_TABLA.get(fid)
        ht_mean = harvest_means.get(fid) or harvest_means.get(tabla, np.mean(y))
        X, _    = build_X_extended(gdf, s1s2_dir, enmap_path, dem_path, twi_path,
                                   irrig_gpkg, coherence_files, baresoil_dir)
        field_data[fid] = {"gdf": gdf, "y": y, "ht_mean": ht_mean, "X": X}
        print(f"  {fid}: {len(y):,} pts | {X.shape[1]} features | HT={ht_mean:.3f}")

    fid_list   = list(field_data.keys())
    fid_to_idx = {f: i for i, f in enumerate(fid_list)}
    input_dim  = next(iter(field_data.values()))["X"].shape[1]
    n_domains  = len(fid_list)

    all_results = []

    for test_fid in fid_list:
        train_fids = [f for f in fid_list if f != test_fid]

        X_tr_raw = np.concatenate([field_data[f]["X"] for f in train_fids])
        y_res    = np.concatenate([field_data[f]["y"] - field_data[f]["ht_mean"]
                                   for f in train_fids])
        d_tr     = np.concatenate([
            np.full(len(field_data[f]["y"]), fid_to_idx[f])
            for f in train_fids
        ]).astype(np.int64)

        X_te  = field_data[test_fid]["X"]
        y_te  = field_data[test_fid]["y"]
        ht_te = field_data[test_fid]["ht_mean"]

        rng   = np.random.default_rng(42)
        n_val = max(1, int(0.15 * len(y_res)))
        idx   = rng.permutation(len(y_res))
        val_i, tr_i = idx[:n_val], idx[n_val:]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        X_tr_sc  = sc.fit_transform(imp.fit_transform(X_tr_raw[tr_i]))
        X_val_sc = sc.transform(imp.transform(X_tr_raw[val_i]))
        X_te_sc  = sc.transform(imp.transform(X_te))

        y_tr_f  = y_res[tr_i]
        y_val_f = y_res[val_i]

        print(f"\n  FOLD test={test_fid}")

        for model_name in ["RF", "XGB", "LGB", "ANN", "DANN"]:

            if model_name == "RF":
                m_obj = RandomForestRegressor(**RF_PARAMS)
                m_obj.fit(X_tr_sc, y_tr_f)
                y_pred = m_obj.predict(X_te_sc) + ht_te

            elif model_name == "XGB":
                params = {k: v for k, v in XGB_PARAMS.items()
                          if k != "early_stopping_rounds"}
                m_obj = xgb.XGBRegressor(
                    **params,
                    early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"]
                )
                m_obj.fit(X_tr_sc, y_tr_f,
                          eval_set=[(X_val_sc, y_val_f)], verbose=False)
                y_pred = m_obj.predict(X_te_sc) + ht_te

            elif model_name == "LGB":
                params = {k: v for k, v in LGB_PARAMS.items() if k != "callbacks"}
                m_obj = lgb.LGBMRegressor(**params, callbacks=LGB_PARAMS["callbacks"])
                m_obj.fit(X_tr_sc, y_tr_f, eval_set=[(X_val_sc, y_val_f)])
                y_pred = m_obj.predict(X_te_sc) + ht_te

            elif model_name == "ANN":
                preds = []
                for seed in ANN_SEEDS:
                    ann = train_ann(X_tr_sc, y_tr_f, X_val_sc, y_val_f,
                                   input_dim=input_dim, seed=seed)
                    preds.append(predict_ann(ann, X_te_sc))
                y_pred = np.mean(preds, axis=0) + ht_te

            elif model_name == "DANN":
                preds = []
                for seed in DANN_SEEDS:
                    y_res_pred = train_dann(
                        X_tr_sc, y_tr_f, d_tr[tr_i],
                        X_val_sc, y_val_f, X_te_sc,
                        input_dim, n_domains, lambda_dann, seed,
                    )
                    preds.append(y_res_pred)
                y_pred = np.mean(preds, axis=0) + ht_te

            m = compute_metrics(y_te, y_pred)
            print_metrics(test_fid, m, prefix=f"    [{model_name}] ")

            all_results.append({
                "model": model_name,
                "group": group_label,
                "field": test_fid,
                **m,
            })

    df = pd.DataFrame(all_results)
    df.to_csv(
        os.path.join(out_dir, f"extended_results_{group_label}.csv"), index=False
    )

    print(f"\n{'='*55}")
    print(f"SUMMARY — Extended experiment ({group_label})")
    print(f"{'='*55}")
    print(df.groupby("model")[["R2", "RMSE", "MAE", "Bias"]].mean().round(3).to_string())


# ── Configuration ──────────────────────────────────────────────────────────
YIELD_DIR    = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
HARVEST_CSV  = r"D:\STUDI\Thesis\mezohegyes\obuza_napi_aratas_2025_fix.csv"
S1S2_DIR     = r"D:\STUDI\Thesis\mezohegyes\VIs\s1+s2"
ENMAP_PATH   = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_kepek\2025_03_13.tif"
DEM_PATH     = r"D:\STUDI\Thesis\mezohegyes\dem10m_reproject_s2.tif"
TWI_PATH     = r"D:\STUDI\Thesis\mezohegyes\twi_from_modeller_reproject_s2.tif"
IRRIG_GPKG   = r"D:\STUDI\Thesis\mezohegyes\irrigated_fields.gpkg"
BARESOIL_DIR = r"D:\STUDI\Thesis\mezohegyes\VIs\baresoil_20240922"
OUT_DIR      = r"D:\STUDI\Thesis\mezohegyes\results\extended"

# SAR coherence files (6 consecutive pairs)
COHERENCE_FILES = {
    "20250328-20250409": r"D:\STUDI\Thesis\mezohegyes\VIs\coherence\coh_20250328_20250409.tif",
    "20250409-20250421": r"D:\STUDI\Thesis\mezohegyes\VIs\coherence\coh_20250409_20250421.tif",
    "20250421-20250503": r"D:\STUDI\Thesis\mezohegyes\VIs\coherence\coh_20250421_20250503.tif",
    "20250503-20250515": r"D:\STUDI\Thesis\mezohegyes\VIs\coherence\coh_20250503_20250515.tif",
    "20250515-20250527": r"D:\STUDI\Thesis\mezohegyes\VIs\coherence\coh_20250515_20250527.tif",
    "20250527-20250608": r"D:\STUDI\Thesis\mezohegyes\VIs\coherence\coh_20250527_20250608.tif",
}

FIELD_GPKG_CALIB = {
    "7":    "7_yield_10px_calib.gpkg",
    "9_ce": "9_ce_yield_10px_calib.gpkg",
    "9_pr": "9_pr_yield_10px_calib.gpkg",
    "9_sy": "9_sy_yield_10px_calib.gpkg",
    "12":   "12_yield_10px_calib.gpkg",
    "25":   "25_yield_10px_calib.gpkg",
    "44":   "44_yield_10px_calib.gpkg",
    "59":   "59_yield_10px_calib.gpkg",
}

GROUPS = {
    "high": {"fields": ["9_ce", "9_pr", "9_sy", "12"],  "lambda": 0.5},
    "low":  {"fields": ["7", "25", "44", "59"],          "lambda": 0.1},
}

COMMON = dict(
    yield_dir       = YIELD_DIR,
    gpkg_map        = FIELD_GPKG_CALIB,
    harvest_csv     = HARVEST_CSV,
    s1s2_dir        = S1S2_DIR,
    enmap_path      = ENMAP_PATH,
    dem_path        = DEM_PATH,
    twi_path        = TWI_PATH,
    irrig_gpkg      = IRRIG_GPKG,
    coherence_files = COHERENCE_FILES,
    baresoil_dir    = BARESOIL_DIR,
    out_dir         = OUT_DIR,
)


if __name__ == "__main__":
    for group_label, cfg in GROUPS.items():
        run_extended(
            fields       = cfg["fields"],
            group_label  = group_label,
            lambda_dann  = cfg["lambda"],
            **COMMON,
        )
