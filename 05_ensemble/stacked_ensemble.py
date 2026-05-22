"""
05_ensemble/stacked_ensemble.py

Ridge meta-learner stacking across three configurations:

    STACK_Trees  : RF + XGB + LGB
    STACK_Hybrid : XGB + LGB + ANN
    STACK_Full   : RF + XGB + LGB + ANN + DANN

Within each LOFO fold:
  1. All base models are trained on 85% of the training data.
  2. Each base model predicts on the held-out 15% validation partition
     (out-of-fold predictions).
  3. A Ridge meta-learner (alpha=1.0) is fit on these OOF predictions.
  4. The meta-learner is applied to base model predictions on the test field.

The residual-to-absolute conversion (add back HT_mean) is applied to the
final ensemble output, matching the base model evaluation design.

Usage:
    python stacked_ensemble.py
    (edit paths at the bottom)
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, detect_yield_col, FIELD_TO_TABLA
from utils.raster_sampling import sample_single_band, sample_all_bands, get_xy
from utils.metrics import compute_metrics, print_metrics

# Reuse model + training code from modelling scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "04_modelling"))
from model_comparison import (
    RF_PARAMS, XGB_PARAMS, LGB_PARAMS, ANN_CFG,
    train_ann, predict_ann, DEVICE, ANN_SEEDS,
    build_X as build_X_330,
)
from dann_lofo import (
    train_dann, DANN_CFG, SEEDS as DANN_SEEDS,
)

import torch

warnings.filterwarnings("ignore")

HEADING_DATES = {
    "20250614", "20250621", "20250622",
    "20250714", "20250721", "20250724",
}
FNAME_PATTERN = re.compile(
    r"(S[12])_(\d{8})_Mezohegyes_Stacked_([A-Za-z0-9_]+)\.tif$"
)

# Stack configurations: name → list of base model names
STACK_CONFIGS = {
    "STACK_Trees":  ["RF", "XGB", "LGB"],
    "STACK_Hybrid": ["XGB", "LGB", "ANN"],
    "STACK_Full":   ["RF", "XGB", "LGB", "ANN", "DANN"],
}

RIDGE_ALPHA = 1.0
VAL_FRAC    = 0.15  # fraction of training data used for meta-learner OOF


def train_base_model(name, X_tr, y_tr, X_val, y_val, input_dim,
                     n_domains, d_tr, lambda_dann):
    """Train a single base model and return predictions on val + test sets."""

    if name == "RF":
        m = RandomForestRegressor(**RF_PARAMS)
        m.fit(X_tr, y_tr)
        return m.predict(X_val), m  # (val_preds, model)

    elif name == "XGB":
        params = {k: v for k, v in XGB_PARAMS.items()
                  if k != "early_stopping_rounds"}
        m = xgb.XGBRegressor(
            **params, early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"]
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return m.predict(X_val), m

    elif name == "LGB":
        params = {k: v for k, v in LGB_PARAMS.items() if k != "callbacks"}
        m = lgb.LGBMRegressor(**params, callbacks=LGB_PARAMS["callbacks"])
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        return m.predict(X_val), m

    elif name == "ANN":
        preds_val = []
        models = []
        for seed in ANN_SEEDS:
            ann = train_ann(X_tr, y_tr, X_val, y_val, input_dim=input_dim, seed=seed)
            preds_val.append(predict_ann(ann, X_val))
            models.append(ann)
        return np.mean(preds_val, axis=0), models

    elif name == "DANN":
        preds_val = []
        # For DANN on the val set we use a temporary dummy domain label
        d_val = np.zeros(len(y_val), dtype=np.int64)
        for seed in DANN_SEEDS:
            # Train on full tr, predict on val (approximate OOF for DANN)
            y_pred = train_dann(
                X_tr, y_tr, d_tr,
                X_val, y_val, X_val,
                input_dim, n_domains, lambda_dann, seed,
            )
            preds_val.append(y_pred)
        return np.mean(preds_val, axis=0), None


def predict_base(name, model, X_te, input_dim, n_domains,
                 d_tr, X_tr, y_tr, X_val, y_val, lambda_dann):
    """Generate test field predictions from a trained base model."""

    if name in ("RF", "XGB", "LGB"):
        return model.predict(X_te)

    elif name == "ANN":
        # model is a list of ANN instances
        return np.mean([predict_ann(m, X_te) for m in model], axis=0)

    elif name == "DANN":
        preds = []
        for seed in DANN_SEEDS:
            y_res_pred = train_dann(
                X_tr, y_tr, d_tr,
                X_val, y_val, X_te,
                input_dim, n_domains, lambda_dann, seed,
            )
            preds.append(y_res_pred)
        return np.mean(preds, axis=0)


def run_stacking(fields, group_label, lambda_dann,
                 yield_dir, gpkg_map, harvest_csv,
                 s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    harvest_means = load_harvest_means(harvest_csv)

    print(f"\n{'█'*55}")
    print(f"  Stacked ensemble — {group_label}")
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
        X, _    = build_X_330(gdf, s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg)
        field_data[fid] = {"gdf": gdf, "y": y, "ht_mean": ht_mean, "X": X}
        print(f"  {fid}: {len(y):,} pts | {X.shape[1]} feat | HT={ht_mean:.3f}")

    fid_list   = list(field_data.keys())
    fid_to_idx = {f: i for i, f in enumerate(fid_list)}
    n_domains  = len(fid_list)
    input_dim  = next(iter(field_data.values()))["X"].shape[1]

    all_results = []

    for test_fid in fid_list:
        train_fids = [f for f in fid_list if f != test_fid]

        X_tr_raw = np.concatenate([field_data[f]["X"] for f in train_fids])
        y_res    = np.concatenate([field_data[f]["y"] - field_data[f]["ht_mean"]
                                   for f in train_fids])
        d_tr_all = np.concatenate([
            np.full(len(field_data[f]["y"]), fid_to_idx[f])
            for f in train_fids
        ]).astype(np.int64)

        X_te  = field_data[test_fid]["X"]
        y_te  = field_data[test_fid]["y"]
        ht_te = field_data[test_fid]["ht_mean"]

        # Split training into 85% for base models, 15% for meta-learner OOF
        rng   = np.random.default_rng(42)
        n_val = max(1, int(VAL_FRAC * len(y_res)))
        idx   = rng.permutation(len(y_res))
        val_i, tr_i = idx[:n_val], idx[n_val:]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        X_tr_sc  = sc.fit_transform(imp.fit_transform(X_tr_raw[tr_i]))
        X_val_sc = sc.transform(imp.transform(X_tr_raw[val_i]))
        X_te_sc  = sc.transform(imp.transform(X_te))

        y_tr_f  = y_res[tr_i]
        y_val_f = y_res[val_i]
        d_tr_f  = d_tr_all[tr_i]

        print(f"\n  FOLD test={test_fid}")

        # Collect OOF val predictions and test predictions for all possible base models
        val_preds_all  = {}
        test_preds_all = {}

        all_base_names = {"RF", "XGB", "LGB", "ANN", "DANN"}
        needed = set()
        for cfg_models in STACK_CONFIGS.values():
            needed |= set(cfg_models)

        for model_name in needed:
            print(f"    Training {model_name} ...")
            val_p, model_obj = train_base_model(
                model_name, X_tr_sc, y_tr_f, X_val_sc, y_val_f,
                input_dim, n_domains, d_tr_f, lambda_dann,
            )
            val_preds_all[model_name] = val_p

            te_p = predict_base(
                model_name, model_obj, X_te_sc, input_dim, n_domains,
                d_tr_f, X_tr_sc, y_tr_f, X_val_sc, y_val_f, lambda_dann,
            )
            test_preds_all[model_name] = te_p

        # Fit and evaluate each stack configuration
        for stack_name, base_names in STACK_CONFIGS.items():
            X_meta_val  = np.column_stack([val_preds_all[n]  for n in base_names])
            X_meta_test = np.column_stack([test_preds_all[n] for n in base_names])

            meta = Ridge(alpha=RIDGE_ALPHA)
            meta.fit(X_meta_val, y_val_f)

            # Report meta-learner weights
            weights = {n: float(w) for n, w in zip(base_names, meta.coef_)}
            print(f"    {stack_name} meta weights: {weights}")

            y_pred = meta.predict(X_meta_test) + ht_te
            m = compute_metrics(y_te, y_pred)
            print_metrics(test_fid, m, prefix=f"    [{stack_name}] ")

            all_results.append({
                "stack":   stack_name,
                "group":   group_label,
                "field":   test_fid,
                **m,
                **{f"w_{n}": weights.get(n, np.nan) for n in base_names},
            })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, f"ensemble_results_{group_label}.csv"), index=False)

    print(f"\n{'='*55}")
    print(f"STACKING SUMMARY — {group_label}")
    print(f"{'='*55}")
    print(df.groupby("stack")[["R2", "RMSE"]].mean().round(3).to_string())


# ── Configuration ──────────────────────────────────────────────────────────
YIELD_DIR   = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
HARVEST_CSV = r"D:\STUDI\Thesis\mezohegyes\obuza_napi_aratas_2025_fix.csv"
S1S2_DIR    = r"D:\STUDI\Thesis\mezohegyes\VIs\s1+s2"
ENMAP_PATH  = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_kepek\2025_03_13.tif"
DEM_PATH    = r"D:\STUDI\Thesis\mezohegyes\dem10m_reproject_s2.tif"
TWI_PATH    = r"D:\STUDI\Thesis\mezohegyes\twi_from_modeller_reproject_s2.tif"
IRRIG_GPKG  = r"D:\STUDI\Thesis\mezohegyes\irrigated_fields.gpkg"
OUT_DIR     = r"D:\STUDI\Thesis\mezohegyes\results\ensemble"

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
    yield_dir   = YIELD_DIR,
    gpkg_map    = FIELD_GPKG_CALIB,
    harvest_csv = HARVEST_CSV,
    s1s2_dir    = S1S2_DIR,
    enmap_path  = ENMAP_PATH,
    dem_path    = DEM_PATH,
    twi_path    = TWI_PATH,
    irrig_gpkg  = IRRIG_GPKG,
    out_dir     = OUT_DIR,
)


if __name__ == "__main__":
    for group_label, cfg in GROUPS.items():
        run_stacking(
            fields       = cfg["fields"],
            group_label  = group_label,
            lambda_dann  = cfg["lambda"],
            **COMMON,
        )
