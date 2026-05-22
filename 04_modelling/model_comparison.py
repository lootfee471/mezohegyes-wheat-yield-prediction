"""
04_modelling/model_comparison.py

LOFO evaluation of RF, XGBoost, LightGBM, and ANN on the 330-feature input.
(DANN is in a separate script: dann_lofo.py)

The 330-feature input:
    - S1/S2 spectral indices from 6 heading-to-harvest dates (98 features)
    - EnMAP raw bands from 13 March 2025 scene (219 bands)
    - DEM (1), TWI (1), irrigation flag (1)

Stochastic models (ANN) are averaged across 5 seeds (42–46).
Tree-based models are deterministic and run once.

Output: per-field and mean R²/RMSE/MAE/Bias for each model, saved as CSV.

Usage:
    python model_comparison.py
    (edit paths at the bottom)
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, detect_yield_col, FIELD_TO_TABLA
from utils.raster_sampling import sample_single_band, sample_all_bands, get_xy
from utils.metrics import compute_metrics, print_metrics

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANN_SEEDS = [42, 43, 44, 45, 46]

# ── Heading-to-harvest dates for 330-feature input ─────────────────────────
HEADING_DATES = {
    "20250614", "20250621", "20250622",
    "20250714", "20250721", "20250724",
}

FNAME_PATTERN = re.compile(
    r"(S[12])_(\d{8})_Mezohegyes_Stacked_([A-Za-z0-9_]+)\.tif$"
)


# ── Model hyperparameters ──────────────────────────────────────────────────

RF_PARAMS = dict(
    n_estimators=300, max_features="sqrt",
    min_samples_leaf=5, max_samples=0.8,
    n_jobs=-1, random_state=42,
)

XGB_PARAMS = dict(
    n_estimators=400, max_depth=6,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1,
    reg_lambda=1.0, min_child_weight=5,
    tree_method="hist", random_state=42,
    early_stopping_rounds=20,
)

LGB_PARAMS = dict(
    n_estimators=400, max_depth=6,
    learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1,
    reg_lambda=1.0, random_state=42,
    callbacks=[lgb.early_stopping(20, verbose=False)],
)

ANN_CFG = dict(
    hidden_dims=[512, 256, 128, 64],
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=200,
    patience=15,
    batch_size=1024,
)


# ── ANN definition ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_ann(X_tr, y_tr, X_val, y_val, input_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(input_dim, ANN_CFG["hidden_dims"], ANN_CFG["dropout"]).to(DEVICE)
    opt   = optim.Adam(model.parameters(),
                       lr=ANN_CFG["lr"], weight_decay=ANN_CFG["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_tr,  dtype=torch.float32),
        torch.tensor(y_tr,  dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=ANN_CFG["batch_size"], shuffle=True)

    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, ANN_CFG["epochs"] + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= ANN_CFG["patience"]:
            break

    model.load_state_dict(best_state)
    return model


def predict_ann(model, X_te):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        return model(X_t).cpu().numpy()


# ── Feature assembly ───────────────────────────────────────────────────────

def build_X(gdf, s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg):
    xs, ys = get_xy(gdf)
    blocks, names = [], []

    # S1/S2 heading-to-harvest indices
    for fname in sorted(os.listdir(s1s2_dir)):
        m = FNAME_PATTERN.match(fname)
        if m is None:
            continue
        if m.group(2) not in HEADING_DATES:
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

    # Terrain
    for path, name in [(dem_path, "DEM"), (twi_path, "TWI")]:
        v = sample_single_band(path, xs, ys)
        blocks.append(v[:, None])
        names.append(name)

    # Irrigation flag
    irr = np.zeros(len(gdf), dtype=np.float32)
    if os.path.exists(irrig_gpkg):
        import geopandas as gpd_inner
        irrig = gpd_inner.read_file(irrig_gpkg)
        if gdf.crs != irrig.crs:
            irrig = irrig.to_crs(gdf.crs)
        pts = gdf[["geometry"]].copy()
        pts["__i"] = np.arange(len(pts))
        joined = gpd_inner.sjoin(pts, irrig[["geometry"]], how="left", predicate="within")
        irr[joined.dropna(subset=["index_right"])["__i"].values.astype(int)] = 1.0
    blocks.append(irr[:, None])
    names.append("Irrigation")

    X = np.nan_to_num(
        np.concatenate(blocks, axis=1).astype(np.float32), nan=0.0
    )
    return X, names


# ── LOFO runner ────────────────────────────────────────────────────────────

def run_lofo(fields, yield_dir, gpkg_map, harvest_csv,
             s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    harvest_means = load_harvest_means(harvest_csv)

    # Load all fields
    field_data = {}
    print("\nLoading fields ...")
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
        X, _    = build_X(gdf, s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg)
        field_data[fid] = {"gdf": gdf, "y": y, "ht_mean": ht_mean, "X": X}
        print(f"  {fid}: {len(y):,} pts | {X.shape[1]} feat | HT={ht_mean:.3f}")

    fid_list = list(field_data.keys())
    all_results = []

    for test_fid in fid_list:
        train_fids = [f for f in fid_list if f != test_fid]
        X_tr  = np.concatenate([field_data[f]["X"] for f in train_fids])
        y_res = np.concatenate([field_data[f]["y"] - field_data[f]["ht_mean"]
                                for f in train_fids])
        X_te  = field_data[test_fid]["X"]
        y_te  = field_data[test_fid]["y"]
        ht_te = field_data[test_fid]["ht_mean"]

        # Val split for tree early stopping and ANN
        rng   = np.random.default_rng(42)
        n_val = max(1, int(0.15 * len(y_res)))
        idx   = rng.permutation(len(y_res))
        val_i, tr_i = idx[:n_val], idx[n_val:]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        X_tr_sc  = sc.fit_transform(imp.fit_transform(X_tr[tr_i]))
        X_val_sc = sc.transform(imp.transform(X_tr[val_i]))
        X_te_sc  = sc.transform(imp.transform(X_te))

        y_tr_f  = y_res[tr_i]
        y_val_f = y_res[val_i]

        print(f"\n  FOLD test={test_fid}  train={train_fids}")

        for model_name in ["RF", "XGB", "LGB", "ANN"]:

            if model_name == "RF":
                rf = RandomForestRegressor(**RF_PARAMS)
                rf.fit(X_tr_sc, y_tr_f)
                y_pred = rf.predict(X_te_sc) + ht_te

            elif model_name == "XGB":
                params = {k: v for k, v in XGB_PARAMS.items()
                          if k != "early_stopping_rounds"}
                model = xgb.XGBRegressor(
                    **params,
                    early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"]
                )
                model.fit(
                    X_tr_sc, y_tr_f,
                    eval_set=[(X_val_sc, y_val_f)],
                    verbose=False,
                )
                y_pred = model.predict(X_te_sc) + ht_te

            elif model_name == "LGB":
                params = {k: v for k, v in LGB_PARAMS.items()
                          if k != "callbacks"}
                model = lgb.LGBMRegressor(**params, callbacks=LGB_PARAMS["callbacks"])
                model.fit(
                    X_tr_sc, y_tr_f,
                    eval_set=[(X_val_sc, y_val_f)],
                )
                y_pred = model.predict(X_te_sc) + ht_te

            elif model_name == "ANN":
                # Average over multiple seeds
                preds = []
                for seed in ANN_SEEDS:
                    ann = train_ann(
                        X_tr_sc, y_tr_f, X_val_sc, y_val_f,
                        input_dim=X_tr_sc.shape[1], seed=seed,
                    )
                    preds.append(predict_ann(ann, X_te_sc))
                y_pred = np.mean(preds, axis=0) + ht_te

            m = compute_metrics(y_te, y_pred)
            print_metrics(test_fid, m, prefix=f"    [{model_name}] ")

            all_results.append({
                "model": model_name,
                "field": test_fid,
                **m,
            })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, "model_comparison_results.csv"), index=False)

    print(f"\n{'='*55}")
    print("SUMMARY — Mean R² per model")
    print(f"{'='*55}")
    print(df.groupby("model")[["R2", "RMSE"]].mean().round(3).to_string())
    print(f"\nResults saved → {out_dir}")


# ── Configuration ──────────────────────────────────────────────────────────
YIELD_DIR   = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
HARVEST_CSV = r"D:\STUDI\Thesis\mezohegyes\obuza_napi_aratas_2025_fix.csv"
S1S2_DIR    = r"D:\STUDI\Thesis\mezohegyes\VIs\s1+s2"
ENMAP_PATH  = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_kepek\2025_03_13.tif"
DEM_PATH    = r"D:\STUDI\Thesis\mezohegyes\dem10m_reproject_s2.tif"
TWI_PATH    = r"D:\STUDI\Thesis\mezohegyes\twi_from_modeller_reproject_s2.tif"
IRRIG_GPKG  = r"D:\STUDI\Thesis\mezohegyes\irrigated_fields.gpkg"
OUT_DIR     = r"D:\STUDI\Thesis\mezohegyes\results\model_comparison"

# High and low yield groups (model comparison uses 8 fields)
HIGH_FIELDS = ["9_ce", "9_pr", "9_sy", "12"]
LOW_FIELDS  = ["7", "25", "44", "59"]

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


if __name__ == "__main__":
    for group_name, fields in [("HIGH", HIGH_FIELDS), ("LOW", LOW_FIELDS)]:
        print(f"\n{'█'*55}")
        print(f"  Model comparison — {group_name} group")
        print(f"{'█'*55}")
        run_lofo(
            fields      = fields,
            yield_dir   = YIELD_DIR,
            gpkg_map    = FIELD_GPKG_CALIB,
            harvest_csv = HARVEST_CSV,
            s1s2_dir    = S1S2_DIR,
            enmap_path  = ENMAP_PATH,
            dem_path    = DEM_PATH,
            twi_path    = TWI_PATH,
            irrig_gpkg  = IRRIG_GPKG,
            out_dir     = os.path.join(OUT_DIR, group_name.lower()),
        )
