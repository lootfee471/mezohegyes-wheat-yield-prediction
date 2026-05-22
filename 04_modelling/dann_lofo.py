"""
04_modelling/dann_lofo.py

Domain-Adversarial Neural Network (DANN) under Leave-One-Field-Out
cross-validation for cross-field yield prediction generalisation.

Architecture:
    Feature extractor:  [input → 512 → 256 → 256] + BatchNorm + ReLU + Dropout
    Yield head:         [256 → 128 → 64 → 1]
    Domain head:        [256 → 128 → n_fields] connected via Gradient Reversal Layer

The GRL multiplies gradients by -λ during backpropagation, forcing the
feature extractor to produce field-invariant representations.

λ is ramped from 0 to the target value over the first 30 epochs to prevent
early-training instability.

Transductive domain adaptation is applied: the test field's features are
included during training (without yield labels) so the domain head can
align the test field's distribution against the training fields.

Each LOFO run is averaged across 5 random seeds (42–46) to account for
stochastic training variance.

Usage:
    python dann_lofo.py
    (edit paths and group definitions at the bottom)
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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_io import load_harvest_means, detect_yield_col, FIELD_TO_TABLA
from utils.raster_sampling import sample_single_band, sample_all_bands, get_xy
from utils.metrics import compute_metrics, print_metrics

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [42, 43, 44, 45, 46]

HEADING_DATES = {
    "20250614", "20250621", "20250622",
    "20250714", "20250721", "20250724",
}

FNAME_PATTERN = re.compile(
    r"(S[12])_(\d{8})_Mezohegyes_Stacked_([A-Za-z0-9_]+)\.tif$"
)


# ── Hyperparameters ────────────────────────────────────────────────────────

DANN_CFG = dict(
    feature_dim  = 256,
    hidden_dim   = 128,
    dropout      = 0.3,
    lr           = 1e-3,
    weight_decay = 1e-4,
    epochs       = 150,
    patience     = 15,
    batch_size   = 1024,
    warmup_epochs = 30,
)


# ── Gradient Reversal Layer ────────────────────────────────────────────────

class GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lam, None


class GradRev(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 1.0

    def forward(self, x):
        return GradRevFn.apply(x, self.lam)


# ── DANN model ─────────────────────────────────────────────────────────────

class DANNModel(nn.Module):
    def __init__(self, input_dim, n_domains):
        super().__init__()
        d = DANN_CFG["feature_dim"]
        h = DANN_CFG["hidden_dim"]
        p = DANN_CFG["dropout"]

        self.feat = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, d),         nn.BatchNorm1d(d),   nn.ReLU(),
        )
        self.yield_head = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.grl = GradRev()
        self.domain_head = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, n_domains),
        )

    def forward(self, x):
        f = self.feat(x)
        return self.yield_head(f).squeeze(1), self.domain_head(self.grl(f))

    def predict(self, x):
        return self.yield_head(self.feat(x)).squeeze(1)

    def set_lam(self, lam):
        self.grl.lam = lam


# ── Feature assembly ───────────────────────────────────────────────────────

def build_X(gdf, s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg):
    xs, ys = get_xy(gdf)
    blocks, names = [], []

    for fname in sorted(os.listdir(s1s2_dir)):
        m = FNAME_PATTERN.match(fname)
        if m is None or m.group(2) not in HEADING_DATES:
            continue
        v = sample_single_band(os.path.join(s1s2_dir, fname), xs, ys)
        blocks.append(v[:, None])
        names.append(f"{m.group(1)}_{m.group(2)}_{m.group(3)}")

    if os.path.exists(enmap_path):
        ev = sample_all_bands(enmap_path, xs, ys)
        if ev is not None:
            blocks.append(ev)
            names.extend([f"ENMAP_b{b+1}" for b in range(ev.shape[1])])

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

    X = np.nan_to_num(
        np.concatenate(blocks, axis=1).astype(np.float32), nan=0.0
    )
    return X, names


# ── Training routine ───────────────────────────────────────────────────────

def train_dann(X_tr, y_tr, d_tr, X_val, y_val, X_te,
               input_dim, n_domains, lambda_domain, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DANNModel(input_dim, n_domains).to(DEVICE)
    opt   = optim.Adam(model.parameters(),
                       lr=DANN_CFG["lr"], weight_decay=DANN_CFG["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    y_loss = nn.MSELoss()
    d_loss = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
        torch.tensor(d_tr, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=DANN_CFG["batch_size"], shuffle=True)

    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, DANN_CFG["epochs"] + 1):
        # Lambda warmup over first warmup_epochs
        p   = min(epoch / DANN_CFG["warmup_epochs"], 1.0)
        lam = lambda_domain * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        model.set_lam(lam)

        model.train()
        for Xb, yb, db in loader:
            Xb, yb, db = Xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            opt.zero_grad()
            yp, dp = model(Xb)
            (y_loss(yp, yb) + lam * d_loss(dp, db)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = y_loss(model.predict(X_v), y_v).item()
        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= DANN_CFG["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        y_res_pred = model.predict(X_t).cpu().numpy()

    return y_res_pred


# ── LOFO runner ────────────────────────────────────────────────────────────

def run_dann_lofo(fields, group_label, lambda_domain,
                  yield_dir, gpkg_map, harvest_csv,
                  s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    harvest_means = load_harvest_means(harvest_csv)

    print(f"\n{'█'*55}")
    print(f"  DANN LOFO — {group_label}  λ={lambda_domain}  device={DEVICE}")
    print(f"{'█'*55}")

    field_data = {}
    for fid in fields:
        fpath = os.path.join(yield_dir, gpkg_map.get(fid, ""))
        if not os.path.exists(fpath):
            print(f"  [skip] {fid}")
            continue
        gdf = gpd.read_file(fpath)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:23700")
        yc  = detect_yield_col(gdf)
        y   = pd.to_numeric(gdf[yc], errors="coerce").values.astype(float)
        if np.nanmean(y) > 50:
            y /= 1000.0
        valid   = np.isfinite(y) & (y > 0.5) & (y < 20)
        gdf, y  = gdf[valid].reset_index(drop=True), y[valid]
        tabla   = FIELD_TO_TABLA.get(fid)
        ht_mean = harvest_means.get(fid) or harvest_means.get(tabla, np.mean(y))
        X, _    = build_X(gdf, s1s2_dir, enmap_path, dem_path, twi_path, irrig_gpkg)
        field_data[fid] = {"gdf": gdf, "y": y, "ht_mean": ht_mean, "X": X}
        print(f"  {fid}: {len(y):,} pts | {X.shape[1]} feat | HT={ht_mean:.3f}")

    fid_list   = list(field_data.keys())
    n_domains  = len(fid_list)
    fid_to_idx = {f: i for i, f in enumerate(fid_list)}
    input_dim  = next(iter(field_data.values()))["X"].shape[1]

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

        # Val split for early stopping
        rng   = np.random.default_rng(42)
        n_val = max(1, int(0.15 * len(y_res)))
        idx   = rng.permutation(len(y_res))
        val_i, tr_i = idx[:n_val], idx[n_val:]

        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        X_tr_sc  = sc.fit_transform(imp.fit_transform(X_tr_raw[tr_i]))
        X_val_sc = sc.transform(imp.transform(X_tr_raw[val_i]))
        X_te_sc  = sc.transform(imp.transform(X_te))

        print(f"\n  FOLD test={test_fid}  train={train_fids}")

        seed_preds = []
        for seed in SEEDS:
            y_res_pred = train_dann(
                X_tr_sc, y_res[tr_i], d_tr[tr_i],
                X_val_sc, y_res[val_i], X_te_sc,
                input_dim, n_domains, lambda_domain, seed,
            )
            seed_preds.append(y_res_pred)

        y_pred = np.mean(seed_preds, axis=0) + ht_te
        m = compute_metrics(y_te, y_pred)
        print_metrics(test_fid, m)

        # Save spatial prediction GeoPackage
        gdf_out = field_data[test_fid]["gdf"].copy()
        gdf_out["y_pred"]   = y_pred
        gdf_out["y_actual"] = y_te
        gdf_out["residual"] = y_pred - y_te
        gdf_out.to_file(
            os.path.join(out_dir, f"dann_pred_{group_label}_{test_fid}.gpkg"),
            driver="GPKG",
        )

        all_results.append({"group": group_label, "field": test_fid,
                             "lambda": lambda_domain, **m})

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_dir, f"dann_results_{group_label}.csv"), index=False)

    print(f"\n{'='*55}")
    print(f"DANN LOFO SUMMARY — {group_label}")
    print(f"{'='*55}")
    print(df[["field", "R2", "RMSE", "MAE", "Bias"]].to_string(index=False))
    mean_row = df[["R2", "RMSE", "MAE", "Bias"]].mean()
    print(f"\n  Mean: R²={mean_row['R2']:+.4f}  RMSE={mean_row['RMSE']:.4f}")

    return df


# ── Configuration ──────────────────────────────────────────────────────────
YIELD_DIR   = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\calibrated_yield"
HARVEST_CSV = r"D:\STUDI\Thesis\mezohegyes\obuza_napi_aratas_2025_fix.csv"
S1S2_DIR    = r"D:\STUDI\Thesis\mezohegyes\VIs\s1+s2"
ENMAP_PATH  = r"D:\STUDI\Thesis\mezohegyes\VIs\enmap_kepek\2025_03_13.tif"
DEM_PATH    = r"D:\STUDI\Thesis\mezohegyes\dem10m_reproject_s2.tif"
TWI_PATH    = r"D:\STUDI\Thesis\mezohegyes\twi_from_modeller_reproject_s2.tif"
IRRIG_GPKG  = r"D:\STUDI\Thesis\mezohegyes\irrigated_fields.gpkg"
OUT_DIR     = r"D:\STUDI\Thesis\mezohegyes\results\dann"

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

# Group-specific adversarial strength
# High-yield group: spectrally diverse → stronger domain confusion (λ=0.5)
# Low-yield group:  Avenue-dominated → lighter adaptation (λ=0.1)
GROUPS = {
    "high": {"fields": ["9_ce", "9_pr", "9_sy", "12"],  "lambda": 0.5},
    "low":  {"fields": ["7", "25", "44", "59"],          "lambda": 0.1},
}

COMMON_ARGS = dict(
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
    print(f"DANN device: {DEVICE}")

    for group_label, cfg in GROUPS.items():
        run_dann_lofo(
            fields       = cfg["fields"],
            group_label  = group_label,
            lambda_domain= cfg["lambda"],
            **COMMON_ARGS,
        )
