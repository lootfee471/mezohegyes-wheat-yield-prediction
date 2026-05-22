"""
02_yield_calibration/stage1_machine_correction.py

Stage 1 of the two-stage yield monitor calibration pipeline.

Steps performed per field:
  1. Machine re-identification using SWATHWIDTH clustering + DBSCAN time-gap separation
  2. Mass computation: tonnes = yield (t/ha) × SWATHWIDTH (m) × DISTANCE (m) / 10,000
  3. Per-machine boundary-zone offset correction (least-squares on shared boundary points)
  4. Global weigh-bridge anchoring: scale so field total matches truck-scale total
  5. Gaussian spatial smoothing (20 m radius) + rescale to preserve anchor

Input:  Cleaned GeoJSON files from 01_yield_cleaning/
Output: Offset-corrected GeoPackage files with a 'yield_s1' column (t/ha)

Usage:
    python stage1_machine_correction.py
    (edit paths and WEIGH_BRIDGE_TOTALS at the bottom)

Note: The weigh-bridge totals (tonnes) must be filled in from the estate's
      daily harvest records before running.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter


# ── Gaussian smoothing radius (metres) ───────────────────────────────────
SMOOTH_RADIUS_M = 20.0

# ── Boundary zone for inter-machine matching (metres) ────────────────────
BOUNDARY_M = 15.0

# ── DBSCAN time-gap threshold for separating machines (seconds) ──────────
TIME_GAP_S = 300


def compute_tonnes(gdf):
    """
    Compute mass (tonnes) represented by each GPS point.
    Formula: t = yield (t/ha) × SWATHWIDTH (m) × DISTANCE (m) / 10,000
    """
    y = gdf["VRYIELDMAS"].values.astype(float)
    w = gdf["SWATHWIDTH"].values.astype(float)
    d = gdf["DISTANCE"].values.astype(float)
    return y * w * d / 10_000.0


def identify_machines(gdf):
    """
    Assign a machine ID to each point using SWATHWIDTH binning (each combine
    has a fixed cutting width) followed by DBSCAN time-gap clustering to
    separate machines that operated at different times on the same swath width.

    Returns an integer array of machine labels.
    """
    swath = gdf["SWATHWIDTH"].values.round(1)
    unique_widths = np.unique(swath)

    machine_id = np.full(len(gdf), -1, dtype=int)
    counter = 0

    for w in unique_widths:
        idx = np.where(swath == w)[0]
        if len(idx) == 0:
            continue

        # Try to use a timestamp column if present; otherwise treat as one group
        if "TIMESTAMP" in gdf.columns or "time" in gdf.columns.str.lower().tolist():
            tcol = next(
                c for c in gdf.columns if c.lower() in ("timestamp", "time", "datetime")
            )
            times = pd.to_datetime(gdf.iloc[idx][tcol], errors="coerce")
            times = times.sort_values()
            gaps = times.diff().dt.total_seconds().fillna(0).values

            group = 0
            groups = np.zeros(len(idx), dtype=int)
            for k in range(1, len(gaps)):
                if gaps[k] > TIME_GAP_S:
                    group += 1
                groups[k] = group

            for g in np.unique(groups):
                machine_id[idx[groups == g]] = counter
                counter += 1
        else:
            machine_id[idx] = counter
            counter += 1

    return machine_id


def boundary_offset_correction(gdf, machine_ids):
    """
    Find points from different machines that are within BOUNDARY_M of each other.
    At these shared boundary zones both machines measured the same crop, so any
    yield difference is sensor bias. Solve a least-squares system to find a
    correction offset per machine.

    Returns corrected VRYIELDMAS values as a numpy array.
    """
    xy = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    y  = gdf["VRYIELDMAS"].values.astype(float)
    unique_machines = np.unique(machine_ids[machine_ids >= 0])

    if len(unique_machines) < 2:
        return y.copy()

    offsets = np.zeros(unique_machines.max() + 1)

    # Compare each pair of machines at their shared boundary
    for i in range(len(unique_machines)):
        for j in range(i + 1, len(unique_machines)):
            m1, m2 = unique_machines[i], unique_machines[j]
            idx1 = np.where(machine_ids == m1)[0]
            idx2 = np.where(machine_ids == m2)[0]

            tree = KDTree(xy[idx2])
            dists, matches = tree.query(xy[idx1], k=1)
            near = dists < BOUNDARY_M

            if near.sum() < 5:
                continue

            y1_boundary = y[idx1[near]]
            y2_boundary = y[idx2[matches[near]]]
            diff = np.median(y1_boundary) - np.median(y2_boundary)

            # Apply half the difference to each machine in opposite directions
            offsets[m1] -= diff / 2.0
            offsets[m2] += diff / 2.0

    y_corrected = y.copy()
    for m in unique_machines:
        mask = machine_ids == m
        y_corrected[mask] = y[mask] + offsets[m]

    return y_corrected


def gaussian_smooth_grid(gdf, yield_col, radius_m):
    """
    Apply a simple Gaussian smooth in pixel space.
    Rasterizes yield points to a grid, smooths, then reads back point values.
    This is an approximation — a proper implementation would use scipy griddata
    and a Gaussian kernel in coordinate space.
    """
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    y  = gdf[yield_col].values.astype(float)

    # Bin to 10 m grid
    res = 10.0
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cols = ((xs - x_min) / res).astype(int)
    rows = ((y_max - ys) / res).astype(int)

    n_rows = rows.max() + 1
    n_cols = cols.max() + 1

    grid     = np.full((n_rows, n_cols), np.nan)
    count    = np.zeros((n_rows, n_cols))

    for k in range(len(y)):
        r, c = rows[k], cols[k]
        if np.isfinite(y[k]):
            grid[r, c] = (grid[r, c] if np.isfinite(grid[r, c]) else 0.0) + y[k]
            count[r, c] += 1

    with np.errstate(invalid="ignore"):
        grid = np.where(count > 0, grid / count, np.nan)

    # Gaussian filter (sigma in pixels)
    sigma = radius_m / res
    filled = np.where(np.isnan(grid), 0.0, grid)
    weights = np.where(np.isnan(grid), 0.0, 1.0)
    smoothed_vals   = gaussian_filter(filled,   sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)

    with np.errstate(invalid="ignore"):
        smoothed = np.where(smoothed_weights > 0, smoothed_vals / smoothed_weights, np.nan)

    # Read back point values from the smoothed grid
    y_smooth = np.full(len(y), np.nan)
    for k in range(len(y)):
        r, c = rows[k], cols[k]
        if 0 <= r < n_rows and 0 <= c < n_cols:
            y_smooth[k] = smoothed[r, c]

    return y_smooth


def calibrate_field(in_path, out_path, weigh_bridge_tonnes):
    """
    Full Stage 1 calibration for one field.
    """
    print(f"\n--- Stage 1: {os.path.basename(in_path)} ---")

    gdf = gpd.read_file(in_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:23700")

    print(f"  Points: {len(gdf):,}")

    # Step 1: machine identification
    machine_ids = identify_machines(gdf)
    n_machines = len(np.unique(machine_ids[machine_ids >= 0]))
    print(f"  Machines identified: {n_machines}")

    # Step 2: mass computation
    tonnes = compute_tonnes(gdf)
    total_raw = tonnes.sum()
    print(f"  Total mass (raw): {total_raw:.1f} t")

    # Step 3: per-machine boundary offset correction
    if n_machines > 1:
        y_corrected = boundary_offset_correction(gdf, machine_ids)
        gdf = gdf.copy()
        gdf["VRYIELDMAS"] = y_corrected
        tonnes_corrected = compute_tonnes(gdf)
    else:
        tonnes_corrected = tonnes

    # Step 4: global weigh-bridge anchoring
    total_corrected = tonnes_corrected.sum()
    if total_corrected > 0 and weigh_bridge_tonnes > 0:
        scale = weigh_bridge_tonnes / total_corrected
    else:
        scale = 1.0
        print("  WARN: weigh-bridge total missing, skipping global anchor")

    gdf["yield_s1"] = gdf["VRYIELDMAS"].values * scale
    print(f"  Weigh-bridge: {weigh_bridge_tonnes:.1f} t  |  scale factor: {scale:.4f}")

    # Step 5: Gaussian spatial smoothing
    y_smooth = gaussian_smooth_grid(gdf, "yield_s1", SMOOTH_RADIUS_M)
    valid = np.isfinite(y_smooth)

    # Rescale smoothed values so the field total still matches weigh-bridge
    if valid.sum() > 0 and y_smooth[valid].mean() > 0:
        smooth_scale = gdf["yield_s1"].values[valid].mean() / y_smooth[valid].mean()
        y_smooth = np.where(valid, y_smooth * smooth_scale, gdf["yield_s1"].values)

    gdf["yield_s1"] = y_smooth

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    print(f"  Saved → {out_path}")
    print(f"  Mean yield after Stage 1: {gdf['yield_s1'].mean():.3f} t/ha")


# ── Field configuration ───────────────────────────────────────────────────
# Edit these paths and weigh-bridge totals before running.

CLEAN_DIR = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\yield_cleaned"
OUT_DIR   = r"D:\STUDI\Thesis\mezohegyes\oszibuza-winterwheat\stage1_corrected"

# Weigh-bridge totals in tonnes, from estate daily harvest records.
# Fill in the actual values from your obuza_napi_aratas_2025_fix.csv.
WEIGH_BRIDGE = {
    "7":    None,   # fill in actual total tonnes
    "9_ce": None,
    "9_lg": None,
    "9_pr": None,
    "9_sy": None,
    "12":   None,
    "25":   None,
    "44":   None,
    "59":   None,
    "63":   None,
    "71":   None,
    "79":   None,
    "84":   None,
}

IN_FILES = {
    fid: os.path.join(CLEAN_DIR, f"{fid}_clean.geojson")
    for fid in WEIGH_BRIDGE
}


if __name__ == "__main__":
    print("=" * 55)
    print("  STAGE 1 — INTER-MACHINE CORRECTION + WEIGH-BRIDGE ANCHOR")
    print("=" * 55)

    for fid, in_path in IN_FILES.items():
        if not os.path.exists(in_path):
            print(f"\n[skip] {fid} — input not found")
            continue

        wb = WEIGH_BRIDGE.get(fid)
        if wb is None:
            print(f"\n[skip] {fid} — no weigh-bridge total set")
            continue

        out_path = os.path.join(OUT_DIR, f"{fid}_yield_s1.gpkg")
        calibrate_field(in_path, out_path, wb)

    print("\nStage 1 complete.")
