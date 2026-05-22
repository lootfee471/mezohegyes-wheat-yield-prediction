"""
utils/data_io.py

Shared helpers for loading harvest table means and field GeoPackages.
Called by scripts in all modelling stages.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd


# Priority order for detecting the yield column in a GeoPackage
YIELD_COL_PRIORITY = [
    "yield_calib_tha", "yield_tha", "YIELD_THA",
    "VALUE", "value", "yield", "VRYIELDMAS",
]

# Maps sub-field IDs to harvest table Tábla numbers
FIELD_TO_TABLA = {
    "7":    7,
    "9_ce": 9, "9_lg": 9, "9_pr": 9, "9_sy": 9,
    "12":   12,
    "25":   25,
    "44":   44,
    "59":   59,
    "63":   63,
    "71":   71,
    "79":   79,
    "84":   84,
}

# Field 9 is split by variety — these keywords match the Fajta column
VARIETY_TO_FIELD = {
    "Celebrity":   "9_ce",
    "LG":          "9_lg",
    "Providence":  "9_pr",
    "SY":          "9_sy",
}


def load_harvest_means(csv_path):
    """
    Parse the daily harvest table CSV and return a dict of
    {field_id: weighted_mean_yield_tha}.

    Field 9 is split per variety using keyword matching on the Fajta column.
    All other fields use the field-level weighted mean (tonnes / hectares).
    """
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", decimal=".")
    df.columns = [
        "Tabla", "Ontozott", "Ontozoberend", "Ontozovis",
        "Fajta", "Hasznositas", "Gen",
        "Aratott_br_t", "Aratott_ha", "Kombajn", "Vezeto",
        "Napi_t_ha", "Datum",
    ]

    df["Tabla"] = pd.to_numeric(df["Tabla"], errors="coerce")
    df["Aratott_br_t"] = pd.to_numeric(
        df["Aratott_br_t"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df["Aratott_ha"] = pd.to_numeric(
        df["Aratott_ha"].astype(str).str.replace(",", "."), errors="coerce"
    )
    df = df.dropna(subset=["Tabla", "Aratott_ha", "Aratott_br_t"])
    df["Tabla"] = df["Tabla"].astype(int)
    df["Fajta"] = df["Fajta"].astype(str).str.strip()

    means = {}

    for tabla, grp in df.groupby("Tabla"):
        means[tabla] = float(grp["Aratott_br_t"].sum() / grp["Aratott_ha"].sum())

    # Per-variety breakdown for field 9
    field9 = df[df["Tabla"] == 9]
    for keyword, field_id in VARIETY_TO_FIELD.items():
        rows = field9[field9["Fajta"].str.contains(keyword, case=False, na=False)]
        if len(rows) > 0:
            means[field_id] = float(rows["Aratott_br_t"].sum() / rows["Aratott_ha"].sum())
        else:
            means[field_id] = means.get(9, np.nan)

    return means


def detect_yield_col(gdf):
    """
    Find the yield column in a GeoDataFrame by checking known column names first,
    then falling back to numeric heuristics (values between 0.5 and 30 t/ha).
    """
    col = next((c for c in YIELD_COL_PRIORITY if c in gdf.columns), None)
    if col is not None:
        return col

    # Fallback: look for any numeric column that looks like yield
    for c in gdf.columns:
        if c == "geometry":
            continue
        converted = pd.to_numeric(gdf[c], errors="coerce")
        if converted.notna().mean() > 0.8 and 0.5 < converted.mean() < 30:
            return c

    return None


def load_field(fid, gpkg_dir, gpkg_map, harvest_means, yield_range=(0.5, 20.0)):
    """
    Load a single field GeoPackage, detect the yield column, filter to valid
    yield range, and look up the harvest-table mean.

    Returns (gdf, y_array, ht_mean) or None if the file is missing or unusable.
    """
    gpkg_name = gpkg_map.get(fid)
    if gpkg_name is None:
        return None

    fpath = os.path.join(gpkg_dir, gpkg_name)
    if not os.path.exists(fpath):
        print(f"  [skip] {fid} — file not found: {fpath}")
        return None

    gdf = gpd.read_file(fpath)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:23700")

    yc = detect_yield_col(gdf)
    if yc is None:
        print(f"  [skip] {fid} — no yield column found")
        return None

    y = pd.to_numeric(gdf[yc], errors="coerce").values.astype(np.float32)
    # Convert from kg/ha if necessary (values > 50 are almost certainly kg/ha)
    if np.nanmean(y) > 50:
        y = y / 1000.0

    valid = np.isfinite(y) & (y > yield_range[0]) & (y < yield_range[1])
    gdf = gdf[valid].reset_index(drop=True)
    y = y[valid]

    tabla = FIELD_TO_TABLA.get(fid)
    ht_mean = harvest_means.get(fid) or harvest_means.get(tabla)
    if ht_mean is None:
        ht_mean = float(np.mean(y))
        print(f"  [warn] {fid} — HT mean not found, using map mean {ht_mean:.3f}")

    return gdf, y, ht_mean
