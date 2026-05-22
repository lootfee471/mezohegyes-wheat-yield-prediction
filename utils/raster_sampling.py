"""
utils/raster_sampling.py

Point-based raster sampling at GeoDataFrame centroid coordinates.
Used by feature extraction scripts.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import rowcol


def sample_single_band(raster_path, xs, ys):
    """
    Sample a single-band raster at (x, y) coordinates.
    Returns a float32 array of length len(xs). NaN for nodata or missing file.
    """
    if not os.path.exists(str(raster_path)):
        return np.full(len(xs), np.nan, dtype=np.float32)

    try:
        with rasterio.open(raster_path) as src:
            rows, cols = rowcol(src.transform, xs, ys)
            rows = np.clip(np.asarray(rows, dtype=int), 0, src.height - 1)
            cols = np.clip(np.asarray(cols, dtype=int), 0, src.width - 1)
            values = src.read(1)[rows, cols].astype(np.float32)
            if src.nodata is not None:
                values[values == src.nodata] = np.nan
        return values
    except Exception as e:
        print(f"  [warn] could not sample {raster_path}: {e}")
        return np.full(len(xs), np.nan, dtype=np.float32)


def sample_all_bands(raster_path, xs, ys):
    """
    Sample all bands of a multi-band raster at (x, y) coordinates.
    Returns a (N, n_bands) float32 array, or None if the file is missing.
    """
    if not os.path.exists(str(raster_path)):
        return None

    try:
        with rasterio.open(raster_path) as src:
            rows, cols = rowcol(src.transform, xs, ys)
            rows = np.clip(np.asarray(rows, dtype=int), 0, src.height - 1)
            cols = np.clip(np.asarray(cols, dtype=int), 0, src.width - 1)
            data = src.read()  # (bands, height, width)
            values = data[:, rows, cols].T.astype(np.float32)  # (N, bands)
            if src.nodata is not None:
                values[values == src.nodata] = np.nan
        return values
    except Exception as e:
        print(f"  [warn] could not sample {raster_path}: {e}")
        return None


def get_xy(gdf):
    """Extract x, y coordinate arrays from a GeoDataFrame's geometry column."""
    return gdf.geometry.x.values, gdf.geometry.y.values
