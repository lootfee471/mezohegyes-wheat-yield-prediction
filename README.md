# Mezőhegyes Winter Wheat Yield Prediction

Code repository for the thesis:
**Field-Scale Winter Wheat Yield Prediction from Sentinel-1, Sentinel-2 and EnMAP Imagery Using Machine Learning and Deep Learning Model Comparison in Mezőhegyes, Hungary**

Aji Lutfi — Geoinformatics M.Sc., University of Szeged, 2026

---

## Overview

This repository contains all scripts used in the thesis, organized to follow the processing pipeline from raw combine harvester data through to final model evaluation. The workflow has six main stages:

1. **Yield data cleaning** — filter raw combine harvester point records
2. **Two-stage yield calibration** — inter-machine offset correction + harvest-table anchoring
3. **Feature extraction** — sample satellite rasters (S1, S2, EnMAP) at yield pixel locations
4. **Feature set comparison** — RF-LOFO across four sensor combinations (S1, S2, S1+S2, S1+S2+EnMAP)
5. **Model comparison** — RF, XGBoost, LightGBM, ANN, DANN under LOFO
6. **Stacked ensemble** — Ridge meta-learner on base model predictions

---

## Repository Structure

```
.
├── 01_yield_cleaning/
│   └── clean_yield_points.py         # Filter raw GeoJSON yield records
│
├── 02_yield_calibration/
│   ├── stage1_machine_correction.py  # Inter-machine offset + weigh-bridge anchoring
│   └── stage2_harvest_table_calib.py # Per-field multiplicative calibration
│
├── 03_feature_extraction/
│   ├── extract_s1s2_features.py      # Sample S1/S2 stacked rasters at pixel centroids
│   └── extract_enmap_terrain.py      # Sample EnMAP bands + DEM, TWI, irrigation flag
│
├── 04_modelling/
│   ├── feature_set_comparison.py     # RF-LOFO across S1/S2/EnMAP combinations
│   ├── model_comparison.py           # All five models under LOFO (330-feature input)
│   ├── dann_lofo.py                  # DANN with group-specific lambda and multi-seed eval
│   └── extended_experiment.py        # 345-feature run (+ SAR coherence + baresoil)
│
├── 05_ensemble/
│   └── stacked_ensemble.py           # Ridge meta-learner on base model OOF predictions
│
├── utils/
│   ├── data_io.py                    # Shared loaders: harvest table, field GeoPackages
│   ├── metrics.py                    # R², RMSE, MAE, Bias
│   └── raster_sampling.py            # Point-based raster sampling helpers
│
├── requirements.txt
└── README.md
```

---

## Data Inputs

| File type | Description |
|-----------|-------------|
| `*.geojson` | Raw yield monitor point records per field (from combine harvester) |
| `obuza_napi_aratas_2025_fix.csv` | Daily harvest table with weigh-bridge totals |
| `*_yield_10px.gpkg` | Stage 1 calibrated yield at 10 m pixel grid |
| `*_yield_10px_calib.gpkg` | Stage 2 calibrated yield (final, used in modelling) |
| `S1_YYYYMMDD_Mezohegyes_Stacked_*.tif` | Single-band S1/S2 index rasters |
| `2025_03_13.tif` | EnMAP Level-2A scene (219 bands) |
| `dem10m_reproject_s2.tif` | Digital Elevation Model at 10 m |
| `twi_from_modeller_reproject_s2.tif` | Topographic Wetness Index |
| `irrigated_fields.gpkg` | Irrigation polygon layer |

---

## Field Groups

| Group | Fields | Mean yield | Lambda (DANN) |
|-------|--------|-----------|---------------|
| High-yield | 9_ce, 9_pr, 9_sy, 12 | ~9.8 t/ha | 0.5 |
| Low-yield | 7, 25, 44, 59 | ~8.0 t/ha | 0.1 |
| Excluded | 71, 79, 84 | — | calibration factor outside 0.85–1.05 |

---

## Requirements

```
python >= 3.10
geopandas
rasterio
numpy
pandas
scikit-learn
xgboost
lightgbm
torch
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each stage in order:

```bash
# 1. Clean raw yield point files
python 01_yield_cleaning/clean_yield_points.py

# 2a. Stage 1 calibration (machine offset + weigh-bridge)
python 02_yield_calibration/stage1_machine_correction.py

# 2b. Stage 2 calibration (harvest table)
python 02_yield_calibration/stage2_harvest_table_calib.py

# 3. Extract features
python 03_feature_extraction/extract_s1s2_features.py
python 03_feature_extraction/extract_enmap_terrain.py

# 4. Run comparisons
python 04_modelling/feature_set_comparison.py
python 04_modelling/model_comparison.py
python 04_modelling/dann_lofo.py

# 5. Stacked ensemble
python 05_ensemble/stacked_ensemble.py
```

---

## Citation

If you use this code, please cite:

> Aji Lutfi (2026). *Field-Scale Winter Wheat Yield Prediction from Sentinel-1, Sentinel-2 and EnMAP Imagery Using Machine Learning and Deep Learning Model Comparison in Mezőhegyes, Hungary*. Geoinformatics M.Sc. Thesis, University of Szeged.
