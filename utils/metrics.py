"""
utils/metrics.py

Evaluation metrics for yield prediction: R², RMSE, MAE, Bias.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    """
    Compute standard regression metrics for yield prediction.

    Returns a dict with R2, RMSE, MAE, and Bias (mean prediction error).
    Bias is positive when the model overpredicts.
    """
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred) - np.mean(y_true)),
    }


def print_metrics(field_id, m, prefix=""):
    """Pretty-print metrics for one field."""
    print(
        f"  {prefix}{field_id:<8}  "
        f"R²={m['R2']:+.4f}  "
        f"RMSE={m['RMSE']:.4f}  "
        f"MAE={m['MAE']:.4f}  "
        f"Bias={m['Bias']:+.4f} t/ha"
    )
