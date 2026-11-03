"""Layer 1: demand forecasting with quantile gradient boosting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import QUANTILES
from .features import CATEGORICAL, FEATURES, TARGET


def smape(actual, forecast) -> float:
    """Symmetric mean absolute percentage error, in %. A day where both are 0 counts as 0 error."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.abs(a) + np.abs(f)
    err = np.where(denom == 0, 0.0, 2 * np.abs(f - a) / np.where(denom == 0, 1, denom))
    return float(100 * err.mean())


def train_quantile_models(train: pd.DataFrame, quantiles=QUANTILES, max_iter=300, verbose=True):
    """One model per quantile. The 0.5 model is the median forecast.

    Models predict sales / (28-day average); predict_quantiles converts back to units.
    """
    X, y = train[FEATURES], train[TARGET]
    models = {}
    for q in quantiles:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=max_iter, learning_rate=0.08,
            max_leaf_nodes=31, min_samples_leaf=40, categorical_features=CATEGORICAL,
            early_stopping=False, random_state=0,
        )
        m.fit(X, y)
        models[q] = m
        if verbose:
            print(f"  trained quantile {q:.1f}")
    return models


def predict_quantiles(models: dict, rows: pd.DataFrame) -> np.ndarray:
    """Array of shape (rows, quantiles), sorted so quantiles never cross."""
    qs = sorted(models)
    scale = rows["mean_28"].where(rows["mean_28"] > 0, 1.0).to_numpy()[:, None]
    preds = np.column_stack([models[q].predict(rows[FEATURES]) for q in qs]) * scale
    return np.clip(np.sort(preds, axis=1), 0, None)


def quantile_at(qpreds: np.ndarray, levels, quantiles=QUANTILES) -> np.ndarray:
    """Demand that is enough on a share `level` of days, interpolated between the trained quantiles.

    `levels` can be one number or one value per row (each item has its own critical ratio).
    """
    qs = np.asarray(sorted(quantiles))
    levels = np.broadcast_to(np.clip(np.asarray(levels, float), qs[0], qs[-1]), (qpreds.shape[0],))
    idx = np.clip(np.searchsorted(qs, levels) , 1, len(qs) - 1)
    lo, hi = qs[idx - 1], qs[idx]
    w = (levels - lo) / (hi - lo)
    rows = np.arange(qpreds.shape[0])
    return qpreds[rows, idx - 1] * (1 - w) + qpreds[rows, idx] * w
