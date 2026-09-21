"""Agent B — Baselines scikit-learn pour StockSense.

Deux approches de prévision probabiliste (P10 / P50 / P90) évaluées sur la
période de test (hold-out) :

1. ``seasonal_naive`` : prédiction = ventes du même jour la semaine précédente
   (colonne ``lag_7``). L'intervalle P10/P90 provient des quantiles empiriques
   des résidus (``units_sold - lag_7``) mesurés sur le train.
2. ``hgb`` : ``HistGradientBoostingRegressor`` — un modèle P50 (perte MSE) et
   deux modèles quantiles (0.1 et 0.9).

Chaque approche écrit ``reports/preds_<name>.parquet`` avec EXACTEMENT les
colonnes de ``config.PRED_COLUMNS``. Les modèles HGB sont sauvegardés dans
``models/`` via joblib.

Exécution : ``./.venv/bin/python -m src.models_sklearn`` depuis la racine.
"""
from __future__ import annotations

import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .dataset import Splits, load_splits


def _set_seed() -> None:
    """Fixe la graine pour numpy et random (déterminisme)."""
    np.random.seed(config.SEED)
    random.seed(config.SEED)


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE = somme|erreurs| / somme|réel|."""
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def _sort_quantiles(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray):
    """Garantit P10 <= P50 <= P90 ligne par ligne (tri des 3 colonnes)."""
    stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
    return stacked[0], stacked[1], stacked[2]


def _make_pred_frame(
    split: pd.DataFrame,
    y_pred: np.ndarray,
    y_p10: np.ndarray,
    y_p90: np.ndarray,
) -> pd.DataFrame:
    """Construit un DataFrame au format strict config.PRED_COLUMNS."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(split["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            "store_id": split["store_id"].to_numpy(),
            "product_id": split["product_id"].to_numpy(),
            "y_true": split[config.TARGET].to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "y_p10": np.asarray(y_p10, dtype=float),
            "y_p90": np.asarray(y_p90, dtype=float),
        }
    )
    return out[config.PRED_COLUMNS]


def run_seasonal_naive(s: Splits) -> pd.DataFrame:
    """Baseline saisonnière : y = lag_7, intervalle via quantiles des résidus."""
    # Résidus sur le train : units_sold - lag_7 (même saisonnalité hebdo).
    resid = (s.train[config.TARGET] - s.train["lag_7"]).to_numpy(dtype=float)
    resid = resid[~np.isnan(resid)]
    q10, q90 = np.quantile(resid, [0.10, 0.90])

    y_pred = s.test["lag_7"].to_numpy(dtype=float)
    y_p10 = np.clip(y_pred + q10, 0.0, None)
    y_p90 = np.clip(y_pred + q90, 0.0, None)
    y_pred = np.clip(y_pred, 0.0, None)

    # Cohérence des quantiles (les décalages empiriques peuvent se croiser
    # après bornage à 0).
    y_p10, y_pred, y_p90 = _sort_quantiles(y_p10, y_pred, y_p90)

    frame = _make_pred_frame(s.test, y_pred, y_p10, y_p90)
    path = config.REPORTS_DIR / "preds_seasonal_naive.parquet"
    frame.to_parquet(path, index=False)

    mae = _mae(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    wmape = _wmape(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    print(f"[seasonal_naive] MAE={mae:.4f}  WMAPE={wmape:.4f}  -> {path}")
    return frame


def run_hgb(s: Splits) -> pd.DataFrame:
    """HistGradientBoosting : P50 (MSE) + P10/P90 (perte pinball quantile)."""
    X_train = s.train[s.feature_columns]
    y_train = s.train[s.target]
    X_test = s.test[s.feature_columns]

    common = dict(
        learning_rate=0.05,
        max_iter=500,
        max_depth=None,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=config.SEED,
    )

    # --- Point (P50) -----------------------------------------------------
    m_p50 = HistGradientBoostingRegressor(loss="squared_error", **common)
    m_p50.fit(X_train, y_train)

    # --- Quantiles P10 / P90 --------------------------------------------
    m_p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **common)
    m_p10.fit(X_train, y_train)

    m_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **common)
    m_p90.fit(X_train, y_train)

    y_pred = np.clip(m_p50.predict(X_test), 0.0, None)
    y_p10 = np.clip(m_p10.predict(X_test), 0.0, None)
    y_p90 = np.clip(m_p90.predict(X_test), 0.0, None)

    # Garantit P10 <= P50 <= P90 (les 3 modèles sont indépendants).
    y_p10, y_pred, y_p90 = _sort_quantiles(y_p10, y_pred, y_p90)

    # Sauvegarde des modèles.
    joblib.dump(m_p50, config.MODELS_DIR / "hgb_p50.joblib")
    joblib.dump(m_p10, config.MODELS_DIR / "hgb_p10.joblib")
    joblib.dump(m_p90, config.MODELS_DIR / "hgb_p90.joblib")

    frame = _make_pred_frame(s.test, y_pred, y_p10, y_p90)
    path = config.REPORTS_DIR / "preds_hgb.parquet"
    frame.to_parquet(path, index=False)

    mae = _mae(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    wmape = _wmape(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    print(f"[hgb] MAE={mae:.4f}  WMAPE={wmape:.4f}  -> {path}")
    return frame


def main() -> None:
    _set_seed()
    s = load_splits()
    print(
        f"Splits chargés — train={s.train.shape[0]}  "
        f"valid={s.valid.shape[0]}  test={s.test.shape[0]}"
    )

    run_seasonal_naive(s)
    run_hgb(s)


if __name__ == "__main__":
    main()
