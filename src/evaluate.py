"""Agent D — Évaluation des modèles de prévision StockSense.

Charge les fichiers de prédiction `reports/preds_<name>.parquet`, calcule les
métriques point + probabilistes (MAE, RMSE, MAPE, WMAPE, perte pinball,
couverture 80 %), écrit `reports/metrics.json` et `reports/comparison.csv`, et
génère les figures matplotlib de comparaison dans `figures/`.

Exécution : `./.venv/bin/python -m src.evaluate` depuis la racine du projet.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # backend non interactif (pas d'affichage)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# --- Constantes de rendu ---------------------------------------------------
DPI = 130
QUANTILES = config.QUANTILES  # (0.1, 0.5, 0.9)
COLORS = {
    "seasonal_naive": "#9e9e9e",
    "hgb": "#1f77b4",
    "lstm": "#d62728",
}
LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "hgb": "HistGradientBoosting",
    "lstm": "LSTM",
}


# --- Chargement ------------------------------------------------------------
def load_predictions() -> dict[str, pd.DataFrame]:
    """Détecte et charge tous les fichiers reports/preds_*.parquet.

    Le nom du modèle est le suffixe après ``preds_``. La colonne ``date`` est
    normalisée en datetime (les baselines la sérialisent en string, le LSTM en
    Timestamp).
    """
    pattern = str(config.REPORTS_DIR / "preds_*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Aucun fichier de prédiction trouvé : {pattern}")

    preds: dict[str, pd.DataFrame] = {}
    for path in paths:
        name = os.path.basename(path)[len("preds_"):-len(".parquet")]
        df = pd.read_parquet(path)
        missing = set(config.PRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{name}: colonnes manquantes {missing}")
        df["date"] = pd.to_datetime(df["date"])
        # Types numériques homogènes (le LSTM écrit en float32).
        for col in ("y_true", "y_pred", "y_p10", "y_p90"):
            df[col] = df[col].astype("float64")
        df = df.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)
        preds[name] = df
        print(f"  chargé preds_{name}.parquet : {len(df)} lignes")
    return preds


# --- Métriques -------------------------------------------------------------
def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Perte pinball moyenne pour un quantile q."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()  # P50
    y_p10 = df["y_p10"].to_numpy()
    y_p90 = df["y_p90"].to_numpy()

    err = y_true - y_pred
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(abs_err / np.maximum(y_true, 1.0)) * 100.0)
    denom = np.sum(np.abs(y_true))
    wmape = float(np.sum(abs_err) / denom * 100.0) if denom else float("nan")

    # Perte pinball moyenne sur les 3 quantiles (P10, P50, P90).
    q_preds = {0.1: y_p10, 0.5: y_pred, 0.9: y_p90}
    pinball = float(np.mean([_pinball_loss(y_true, q_preds[q], q) for q in QUANTILES]))

    inside = (y_true >= y_p10) & (y_true <= y_p90)
    coverage_80 = float(np.mean(inside) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "wmape": round(wmape, 4),
        "pinball": round(pinball, 4),
        "coverage_80": round(coverage_80, 4),
    }


# --- Sélection d'une série représentative ----------------------------------
def pick_representative_series(df: pd.DataFrame) -> tuple[int, int]:
    """Choisit un (product_id, store_id) de volume moyen sur le test.

    On sélectionne la série dont le volume total est le plus proche de la
    médiane des volumes par série : ni valeur aberrante haute ni série creuse.
    """
    vol = df.groupby(["product_id", "store_id"])["y_true"].sum()
    median_vol = vol.median()
    product_id, store_id = vol.sub(median_vol).abs().idxmin()
    return int(product_id), int(store_id)


def _series(df: pd.DataFrame, product_id: int, store_id: int) -> pd.DataFrame:
    sub = df[(df["product_id"] == product_id) & (df["store_id"] == store_id)]
    return sub.sort_values("date").reset_index(drop=True)


# --- Figures ---------------------------------------------------------------
def fig_mae_wmape(metrics: dict[str, dict], models: list[str]) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.38

    mae_vals = [metrics[m]["mae"] for m in models]
    wmape_vals = [metrics[m]["wmape"] for m in models]

    b1 = ax1.bar(x - width / 2, mae_vals, width, label="MAE (unités)",
                 color="#1f77b4")
    ax1.set_ylabel("MAE (unités vendues)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, wmape_vals, width, label="WMAPE (%)",
                 color="#ff7f0e")
    ax2.set_ylabel("WMAPE (%)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    for bars, ax in ((b1, ax1), (b2, ax2)):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS.get(m, m) for m in models])
    ax1.set_title("Comparaison des modèles — MAE et WMAPE (jeu de test)")
    lines = [b1, b2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig_mae_wmape.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_forecast_vs_actual(preds: dict[str, pd.DataFrame], models: list[str],
                           product_id: int, store_id: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    ref = _series(preds[models[0]], product_id, store_id)
    ax.plot(ref["date"], ref["y_true"], color="black", linewidth=2.2,
            label="Réel", zorder=5)

    for m in models:
        s = _series(preds[m], product_id, store_id)
        ax.plot(s["date"], s["y_pred"], color=COLORS.get(m), linewidth=1.5,
                alpha=0.9, label=f"P50 — {LABELS.get(m, m)}")

    ax.set_title(
        f"Prévision vs réel — produit {product_id}, magasin {store_id} "
        f"(période de test)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Unités vendues")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig_forecast_vs_actual.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_prediction_interval(df: pd.DataFrame, model: str,
                            product_id: int, store_id: int) -> None:
    s = _series(df, product_id, store_id)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(s["date"], s["y_p10"], s["y_p90"], color=COLORS.get(model, "#d62728"),
                    alpha=0.22, label="Intervalle P10–P90")
    ax.plot(s["date"], s["y_pred"], color=COLORS.get(model, "#d62728"),
            linewidth=1.6, label="P50 (médiane)")
    ax.plot(s["date"], s["y_true"], color="black", linewidth=2.0,
            marker="o", markersize=3, label="Réel")

    inside = ((s["y_true"] >= s["y_p10"]) & (s["y_true"] <= s["y_p90"])).mean() * 100
    ax.set_title(
        f"Intervalle prédictif {LABELS.get(model, model)} — produit {product_id}, "
        f"magasin {store_id} (couverture locale {inside:.0f} %)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Unités vendues")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig_prediction_interval.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_coverage(metrics: dict[str, dict], models: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    vals = [metrics[m]["coverage_80"] for m in models]
    bars = ax.bar(x, vals, width=0.55,
                  color=[COLORS.get(m, "#1f77b4") for m in models])

    ax.axhline(80, color="red", linestyle="--", linewidth=1.5,
               label="Cible 80 %")
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f} %", (rect.get_x() + rect.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models])
    ax.set_ylabel("Couverture de l'intervalle P10–P90 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Calibration probabiliste — couverture 80 % par modèle")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "fig_coverage.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    print("== Agent D — Évaluation ==")
    print("Chargement des prédictions...")
    preds = load_predictions()

    print("Calcul des métriques...")
    metrics = {name: compute_metrics(df) for name, df in preds.items()}

    # metrics.json
    metrics_path = config.REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Écrit {metrics_path}")

    # comparison.csv (trié par MAE croissant)
    comparison = pd.DataFrame(metrics).T
    comparison.index.name = "model"
    comparison = comparison.sort_values("mae")
    comparison_path = config.REPORTS_DIR / "comparison.csv"
    comparison.to_csv(comparison_path)
    print(f"Écrit {comparison_path}")

    # Ordre des modèles pour les figures : baseline -> hgb -> lstm si présents.
    preferred = ["seasonal_naive", "hgb", "lstm"]
    models = [m for m in preferred if m in preds] + \
             [m for m in preds if m not in preferred]

    # Série représentative (volume médian) déterminée sur la baseline.
    ref_model = models[0]
    product_id, store_id = pick_representative_series(preds[ref_model])
    print(f"Série représentative : produit {product_id}, magasin {store_id}")

    print("Génération des figures...")
    fig_mae_wmape(metrics, models)
    fig_forecast_vs_actual(preds, models, product_id, store_id)
    prob_model = "lstm" if "lstm" in preds else models[-1]
    fig_prediction_interval(preds[prob_model], prob_model, product_id, store_id)
    fig_coverage(metrics, models)
    for fn in ("fig_mae_wmape.png", "fig_forecast_vs_actual.png",
               "fig_prediction_interval.png", "fig_coverage.png"):
        print(f"  écrit figures/{fn}")

    # Affichage final
    print("\n=== Tableau comparatif (trié par MAE croissant) ===")
    with pd.option_context("display.width", 120,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(comparison)

    best = comparison.index[0]
    print(f"\n>> Meilleur modèle (MAE min) : {LABELS.get(best, best)} "
          f"(MAE={comparison.loc[best, 'mae']:.4f}, "
          f"WMAPE={comparison.loc[best, 'wmape']:.2f}%, "
          f"couverture80={comparison.loc[best, 'coverage_80']:.1f}%)")


if __name__ == "__main__":
    main()
