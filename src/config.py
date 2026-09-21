"""Configuration centrale du projet StockSense.

Toutes les constantes (chemins, graine aléatoire, horizon de prévision,
quantiles) sont définies ici pour garantir la reproductibilité entre les
modules construits par les différents agents.
"""
from __future__ import annotations

from pathlib import Path

# --- Chemins ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
DB_PATH = DATA_DIR / "stocksense.db"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproductibilité ------------------------------------------------------
SEED = 42

# --- Périmètre des données synthétiques ------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"          # 3 ans de ventes quotidiennes
N_STORES = 6                      # points de distribution (Antananarivo, Tamatave, ...)
N_PRODUCTS = 40                   # SKU import/distribution

# --- Découpage temporel (walk-forward) -------------------------------------
# Split par date pour éviter toute fuite temporelle.
VALID_START = "2024-07-01"        # 6 derniers mois -> validation + test
TEST_START = "2024-10-01"         # 3 derniers mois -> test (hold-out)

# --- Prévision -------------------------------------------------------------
HORIZON = 14                      # prévision à 14 jours (réappro hebdo/bihebdo)
QUANTILES = (0.1, 0.5, 0.9)       # prévision probabiliste P10 / P50 / P90
TARGET = "units_sold"

# --- Colonnes du dataset ML (contrat partagé entre agents) -----------------
# Voir COORDINATION.md pour le schéma détaillé de la table `ml_features`.
FEATURE_COLUMNS = [
    "price_mga", "promo_flag", "is_holiday", "is_weekend",
    "dow", "month", "weekofyear",
    "mga_per_eur", "unit_cost_eur", "is_imported", "lead_time_days",
    "category_id",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_28", "roll_std_7",
    "roll_mean_7_lag7", "days_since_promo",
]

# --- Schéma canonique des fichiers de prédiction ---------------------------
# Chaque modèle écrit reports/preds_<name>.parquet avec EXACTEMENT ces colonnes.
PRED_COLUMNS = [
    "date", "store_id", "product_id",
    "y_true", "y_pred", "y_p10", "y_p90",
]
