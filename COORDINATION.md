# 🗂️ Fichier de coordination — StockSense (projet Data Science)

> Ce document est le **contrat unique** partagé par tous les agents. Il définit
> l'architecture, les schémas de données, les signatures de fonctions et les
> fichiers d'échange. Un agent ne doit modifier QUE les fichiers qui lui sont
> attribués et respecter STRICTEMENT les interfaces ci-dessous.

## 🎯 Objectif du projet
Prévision probabiliste de la demande (14 jours) **et** optimisation des stocks
pour un distributeur/importateur (contexte Madagascar). Chaîne complète :
SQL → feature engineering → baselines scikit-learn → LSTM PyTorch → évaluation
→ simulation de politique de réapprovisionnement sensible au taux de change.

## 🧱 Architecture & flux de données
```
data_generation.py ─▶ data/*.csv
        │
build_database.py + sql/schema.sql + sql/features.sql ─▶ data/stocksense.db (table ml_features)
        │
   src/dataset.py  (load_splits)  ◀── CONTRAT d'accès commun
        │
   ┌────┴───────────────┐
models_sklearn.py     model_pytorch.py     ─▶ reports/preds_<name>.parquet
   └────┬───────────────┘
   evaluate.py ─▶ reports/metrics.json, reports/comparison.csv, figures/*.png
   inventory_optimization.py ─▶ reports/inventory_sim.csv, figures/inventory_*.png
        │
   run_pipeline.py (orchestration bout-en-bout)
```

## 🗃️ Schéma de la base (SQLite, star schema)
- **dim_product**(product_id INT PK, sku TEXT, category TEXT, category_id INT,
  unit_cost_eur REAL, is_imported INT, lead_time_days INT, supplier_country TEXT)
- **dim_store**(store_id INT PK, name TEXT, region TEXT, city TEXT, channel TEXT)
- **dim_date**(date TEXT PK 'YYYY-MM-DD', year, month, day, dow, weekofyear,
  is_weekend INT, is_holiday INT, holiday_name TEXT)
- **dim_fx**(date TEXT PK, mga_per_eur REAL)  ← coût d'import (innovation)
- **fct_sales**(date TEXT, store_id INT, product_id INT, units_sold INT,
  price_mga REAL, promo_flag INT, revenue_mga REAL)
  PK = (date, store_id, product_id)

## 📊 Contrat table `ml_features` (produite par build_database.py)
Une ligne = (date, store_id, product_id). Colonnes OBLIGATOIRES :
`date, store_id, product_id, units_sold` (cible)
+ features (voir `config.FEATURE_COLUMNS`) :
`price_mga, promo_flag, is_holiday, is_weekend, dow, month, weekofyear,
mga_per_eur, unit_cost_eur, is_imported, lead_time_days, category_id,
lag_1, lag_7, lag_14, lag_28, roll_mean_7, roll_mean_28, roll_std_7,
roll_mean_7_lag7, days_since_promo`
> Les lags/rolling sont calculés PAR (product_id, store_id) trié par date, en
> **décalant** correctement pour éviter toute fuite (pas de valeur du jour J
> dans les features de J). `days_since_promo` = nb de jours depuis la dernière promo.

## 🔁 Contrat fichiers de prédiction (`config.PRED_COLUMNS`)
Chaque modèle écrit `reports/preds_<name>.parquet` avec EXACTEMENT :
`date, store_id, product_id, y_true, y_pred, y_p10, y_p90`
sur la **période de test** (`date >= config.TEST_START`). `y_p10/y_p90` =
bornes de l'intervalle prédictif (P10/P90). Noms attendus :
`seasonal_naive`, `hgb` (HistGradientBoosting), `lstm`.

## 🧮 Contrat métriques (`evaluate.py` → reports/metrics.json)
Dict `{model_name: {mae, rmse, wmape, mape, pinball, coverage_80}}`.

## 👥 Attribution des agents
| Agent | Fichiers | Dépend de | Doit livrer |
|-------|----------|-----------|-------------|
| **A — Données/SQL** | `src/data_generation.py`, `sql/schema.sql`, `src/build_database.py`, `sql/features.sql` | config, dataset (contrat) | data/*.csv + data/stocksense.db + table ml_features, EXÉCUTÉ, compte de lignes affiché |
| **B — Baselines** | `src/models_sklearn.py` | dataset, DB | seasonal_naive + HistGradientBoosting (P50) + quantiles P10/P90 → preds_seasonal_naive.parquet, preds_hgb.parquet |
| **C — Deep Learning** | `src/model_pytorch.py` | dataset, DB | LSTM multi-horizon, perte pinball (quantiles) → preds_lstm.parquet + models/lstm.pt |
| **D — Évaluation** | `src/evaluate.py` | preds_*.parquet | metrics.json, comparison.csv, figures (barres MAE/WMAPE, courbes réel vs prédit, intervalle) |
| **E — Optimisation stock** | `src/inventory_optimization.py` | preds probabilistes | safety stock, ROP, EOQ, simulation service level & coût, sensibilité FX → inventory_sim.csv + figures |
| **Orchestrateur (moi)** | `src/config.py`, `src/dataset.py`, `src/run_pipeline.py`, `README.md`, intégration & debug | tout | pipeline exécutable + résultats réels + dépôt GitHub |

## ⚙️ Règles communes
- Python : `./.venv/bin/python`. Imports intra-package en relatif (`from . import config`).
- Exécuter un module depuis la racine : `./.venv/bin/python -m src.<module>`.
- Graine `config.SEED = 42` partout (numpy, torch). Résultats déterministes.
- Aucune fuite temporelle : features décalées, split par date.
- Chaque agent EXÉCUTE son module et corrige jusqu'à succès (mode autonome).

## 📌 Journal d'avancement
- [x] Squelette + venv + dépendances (orchestrateur)
- [x] config.py, dataset.py, COORDINATION.md (orchestrateur)
- [x] Agent A — données & base SQL (263 040 lignes ; ml_features 25 col ; splits 212160/22080/22080 ; anti-fuite vérifiée)
- [x] Agent B — baselines scikit-learn (seasonal_naive MAE 5.24/WMAPE 30.3% ; hgb MAE 3.37/WMAPE 19.4%)
- [x] Agent C — LSTM PyTorch (MAE 3.23/WMAPE 18.7%/couverture80 80.6% ; entraînement 96s)
- [x] Agent D — évaluation (LSTM meilleur : MAE 3.23/WMAPE 18.67%/couv80 80.6% ; hgb 3.37/19.44%/78.8% ; naive 5.24/30.25%/80.4% ; 4 figures)
- [x] Agent E — optimisation des stocks (politique modèle : fill rate >99.5%, −53 à −62% de valeur de stock, coût total −6 à −45% ; FX +10% → +8% coût stock, 80% sur imports)
- [x] Intégration run_pipeline + résultats (pipeline complet ré-exécuté OK en ~110s, résultats déterministes)
- [x] README + push GitHub → https://github.com/heniwizeup-dotcom/stocksense
