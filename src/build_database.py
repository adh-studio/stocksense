"""Construction de la base SQLite StockSense.

1. Crée data/stocksense.db (repart de zéro).
2. Exécute sql/schema.sql (DDL du star schema).
3. Charge les CSV générés dans les tables de dimensions / faits.
4. Exécute sql/features.sql pour matérialiser la table ml_features.

Une fonction utilitaire enregistre `sqrt` sur la connexion si le build SQLite
n'expose pas les fonctions mathématiques (portabilité).
"""
from __future__ import annotations

import math
import sqlite3

import pandas as pd

from . import config

_TABLES = ["dim_product", "dim_store", "dim_date", "dim_fx", "fct_sales"]


def _register_math(con: sqlite3.Connection) -> None:
    """Garantit la disponibilité de sqrt (certains builds SQLite ne l'ont pas)."""
    con.create_function("sqrt", 1, lambda x: math.sqrt(x) if x is not None else None)


def _read_sql_file(path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def build() -> None:
    if not all((config.DATA_DIR / f"{t}.csv").exists() for t in _TABLES):
        raise FileNotFoundError(
            "CSV manquants dans data/. Lancer d'abord: python -m src.data_generation"
        )

    if config.DB_PATH.exists():
        config.DB_PATH.unlink()

    con = sqlite3.connect(config.DB_PATH)
    _register_math(con)
    try:
        # 1-2. Schéma
        con.executescript(_read_sql_file(config.SQL_DIR / "schema.sql"))

        # 3. Chargement des CSV
        for table in _TABLES:
            df = pd.read_csv(config.DATA_DIR / f"{table}.csv")
            df.to_sql(table, con, if_exists="append", index=False)

        con.commit()

        # 4. Feature engineering -> ml_features
        con.executescript(_read_sql_file(config.SQL_DIR / "features.sql"))
        con.commit()

        # --- Rapport ---
        cur = con.cursor()
        print("== Lignes par table ==")
        for table in _TABLES + ["ml_features"]:
            n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<14}: {n:,}")

        cols = [r[1] for r in cur.execute("PRAGMA table_info(ml_features)").fetchall()]
        print("\n== Colonnes ml_features ==")
        print(" ", cols)

        # Contrôle du contrat
        required = ["date", "store_id", "product_id", config.TARGET] + list(config.FEATURE_COLUMNS)
        missing = [c for c in required if c not in cols]
        if missing:
            raise RuntimeError(f"Colonnes manquantes dans ml_features: {missing}")
        print("\nContrat ml_features OK (toutes les colonnes requises presentes).")
    finally:
        con.close()

    print(f"\nBase construite: {config.DB_PATH}")


if __name__ == "__main__":
    build()
