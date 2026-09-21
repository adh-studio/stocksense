"""Contrat d'accès aux données — utilisé par tous les modèles.

Charge la table `ml_features` (matérialisée par build_database.py à partir
des requêtes SQL de feature engineering) et fournit un découpage temporel
train/valid/test sans fuite. TOUS les agents de modélisation doivent passer
par `load_splits()` afin de partager exactement le même dataset.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list
    target: str


def load_features() -> pd.DataFrame:
    """Charge la table ml_features complète depuis SQLite, triée par clé."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(
            f"{config.DB_PATH} introuvable. Lancer d'abord build_database.py."
        )
    con = sqlite3.connect(config.DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM ml_features", con)
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)
    return df


def load_splits() -> Splits:
    """Découpe temporel train / valid / test (walk-forward, sans fuite)."""
    df = load_features()
    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET]).reset_index(drop=True)

    valid_start = pd.Timestamp(config.VALID_START)
    test_start = pd.Timestamp(config.TEST_START)

    train = df[df["date"] < valid_start].copy()
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()

    return Splits(
        train=train,
        valid=valid,
        test=test,
        feature_columns=list(config.FEATURE_COLUMNS),
        target=config.TARGET,
    )


if __name__ == "__main__":
    s = load_splits()
    print("train", s.train.shape, "valid", s.valid.shape, "test", s.test.shape)
    print("features:", s.feature_columns)
