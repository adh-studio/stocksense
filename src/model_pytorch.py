"""Agent C — LSTM de prévision probabiliste de la demande (PyTorch).

Modèle LSTM entraîné avec une **perte pinball (quantile)** produisant
simultanément P10 / P50 / P90 pour la demande quotidienne, par série
(product_id, store_id). Monotonie P10 <= P50 <= P90 garantie par une
paramétrisation cumulative (softplus).

Exécution :  ``./.venv/bin/python -m src.model_pytorch``

Sortie stricte (``config.PRED_COLUMNS``) :
    reports/preds_lstm.parquet -> date, store_id, product_id,
                                   y_true, y_pred, y_p10, y_p90
Modèle sauvegardé dans models/lstm.pt.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .dataset import load_features

# --------------------------------------------------------------------------- #
# Hyperparamètres
# --------------------------------------------------------------------------- #
SEQ_LEN = 28          # fenêtre glissante (jours) en entrée du LSTM
HIDDEN = 64           # taille de l'état caché
NUM_LAYERS = 2        # couches LSTM empilées
DROPOUT = 0.1
BATCH_SIZE = 512
MAX_EPOCHS = 15
PATIENCE = 3          # early stopping (sur la perte pinball de validation)
LR = 1e-3
QUANTILES = tuple(config.QUANTILES)   # (0.1, 0.5, 0.9)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Construction des séquences (fenêtre glissante par série)
# --------------------------------------------------------------------------- #
class SequenceDataset(Dataset):
    """Fenêtres glissantes de longueur ``seq_len`` -> cible du dernier jour.

    Les features ``ml_features`` sont déjà décalées (aucune valeur du jour J
    dans les features de J), donc la fenêtre [t-L+1 .. t] utilisée pour prédire
    ``units_sold[t]`` n'introduit aucune fuite temporelle. Les fenêtres ne
    traversent jamais une frontière de série.
    """

    def __init__(self, feats: np.ndarray, targets: np.ndarray,
                 end_indices: np.ndarray, seq_len: int):
        self.feats = torch.from_numpy(feats)          # (N, F) float32
        self.targets = torch.from_numpy(targets)      # (N,)   float32
        self.ends = end_indices                       # positions de fin valides
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int):
        e = int(self.ends[i])
        x = self.feats[e - self.seq_len + 1: e + 1]   # (L, F)
        y = self.targets[e]                           # scalaire
        return x, y


def _build_index(df: pd.DataFrame, seq_len: int):
    """Prépare les arrays globaux et les indices de fin de fenêtre par split.

    Retourne feats (N,F) brut, targets (N,), et trois listes d'indices de fin
    (train/valid/test) selon la date de la cible. Les fenêtres restent dans
    une même série (product_id, store_id).
    """
    feats_all = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets_all = df[config.TARGET].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy()

    valid_start = np.datetime64(config.VALID_START)
    test_start = np.datetime64(config.TEST_START)

    train_ends, valid_ends, test_ends = [], [], []

    # positions contiguës par série (df déjà trié product_id, store_id, date)
    codes = (df["product_id"].astype(str) + "_" + df["store_id"].astype(str)).to_numpy()
    n = len(df)
    start = 0
    for i in range(1, n + 1):
        if i == n or codes[i] != codes[start]:
            # série = [start, i)
            for e in range(start + seq_len - 1, i):
                d = dates[e]
                if d < valid_start:
                    train_ends.append(e)
                elif d < test_start:
                    valid_ends.append(e)
                else:
                    test_ends.append(e)
            start = i

    return (feats_all, targets_all,
            np.asarray(train_ends, dtype=np.int64),
            np.asarray(valid_ends, dtype=np.int64),
            np.asarray(test_ends, dtype=np.int64))


# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #
class QuantileLSTM(nn.Module):
    """LSTM -> tête linéaire à 3 sorties (quantiles), monotonie garantie."""

    def __init__(self, n_features: int, hidden: int, num_layers: int,
                 n_quantiles: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_quantiles)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, L, H)
        last = out[:, -1, :]           # dernier pas de temps
        raw = self.head(last)          # (B, Q)
        # Paramétrisation cumulative : 0 <= q0 <= q1 <= q2
        q0 = self.softplus(raw[:, 0:1])
        q1 = q0 + self.softplus(raw[:, 1:2])
        q2 = q1 + self.softplus(raw[:, 2:3])
        return torch.cat([q0, q1, q2], dim=1)   # (B, 3)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor,
                 quantiles) -> torch.Tensor:
    """Perte pinball moyennée sur les quantiles. pred (B,Q), target (B,)."""
    target = target.unsqueeze(1)                      # (B,1)
    losses = []
    for k, q in enumerate(quantiles):
        err = target - pred[:, k:k + 1]
        losses.append(torch.maximum(q * err, (q - 1.0) * err))
    return torch.mean(torch.cat(losses, dim=1))


# --------------------------------------------------------------------------- #
# Entraînement / évaluation
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = pinball_loss(pred, y, QUANTILES)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


def main() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = _select_device()
    print(f"[lstm] device = {device}")

    # --- Données ----------------------------------------------------------
    df = load_features()
    # On retire les lignes où features/cible sont NaN (débuts de série : lags).
    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET]).reset_index(drop=True)
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    feats_all, targets_all, train_ends, valid_ends, test_ends = _build_index(df, SEQ_LEN)

    # --- Normalisation : StandardScaler ajusté sur le TRAIN uniquement ----
    train_rows = df["date"].to_numpy() < np.datetime64(config.VALID_START)
    mean = feats_all[train_rows].mean(axis=0)
    std = feats_all[train_rows].std(axis=0)
    std[std < 1e-8] = 1.0
    feats_scaled = ((feats_all - mean) / std).astype(np.float32)

    print(f"[lstm] séquences  train={len(train_ends)}  "
          f"valid={len(valid_ends)}  test={len(test_ends)}")

    g = torch.Generator()
    g.manual_seed(config.SEED)
    train_ds = SequenceDataset(feats_scaled, targets_all, train_ends, SEQ_LEN)
    valid_ds = SequenceDataset(feats_scaled, targets_all, valid_ends, SEQ_LEN)
    test_ds = SequenceDataset(feats_scaled, targets_all, test_ends, SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- Modèle -----------------------------------------------------------
    model = QuantileLSTM(
        n_features=len(config.FEATURE_COLUMNS),
        hidden=HIDDEN, num_layers=NUM_LAYERS,
        n_quantiles=len(QUANTILES), dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Boucle d'entraînement + early stopping ---------------------------
    best_valid = float("inf")
    best_state = None
    epochs_no_improve = 0
    t0 = time.time()
    print(f"{'epoch':>5} | {'train_pinball':>13} | {'valid_pinball':>13}")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr = _run_epoch(model, train_dl, device, optimizer)
        va = _run_epoch(model, valid_dl, device, optimizer=None)
        print(f"{epoch:5d} | {tr:13.4f} | {va:13.4f}")
        if va < best_valid - 1e-4:
            best_valid = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[lstm] early stopping (epoch {epoch})")
                break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[lstm] entraînement terminé en {train_time:.1f}s "
          f"(best valid pinball={best_valid:.4f})")

    # --- Prédiction sur le test ------------------------------------------
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_dl:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
    preds = np.concatenate(preds, axis=0)             # (n_test, 3)
    preds = np.clip(preds, 0.0, None)                 # bornées >= 0
    # sécurité monotonie (déjà garantie par la paramétrisation)
    preds = np.sort(preds, axis=1)

    p10 = preds[:, 0]
    p50 = preds[:, 1]
    p90 = preds[:, 2]

    out = pd.DataFrame({
        "date": df["date"].to_numpy()[test_ends],
        "store_id": df["store_id"].to_numpy()[test_ends],
        "product_id": df["product_id"].to_numpy()[test_ends],
        "y_true": targets_all[test_ends].astype(float),
        "y_pred": p50,
        "y_p10": p10,
        "y_p90": p90,
    })[config.PRED_COLUMNS]
    out = out.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

    # --- Métriques indicatives -------------------------------------------
    y_true = out["y_true"].to_numpy()
    y_pred = out["y_pred"].to_numpy()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum(np.abs(y_true)))
    wmape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")
    coverage = float(np.mean((y_true >= out["y_p10"].to_numpy()) &
                             (y_true <= out["y_p90"].to_numpy())))
    print(f"[lstm] TEST  MAE={mae:.3f}  WMAPE={wmape:.3%}  "
          f"coverage_80={coverage:.3%}  n={len(out)}")
    neg = int((out[["y_pred", "y_p10", "y_p90"]].to_numpy() < 0).sum())
    mono = bool(((out["y_p10"] <= out["y_pred"]) & (out["y_pred"] <= out["y_p90"])).all())
    print(f"[lstm] contrôles : négatifs={neg}  monotonie(P10<=P50<=P90)={mono}")

    # --- Sauvegardes ------------------------------------------------------
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = config.REPORTS_DIR / "preds_lstm.parquet"
    out.to_parquet(pred_path, index=False)
    print(f"[lstm] écrit {pred_path}  ({len(out)} lignes, {len(out.columns)} colonnes)")

    model_path = config.MODELS_DIR / "lstm.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "scaler_mean": mean, "scaler_std": std,
        "seq_len": SEQ_LEN, "hidden": HIDDEN, "num_layers": NUM_LAYERS,
        "quantiles": QUANTILES, "feature_columns": list(config.FEATURE_COLUMNS),
    }, model_path)
    print(f"[lstm] modèle sauvegardé -> {model_path}")


if __name__ == "__main__":
    main()
