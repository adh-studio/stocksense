"""Orchestration bout-en-bout du pipeline StockSense.

Exécute successivement toutes les étapes, chacune dans un sous-processus isolé
(`python -m src.<module>`), et s'arrête à la première erreur. Reproduit
l'intégralité du projet à partir de zéro :

    ./.venv/bin/python -m src.run_pipeline

Étapes :
  1. Génération des données synthétiques           (src.data_generation)
  2. Construction de la base + table de features    (src.build_database)
  3. Baselines scikit-learn                         (src.models_sklearn)
  4. LSTM PyTorch (prévision probabiliste)          (src.model_pytorch)
  5. Évaluation comparative                         (src.evaluate)
  6. Optimisation des stocks                        (src.inventory_optimization)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Génération des données", "src.data_generation"),
    ("Construction base + features", "src.build_database"),
    ("Baselines scikit-learn", "src.models_sklearn"),
    ("LSTM PyTorch", "src.model_pytorch"),
    ("Évaluation comparative", "src.evaluate"),
    ("Optimisation des stocks", "src.inventory_optimization"),
]


def run() -> int:
    python = sys.executable
    print("=" * 70)
    print("STOCKSENSE — PIPELINE COMPLET")
    print("=" * 70)
    for i, (label, module) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label}  →  python -m {module}")
        t0 = time.time()
        proc = subprocess.run(
            [python, "-m", module], cwd=ROOT, text=True
        )
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"\n❌ Échec à l'étape « {label} » (code {proc.returncode}).")
            return proc.returncode
        print(f"✅ {label} terminé en {dt:.1f}s")
    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ — voir reports/ et figures/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
