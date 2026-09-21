"""Génération de données synthétiques réalistes pour StockSense.

Simule l'activité d'un distributeur / importateur à Madagascar :
catalogue produits (locaux + importés), réseau de magasins régionaux,
calendrier avec jours fériés malgaches, série de taux de change Ariary/Euro,
et ventes quotidiennes combinant tendance, saisonnalités hebdomadaire et
annuelle, promotions, effet des jours fériés, bruit de Poisson et sensibilité
au taux de change pour les produits importés.

Tout est déterministe (graine `config.SEED`). Écrit 5 CSV dans `data/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Référentiels métier
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Alimentaire": 1,
    "Hygiene": 2,
    "Electronique": 3,
    "Textile": 4,
    "Maison": 5,
}

# Régions / villes de Madagascar avec canal de distribution
STORES = [
    ("Tana Analakely",   "Analamanga",     "Antananarivo", "retail"),
    ("Tana Depot Gros",  "Analamanga",     "Antananarivo", "wholesale"),
    ("Tamatave Port",    "Atsinanana",     "Toamasina",    "wholesale"),
    ("Majunga Centre",   "Boeny",          "Mahajanga",    "retail"),
    ("Fianarantsoa Sud", "Haute Matsiatra","Fianarantsoa", "retail"),
    ("Diego Nord",       "Diana",          "Antsiranana",  "retail"),
]

# Profil hebdomadaire (lundi..dimanche) selon le canal
WEEKLY_RETAIL = np.array([0.85, 0.90, 0.95, 1.00, 1.20, 1.45, 1.05])
WEEKLY_WHOLE = np.array([1.25, 1.20, 1.15, 1.10, 1.05, 0.70, 0.45])

# Amplitude de la saisonnalité annuelle par catégorie (pic de mois)
# facteur = 1 + amp * cos(2pi (month - peak)/12)
SEASON_PROFILE = {
    "Alimentaire": (0.18, 12),   # pic décembre (fêtes)
    "Hygiene":     (0.08, 7),
    "Electronique":(0.25, 12),   # pic décembre (cadeaux)
    "Textile":     (0.20, 6),    # pic milieu d'année (hiver austral) + rentrée
    "Maison":      (0.12, 11),
}


def _madagascar_holidays() -> dict[str, str]:
    """Jours fériés malgaches 2022-2024 (fixes + mobiles hardcodés)."""
    fixed = {
        "01-01": "Nouvel An",
        "03-29": "Fete des Martyrs",
        "05-01": "Fete du Travail",
        "06-26": "Fete de l'Independance",
        "08-15": "Assomption",
        "11-01": "Toussaint",
        "12-25": "Noel",
    }
    holidays: dict[str, str] = {}
    for year in range(2022, 2025):
        for md, name in fixed.items():
            holidays[f"{year}-{md}"] = name
    # Fêtes mobiles (lundi de Pâques, Ascension, lundi de Pentecôte)
    movable = {
        "2022-04-18": "Lundi de Paques",
        "2022-05-26": "Ascension",
        "2022-06-06": "Lundi de Pentecote",
        "2023-04-10": "Lundi de Paques",
        "2023-05-18": "Ascension",
        "2023-05-29": "Lundi de Pentecote",
        "2024-04-01": "Lundi de Paques",
        "2024-05-09": "Ascension",
        "2024-05-20": "Lundi de Pentecote",
    }
    holidays.update(movable)
    return holidays


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
def make_dim_product(rng: np.random.Generator) -> pd.DataFrame:
    n = config.N_PRODUCTS
    cat_names = list(CATEGORIES.keys())
    rows = []
    for pid in range(1, n + 1):
        category = cat_names[(pid - 1) % len(cat_names)]
        cat_id = CATEGORIES[category]
        # Électronique et une partie du textile / maison sont importés
        if category == "Electronique":
            is_imported = 1
        elif category in ("Textile", "Maison"):
            is_imported = int(rng.random() < 0.6)
        elif category == "Hygiene":
            is_imported = int(rng.random() < 0.4)
        else:  # Alimentaire majoritairement local
            is_imported = int(rng.random() < 0.2)

        if is_imported:
            lead_time = int(rng.integers(20, 61))          # 20-60 j
            supplier = rng.choice(["China", "France", "India", "UAE"])
        else:
            lead_time = int(rng.integers(2, 8))            # 2-7 j
            supplier = "Madagascar"

        # coût unitaire (EUR) selon catégorie
        base_cost = {
            "Alimentaire": (0.5, 4.0),
            "Hygiene": (0.8, 6.0),
            "Electronique": (8.0, 120.0),
            "Textile": (2.0, 25.0),
            "Maison": (1.5, 40.0),
        }[category]
        unit_cost = round(float(rng.uniform(*base_cost)), 2)

        sku = f"{category[:3].upper()}-{pid:03d}"
        rows.append((pid, sku, category, cat_id, unit_cost,
                     is_imported, lead_time, str(supplier)))

    return pd.DataFrame(rows, columns=[
        "product_id", "sku", "category", "category_id", "unit_cost_eur",
        "is_imported", "lead_time_days", "supplier_country",
    ])


def make_dim_store() -> pd.DataFrame:
    rows = [(i + 1, name, region, city, channel)
            for i, (name, region, city, channel) in enumerate(STORES)]
    return pd.DataFrame(rows, columns=[
        "store_id", "name", "region", "city", "channel",
    ])


def make_dim_date() -> pd.DataFrame:
    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    holidays = _madagascar_holidays()
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    iso = dates.isocalendar()
    df["year"] = dates.year
    df["month"] = dates.month
    df["day"] = dates.day
    df["dow"] = dates.dayofweek                      # 0=lundi
    df["weekofyear"] = iso["week"].to_numpy()
    df["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    df["holiday_name"] = df["date"].map(holidays)
    df["is_holiday"] = df["holiday_name"].notna().astype(int)
    df["holiday_name"] = df["holiday_name"].where(df["holiday_name"].notna(), None)
    return df[["date", "year", "month", "day", "dow", "weekofyear",
               "is_weekend", "is_holiday", "holiday_name"]]


def make_dim_fx(dim_date: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dim_date)
    t = np.arange(n)
    start_rate, end_rate = 4300.0, 5000.0
    trend = start_rate + (end_rate - start_rate) * (t / (n - 1))
    seasonal = 55.0 * np.sin(2 * np.pi * t / 365.25)          # saisonnalité douce
    noise = rng.normal(0, 12, n).cumsum() * 0.15              # marche aléatoire douce
    noise = noise - noise.mean()
    mga = trend + seasonal + noise
    return pd.DataFrame({
        "date": dim_date["date"].to_numpy(),
        "mga_per_eur": np.round(mga, 1),
    })


# ---------------------------------------------------------------------------
# Faits de ventes
# ---------------------------------------------------------------------------
def make_fct_sales(dim_product: pd.DataFrame, dim_store: pd.DataFrame,
                   dim_date: pd.DataFrame, dim_fx: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.to_datetime(dim_date["date"])
    n_days = len(dates)
    dow = dim_date["dow"].to_numpy()
    month = dim_date["month"].to_numpy()
    is_holiday = dim_date["is_holiday"].to_numpy().astype(bool)
    t = np.arange(n_days)
    trend_global = 1.0 + 0.20 * (t / (n_days - 1))            # +20% sur 3 ans
    fx = dim_fx["mga_per_eur"].to_numpy()
    fx_ref = fx.mean()

    # attributs produit
    prod_base = {}       # niveau de demande de base par produit
    for _, r in dim_product.iterrows():
        # base plus élevé pour l'alimentaire/hygiène (rotation rapide)
        scale = {"Alimentaire": 22, "Hygiene": 16, "Electronique": 3,
                 "Textile": 6, "Maison": 5}[r["category"]]
        prod_base[r["product_id"]] = float(rng.uniform(0.6, 1.4)) * scale

    # multiplicateur magasin (wholesale = plus gros volumes)
    store_mult = {}
    for _, s in dim_store.iterrows():
        base = 1.9 if s["channel"] == "wholesale" else 1.0
        store_mult[s["store_id"]] = base * float(rng.uniform(0.8, 1.25))

    frames = []
    for _, prod in dim_product.iterrows():
        pid = prod["product_id"]
        category = prod["category"]
        imported = bool(prod["is_imported"])
        unit_cost = prod["unit_cost_eur"]

        # saisonnalité annuelle propre à la catégorie
        amp, peak = SEASON_PROFILE[category]
        annual = 1.0 + amp * np.cos(2 * np.pi * (month - peak) / 12.0)

        # effet férié (bump renforcé en décembre pour alim/électro)
        holiday_factor = np.ones(n_days)
        holiday_factor[is_holiday] = 1.30
        if category in ("Alimentaire", "Electronique"):
            dec_mask = (month == 12)
            holiday_factor[dec_mask] *= 1.10

        # sensibilité négative au FX pour les importés (répercussion coût)
        if imported:
            fx_effect = (fx_ref / fx) ** 0.35
        else:
            fx_effect = np.ones(n_days)

        # prix de vente (MGA) : coût EUR * marge * FX
        markup = 1.6 if category in ("Alimentaire", "Hygiene") else 2.1
        price_eur = unit_cost * markup
        base_price_mga = price_eur * fx                     # varie avec le FX

        for _, st in dim_store.iterrows():
            sid = st["store_id"]
            channel = st["channel"]
            weekly = WEEKLY_WHOLE if channel == "wholesale" else WEEKLY_RETAIL
            weekly_factor = weekly[dow]

            # promotions : ~4% des jours, par série produit/magasin
            promo_flag = (rng.random(n_days) < 0.045).astype(int)
            promo_mult = np.where(promo_flag == 1,
                                  rng.uniform(1.5, 2.5, n_days), 1.0)

            lam = (prod_base[pid] * store_mult[sid]
                   * weekly_factor * annual * trend_global
                   * holiday_factor * fx_effect * promo_mult)
            lam = np.clip(lam, 0.05, None)
            units = rng.poisson(lam)

            # prix : baisse ~20% en promo, léger bruit multiplicatif
            price = base_price_mga * np.where(promo_flag == 1, 0.80, 1.0)
            price = np.round(price * rng.uniform(0.98, 1.02, n_days), 0)
            revenue = np.round(units * price, 0)

            frames.append(pd.DataFrame({
                "date": dim_date["date"].to_numpy(),
                "store_id": sid,
                "product_id": pid,
                "units_sold": units.astype(int),
                "price_mga": price,
                "promo_flag": promo_flag,
                "revenue_mga": revenue,
            }))

    sales = pd.concat(frames, ignore_index=True)
    sales = sales.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)
    return sales


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all() -> None:
    rng = np.random.default_rng(config.SEED)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    dim_product = make_dim_product(rng)
    dim_store = make_dim_store()
    dim_date = make_dim_date()
    dim_fx = make_dim_fx(dim_date, rng)
    fct_sales = make_fct_sales(dim_product, dim_store, dim_date, dim_fx, rng)

    dim_product.to_csv(config.DATA_DIR / "dim_product.csv", index=False)
    dim_store.to_csv(config.DATA_DIR / "dim_store.csv", index=False)
    dim_date.to_csv(config.DATA_DIR / "dim_date.csv", index=False)
    dim_fx.to_csv(config.DATA_DIR / "dim_fx.csv", index=False)
    fct_sales.to_csv(config.DATA_DIR / "fct_sales.csv", index=False)

    print("dim_product :", dim_product.shape)
    print("dim_store   :", dim_store.shape)
    print("dim_date    :", dim_date.shape)
    print("dim_fx      :", dim_fx.shape,
          f"(MGA/EUR {dim_fx['mga_per_eur'].iloc[0]:.0f} -> {dim_fx['mga_per_eur'].iloc[-1]:.0f})")
    print("fct_sales   :", fct_sales.shape,
          f"(total units={int(fct_sales['units_sold'].sum()):,})")
    print("CSV ecrits dans", config.DATA_DIR)


if __name__ == "__main__":
    generate_all()
