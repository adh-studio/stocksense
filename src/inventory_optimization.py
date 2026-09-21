"""Agent E — Optimisation des stocks (politique R,Q pilotée par la prévision).

Transforme la prévision probabiliste de la demande (P10/P50/P90) en décisions
de réapprovisionnement chiffrées, puis simule et compare deux politiques de
point de commande (R,Q) sur la période de test :

  * Politique "modèle"   : point de commande + stock de sécurité dérivés de
    l'intervalle prédictif du modèle LSTM (bien calibré). L'incertitude propre à
    CHAQUE couple (produit, magasin) pilote le stock de sécurité.
  * Politique "baseline" : point de commande basé sur une prévision naïve
    (seasonal_naive) et une "règle du pouce" typique de la pratique — stock de
    sécurité dimensionné PROPORTIONNELLEMENT au délai de livraison (∝ L) avec un
    coefficient de variation forfaitaire. C'est l'erreur classique : ignorer que
    l'écart-type de la demande cumulée croît en sqrt(L), ce qui sur-stocke
    massivement les articles à long délai (les IMPORTS).

On mesure, à niveau de service cible équivalent (90 / 95 / 98 %) :
  - le taux de service réalisé (fill rate),
  - le nombre de ruptures,
  - le stock moyen immobilisé (unités et valeur MGA),
  - le coût total (détention + rupture + commande).

Enfin on quantifie la sensibilité du coût des stocks IMPORTÉS à une hausse du
taux de change MGA/EUR (contexte importateur Madagascar).

Exécution :  ./.venv/bin/python -m src.inventory_optimization
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # backend non interactif (obligatoire hors écran)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# HYPOTHÈSES DE COÛTS (documentées, ajustables)
# ---------------------------------------------------------------------------
# Toutes les valeurs monétaires internes sont en Ariary malgache (MGA).
# Le coût unitaire produit est fourni en EUR (dim_product.unit_cost_eur) et
# converti en MGA au taux de change (dim_fx.mga_per_eur).

HOLDING_RATE_ANNUAL = 0.25       # coût de détention = 25 %/an de la valeur stock
                                 # (capital immobilisé + stockage + obsolescence)
ORDER_COST_EUR = 10.0            # coût fixe par commande passée (administratif,
                                 # transport, dédouanement) — exprimé en EUR.
                                 # Modéré : le réappro reste fréquent, si bien
                                 # que le stock de sécurité (et non le stock de
                                 # cycle) constitue la protection déterminante.
SHORTAGE_PENALTY_FACTOR = 1.0    # coût de rupture par unité non servie ≈ 1 ×
                                 # coût unitaire (marge brute perdue + perte de
                                 # clientèle / substitution, ordre de grandeur
                                 # d'un coût unitaire pour un distributeur)
CV_FLAT_BASELINE = 0.50          # coefficient de variation forfaitaire de la
                                 # baseline (règle du pouce : même dispersion
                                 # supposée pour tous les articles, faute de
                                 # modèle probabiliste par SKU)
DAYS_PER_YEAR = 365

# Niveaux de service cible testés -> quantile z de la loi normale standard.
# (SS = z * sigma_jour * sqrt(L))
SERVICE_LEVELS = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.98: 2.0537,
}
# z associé au P90 de l'intervalle prédictif (sigma = (P90 - P50) / Z_P90).
Z_P90 = 1.2816

# Scénarios de sensibilité au taux de change (multiplicateur du MGA/EUR).
FX_SCENARIOS = {
    "base": 1.00,
    "+10%": 1.10,
    "+20%": 1.20,
}


@dataclass
class SimResult:
    fill_rate: float
    n_stockouts: int          # nb de jours en rupture (demande non servie)
    units_short: float        # total d'unités non servies
    avg_stock_units: float
    avg_stock_value_mga: float
    holding_cost: float
    shortage_cost: float
    order_cost: float
    n_orders: int
    total_cost: float
    on_hand_series: np.ndarray
    receipt_days: list
    stockout_days: list


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
def load_products_fx() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(config.DB_PATH)
    try:
        prod = pd.read_sql_query(
            "SELECT product_id, unit_cost_eur, is_imported, lead_time_days, "
            "category, supplier_country FROM dim_product",
            con,
        )
        fx = pd.read_sql_query("SELECT date, mga_per_eur FROM dim_fx", con)
    finally:
        con.close()
    fx["date"] = pd.to_datetime(fx["date"])
    return prod, fx


def load_preds(name: str) -> pd.DataFrame:
    df = pd.read_parquet(config.REPORTS_DIR / f"preds_{name}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paramètres de la politique (R,Q) par couple (produit, magasin)
# ---------------------------------------------------------------------------
def compute_policy_params(
    mean_daily: float,
    sigma_daily: float,
    lead_time: int,
    z: float,
    unit_cost_mga: float,
    order_cost_mga: float,
    lead_time_exponent: float = 0.5,
) -> tuple[float, float, float]:
    """Retourne (ROP, EOQ, safety_stock).

    ROP = demande moyenne sur le délai L + stock de sécurité.
    SS  = z * sigma_jour * L**exposant.

    Le paramètre ``lead_time_exponent`` matérialise la différence FONDAMENTALE
    entre les deux politiques :
      * modèle   -> 0.5  : la variance sur le délai s'additionne, l'écart-type de
        la demande cumulée croît en **sqrt(L)** (formule statistiquement correcte).
      * baseline -> 1.0  : "règle du pouce" qui dimensionne le stock de sécurité
        **proportionnellement au délai L**. C'est l'erreur la plus répandue en
        pratique : elle sur-dimensionne massivement les articles à long délai
        (typiquement les IMPORTS), et donc immobilise un capital considérable.

    EOQ = sqrt(2 * D_annuel * coût_commande / coût_détention_unitaire).
    """
    lead_time = max(int(lead_time), 1)
    mean_daily = max(mean_daily, 0.0)
    sigma_daily = max(sigma_daily, 0.0)

    lead_demand = mean_daily * lead_time
    safety_stock = z * sigma_daily * (lead_time ** lead_time_exponent)
    rop = lead_demand + safety_stock

    d_annual = mean_daily * DAYS_PER_YEAR
    holding_per_unit_year = HOLDING_RATE_ANNUAL * unit_cost_mga
    if holding_per_unit_year > 0 and d_annual > 0:
        eoq = np.sqrt(2.0 * d_annual * order_cost_mga / holding_per_unit_year)
    else:
        eoq = mean_daily * lead_time
    # borne basse : au moins une semaine de demande, jamais nul
    eoq = max(eoq, mean_daily * 7.0, 1.0)
    return rop, eoq, safety_stock


# ---------------------------------------------------------------------------
# Simulation jour par jour d'une politique (R,Q) sur une série
# ---------------------------------------------------------------------------
def simulate_series(
    demand: np.ndarray,
    rop: float,
    eoq: float,
    lead_time: int,
    unit_cost_mga: float,
    order_cost_mga: float,
) -> SimResult:
    lead_time = max(int(lead_time), 1)
    n = len(demand)
    # Warm start neutre au point de commande (identique aux 2 politiques) : on
    # démarre au ROP, ce qui expose immédiatement la première fenêtre de délai
    # et laisse le stock de sécurité jouer son rôle protecteur.
    on_hand = rop
    pipeline: dict[int, float] = {}  # jour d'arrivée -> quantité en transit

    on_hand_series = np.empty(n, dtype=float)
    receipt_days: list[int] = []
    stockout_days: list[int] = []
    units_short = 0.0
    total_demand = 0.0
    n_orders = 0

    daily_holding = HOLDING_RATE_ANNUAL / DAYS_PER_YEAR * unit_cost_mga

    for t in range(n):
        # 1) réception des commandes arrivant aujourd'hui
        if t in pipeline:
            on_hand += pipeline.pop(t)
            receipt_days.append(t)

        # 2) service de la demande réelle du jour
        d = float(demand[t])
        total_demand += d
        served = min(on_hand, d)
        on_hand -= served
        short = d - served
        if short > 1e-9:
            units_short += short
            stockout_days.append(t)

        on_hand_series[t] = on_hand  # stock détenu en fin de journée

        # 3) déclenchement d'une commande si position de stock <= ROP
        on_order = sum(pipeline.values())
        inventory_position = on_hand + on_order
        if inventory_position <= rop:
            arrival = t + lead_time
            pipeline[arrival] = pipeline.get(arrival, 0.0) + eoq
            n_orders += 1

    fill_rate = 1.0 - units_short / total_demand if total_demand > 0 else 1.0
    avg_stock_units = float(on_hand_series.mean())
    avg_stock_value = avg_stock_units * unit_cost_mga
    holding_cost = float(on_hand_series.sum()) * daily_holding
    shortage_cost = units_short * SHORTAGE_PENALTY_FACTOR * unit_cost_mga
    order_cost = n_orders * order_cost_mga
    total_cost = holding_cost + shortage_cost + order_cost

    return SimResult(
        fill_rate=fill_rate,
        n_stockouts=len(stockout_days),
        units_short=units_short,
        avg_stock_units=avg_stock_units,
        avg_stock_value_mga=avg_stock_value,
        holding_cost=holding_cost,
        shortage_cost=shortage_cost,
        order_cost=order_cost,
        n_orders=n_orders,
        total_cost=total_cost,
        on_hand_series=on_hand_series,
        receipt_days=receipt_days,
        stockout_days=stockout_days,
    )


# ---------------------------------------------------------------------------
# Estimation demande / incertitude par série et par politique
# ---------------------------------------------------------------------------
def build_series_stats(preds_model: pd.DataFrame, preds_base: pd.DataFrame) -> pd.DataFrame:
    """Agrège par (product_id, store_id) : demande moyenne prévue et sigma jour.

    - modèle   : mean(P50 LSTM), sigma = mean((P90-P50)/Z_P90) (incertitude
      prédictive propre à la série).
    - baseline : mean(P50 seasonal_naive), sigma = CV_FLAT * mean_daily
      (règle du pouce forfaitaire, sans intervalle prédictif).
    """
    g_model = preds_model.groupby(["product_id", "store_id"])
    model_stats = g_model.apply(
        lambda d: pd.Series(
            {
                "mean_daily_model": d["y_pred"].mean(),
                "sigma_daily_model": np.maximum(
                    (d["y_p90"] - d["y_pred"]) / Z_P90, 0.0
                ).mean(),
            }
        ),
        include_groups=False,
    )

    g_base = preds_base.groupby(["product_id", "store_id"])
    base_stats = g_base.apply(
        lambda d: pd.Series({"mean_daily_base": d["y_pred"].mean()}),
        include_groups=False,
    )
    base_stats["sigma_daily_base"] = CV_FLAT_BASELINE * base_stats["mean_daily_base"]

    stats = model_stats.join(base_stats).reset_index()
    return stats


# ---------------------------------------------------------------------------
# Boucle principale de simulation
# ---------------------------------------------------------------------------
def run_simulations(
    preds_model: pd.DataFrame,
    stats: pd.DataFrame,
    prod: pd.DataFrame,
    fx_mean: float,
) -> tuple[pd.DataFrame, dict]:
    """Simule les 2 politiques × 3 niveaux de service sur toutes les séries."""
    order_cost_mga = ORDER_COST_EUR * fx_mean
    prod_idx = prod.set_index("product_id")

    # demande réelle (y_true) par série, ordonnée par date
    demand_map = {
        (pid, sid): d.sort_values("date")["y_true"].to_numpy(dtype=float)
        for (pid, sid), d in preds_model.groupby(["product_id", "store_id"])
    }

    rows = []
    trajectories: dict = {}  # (policy, service, pid, sid) -> SimResult (échantillon)

    for service, z in SERVICE_LEVELS.items():
        for _, r in stats.iterrows():
            pid = int(r["product_id"])
            sid = int(r["store_id"])
            pinfo = prod_idx.loc[pid]
            lead_time = int(pinfo["lead_time_days"])
            unit_cost_mga = float(pinfo["unit_cost_eur"]) * fx_mean
            is_imported = int(pinfo["is_imported"])
            demand = demand_map[(pid, sid)]

            for policy in ("baseline", "model"):
                if policy == "model":
                    mean_daily = r["mean_daily_model"]
                    sigma_daily = r["sigma_daily_model"]
                    lt_exp = 0.5          # sqrt(L) — statistiquement correct
                else:
                    mean_daily = r["mean_daily_base"]
                    sigma_daily = r["sigma_daily_base"]
                    lt_exp = 1.0          # proportionnel à L — règle du pouce

                rop, eoq, ss = compute_policy_params(
                    mean_daily, sigma_daily, lead_time, z,
                    unit_cost_mga, order_cost_mga, lead_time_exponent=lt_exp,
                )
                sim = simulate_series(
                    demand, rop, eoq, lead_time, unit_cost_mga, order_cost_mga
                )
                rows.append(
                    {
                        "policy": policy,
                        "service_target": service,
                        "product_id": pid,
                        "store_id": sid,
                        "is_imported": is_imported,
                        "lead_time_days": lead_time,
                        "unit_cost_mga": unit_cost_mga,
                        "rop": rop,
                        "eoq": eoq,
                        "safety_stock": ss,
                        "fill_rate": sim.fill_rate,
                        "n_stockouts": sim.n_stockouts,
                        "units_short": sim.units_short,
                        "avg_stock_units": sim.avg_stock_units,
                        "avg_stock_value_mga": sim.avg_stock_value_mga,
                        "holding_cost": sim.holding_cost,
                        "shortage_cost": sim.shortage_cost,
                        "order_cost": sim.order_cost,
                        "n_orders": sim.n_orders,
                        "total_cost": sim.total_cost,
                        "total_demand": float(demand.sum()),
                    }
                )
                trajectories[(policy, service, pid, sid)] = sim

    detail = pd.DataFrame(rows)
    return detail, trajectories


def aggregate_kpis(detail: pd.DataFrame) -> pd.DataFrame:
    """KPI agrégés par (policy, service_target). Fill rate pondéré par la demande."""
    grp = detail.groupby(["policy", "service_target"])
    agg = grp.apply(
        lambda d: pd.Series(
            {
                "fill_rate": 1.0 - d["units_short"].sum() / d["total_demand"].sum(),
                "n_stockouts": d["n_stockouts"].sum(),
                "units_short": d["units_short"].sum(),
                "avg_stock_units": d["avg_stock_units"].mean(),
                "avg_stock_value_mga": d["avg_stock_value_mga"].sum(),
                "holding_cost": d["holding_cost"].sum(),
                "shortage_cost": d["shortage_cost"].sum(),
                "order_cost": d["order_cost"].sum(),
                "total_cost": d["total_cost"].sum(),
                "n_orders": d["n_orders"].sum(),
                "n_series": len(d),
            }
        ),
        include_groups=False,
    ).reset_index()
    return agg


# ---------------------------------------------------------------------------
# Sensibilité au taux de change (stocks importés)
# ---------------------------------------------------------------------------
def fx_sensitivity(detail: pd.DataFrame, service_ref: float) -> pd.DataFrame:
    """Recalcule coût de détention & capital immobilisé des stocks importés
    sous plusieurs scénarios de hausse du MGA/EUR, pour la politique modèle.

    Le coût de détention et la valeur du stock sont proportionnels au coût
    unitaire en MGA, donc à un multiplicateur du taux de change appliqué aux
    seuls articles importés (is_imported = 1).
    """
    sub = detail[
        (detail["policy"] == "model")
        & (detail["service_target"] == service_ref)
    ].copy()
    imported = sub[sub["is_imported"] == 1]
    local = sub[sub["is_imported"] == 0]

    rows = []
    for name, mult in FX_SCENARIOS.items():
        holding_imported = imported["holding_cost"].sum() * mult
        holding_local = local["holding_cost"].sum()
        stock_value_imported = imported["avg_stock_value_mga"].sum() * mult
        rows.append(
            {
                "scenario": name,
                "fx_multiplier": mult,
                "holding_cost_imported_mga": holding_imported,
                "holding_cost_local_mga": holding_local,
                "holding_cost_total_mga": holding_imported + holding_local,
                "stock_value_imported_mga": stock_value_imported,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_service_vs_cost(agg: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"baseline": "#d1495b", "model": "#2e86ab"}
    for policy in ("baseline", "model"):
        d = agg[agg["policy"] == policy].sort_values("total_cost")
        ax.plot(
            d["total_cost"] / 1e6,
            d["fill_rate"] * 100,
            "o-",
            color=colors[policy],
            label=f"Politique {policy}",
            markersize=9,
            linewidth=2,
        )
        for _, r in d.iterrows():
            ax.annotate(
                f"{int(r['service_target']*100)}%",
                (r["total_cost"] / 1e6, r["fill_rate"] * 100),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
    ax.set_xlabel("Coût total simulé (millions MGA)")
    ax.set_ylabel("Taux de service réalisé (fill rate, %)")
    ax.set_title(
        "Frontière service / coût\nprévision probabiliste (modèle) vs baseline forfaitaire"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_stock_trajectory(sim: SimResult, rop: float, meta: dict, path):
    n = len(sim.on_hand_series)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(n), sim.on_hand_series, color="#2e86ab", lw=1.8, label="Stock détenu")
    ax.axhline(rop, color="#f0a202", ls="--", lw=1.5, label=f"Point de commande (ROP={rop:.0f})")
    ax.axhline(0, color="grey", lw=0.8)
    if sim.receipt_days:
        ax.scatter(
            sim.receipt_days,
            sim.on_hand_series[sim.receipt_days],
            marker="^", color="#2a9d8f", s=90, zorder=5, label="Réceptions (EOQ)",
        )
    if sim.stockout_days:
        ax.scatter(
            sim.stockout_days,
            np.zeros(len(sim.stockout_days)),
            marker="x", color="#d1495b", s=70, zorder=5, label="Jours de rupture",
        )
    ax.set_xlabel("Jour (période de test)")
    ax.set_ylabel("Unités en stock")
    ax.set_title(
        f"Trajectoire de stock — produit {meta['product_id']} / magasin {meta['store_id']} "
        f"({meta['category']}, L={meta['lead_time']}j, importé={meta['is_imported']})\n"
        f"Politique modèle @ service {int(meta['service']*100)}% — "
        f"fill rate {sim.fill_rate*100:.1f}%"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_fx(fx_df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(fx_df))
    w = 0.42
    ax.bar(
        x - w / 2,
        fx_df["holding_cost_imported_mga"] / 1e6,
        w, label="Détention stock importé", color="#d1495b",
    )
    ax.bar(
        x + w / 2,
        fx_df["holding_cost_local_mga"] / 1e6,
        w, label="Détention stock local", color="#2a9d8f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(×{m:.2f})" for s, m in zip(fx_df["scenario"], fx_df["fx_multiplier"])])
    ax.set_ylabel("Coût de détention (millions MGA)")
    ax.set_title(
        "Sensibilité au taux de change MGA/EUR\ncoût de détention des stocks importés vs locaux"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    # annotation du surcoût import
    base = fx_df.iloc[0]["holding_cost_imported_mga"]
    for i, r in fx_df.iterrows():
        delta = (r["holding_cost_imported_mga"] - base) / base * 100 if base else 0
        if i > 0:
            ax.annotate(
                f"+{delta:.0f}%",
                (i - w / 2, r["holding_cost_imported_mga"] / 1e6),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=9, color="#d1495b", fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    np.random.seed(config.SEED)

    print("→ Chargement des prévisions et référentiels...")
    preds_model = load_preds("lstm")            # meilleur modèle (bien calibré)
    preds_base = load_preds("seasonal_naive")   # prévision naïve (baseline)
    prod, fx = load_products_fx()

    # FX moyen sur la période de test (référence de conversion EUR -> MGA)
    test_start = pd.Timestamp(config.TEST_START)
    fx_test = fx[fx["date"] >= test_start]
    fx_mean = float(fx_test["mga_per_eur"].mean())
    print(f"  FX moyen période test : {fx_mean:.0f} MGA/EUR")

    print("→ Estimation demande / incertitude par (produit, magasin)...")
    stats = build_series_stats(preds_model, preds_base)
    print(f"  {len(stats)} séries (produit × magasin)")

    print("→ Simulation des politiques (R,Q) × niveaux de service...")
    detail, trajectories = run_simulations(preds_model, stats, prod, fx_mean)

    agg = aggregate_kpis(detail)

    # --- Sauvegarde CSV KPI agrégés ---
    csv_path = config.REPORTS_DIR / "inventory_sim.csv"
    agg_out = agg.copy()
    agg_out["fill_rate_pct"] = agg_out["fill_rate"] * 100
    agg_out.to_csv(csv_path, index=False)
    print(f"✓ {csv_path}")

    # --- Synthèse du gain modèle vs baseline (à service cible équivalent) ---
    summary = {"assumptions": {
        "holding_rate_annual": HOLDING_RATE_ANNUAL,
        "order_cost_eur": ORDER_COST_EUR,
        "shortage_penalty_factor": SHORTAGE_PENALTY_FACTOR,
        "cv_flat_baseline": CV_FLAT_BASELINE,
        "fx_mean_mga_per_eur": fx_mean,
        "service_levels": list(SERVICE_LEVELS.keys()),
    }, "by_service_level": {}}

    for service in SERVICE_LEVELS:
        b = agg[(agg["policy"] == "baseline") & (agg["service_target"] == service)].iloc[0]
        m = agg[(agg["policy"] == "model") & (agg["service_target"] == service)].iloc[0]
        cost_reduction = (b["total_cost"] - m["total_cost"]) / b["total_cost"] * 100
        stock_reduction = (b["avg_stock_value_mga"] - m["avg_stock_value_mga"]) / b["avg_stock_value_mga"] * 100
        summary["by_service_level"][f"{service:.2f}"] = {
            "baseline_fill_rate": round(float(b["fill_rate"]), 4),
            "model_fill_rate": round(float(m["fill_rate"]), 4),
            "fill_rate_gain_pts": round(float((m["fill_rate"] - b["fill_rate"]) * 100), 2),
            "baseline_total_cost_mga": round(float(b["total_cost"]), 0),
            "model_total_cost_mga": round(float(m["total_cost"]), 0),
            "cost_reduction_pct": round(float(cost_reduction), 2),
            "baseline_avg_stock_value_mga": round(float(b["avg_stock_value_mga"]), 0),
            "model_avg_stock_value_mga": round(float(m["avg_stock_value_mga"]), 0),
            "stock_value_reduction_pct": round(float(stock_reduction), 2),
            "baseline_stockouts": int(b["n_stockouts"]),
            "model_stockouts": int(m["n_stockouts"]),
        }

    # --- Sensibilité FX (au service de référence 95 %) ---
    service_ref = 0.95
    fx_df = fx_sensitivity(detail, service_ref)
    row_base = fx_df.iloc[0]
    row_10 = fx_df[fx_df["scenario"] == "+10%"].iloc[0]
    base_imp = float(row_base["holding_cost_imported_mga"])
    fx10 = float(row_10["holding_cost_imported_mga"])
    base_total = float(row_base["holding_cost_total_mga"])
    fx10_total = float(row_10["holding_cost_total_mga"])
    imported_share = base_imp / base_total if base_total else 0.0
    summary["fx_sensitivity"] = {
        "service_ref": service_ref,
        "imported_share_of_holding_pct": round(imported_share * 100, 1),
        "holding_cost_imported_base_mga": round(base_imp, 0),
        "holding_cost_imported_plus10_mga": round(fx10, 0),
        "extra_holding_cost_imported_plus10_mga": round(fx10 - base_imp, 0),
        "extra_holding_cost_imported_plus10_pct": round((fx10 - base_imp) / base_imp * 100, 2),
        "holding_cost_total_base_mga": round(base_total, 0),
        "holding_cost_total_plus10_mga": round(fx10_total, 0),
        "extra_holding_cost_total_plus10_pct": round((fx10_total - base_total) / base_total * 100, 2),
        "note": "Une hausse de 10% du MGA/EUR renchérit de 10% le coût de "
                "détention des seuls articles importés ; comme ceux-ci pèsent "
                f"{imported_share*100:.0f}% du coût de détention total, le coût "
                "d'immobilisation global du stock progresse de "
                f"{(fx10_total - base_total) / base_total * 100:.1f}%. Risque de "
                "change concentré sur le portefeuille importé.",
    }

    json_path = config.REPORTS_DIR / "inventory_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ {json_path}")

    # --- Figures ---
    fig_service_vs_cost(agg, config.FIGURES_DIR / "fig_service_vs_cost.png")
    print(f"✓ {config.FIGURES_DIR / 'fig_service_vs_cost.png'}")

    # choix d'un article représentatif : importé, forte demande (histoire riche)
    prod_idx = prod.set_index("product_id")
    demand_by_series = (
        detail[(detail["policy"] == "model") & (detail["service_target"] == service_ref)]
        .assign(imp=lambda d: d["is_imported"])
    )
    cand = demand_by_series[demand_by_series["imp"] == 1].sort_values(
        "total_demand", ascending=False
    ).iloc[0]
    pid, sid = int(cand["product_id"]), int(cand["store_id"])
    sim_repr = trajectories[("model", service_ref, pid, sid)]
    meta = {
        "product_id": pid, "store_id": sid,
        "category": prod_idx.loc[pid, "category"],
        "lead_time": int(prod_idx.loc[pid, "lead_time_days"]),
        "is_imported": int(prod_idx.loc[pid, "is_imported"]),
        "service": service_ref,
    }
    fig_stock_trajectory(
        sim_repr, float(cand["rop"]), meta,
        config.FIGURES_DIR / "fig_stock_trajectory.png",
    )
    print(f"✓ {config.FIGURES_DIR / 'fig_stock_trajectory.png'}")

    fig_fx(fx_df, config.FIGURES_DIR / "fig_fx_sensitivity.png")
    print(f"✓ {config.FIGURES_DIR / 'fig_fx_sensitivity.png'}")

    # --- Synthèse chiffrée à l'écran ---
    print("\n" + "=" * 72)
    print("SYNTHÈSE — Optimisation des stocks (modèle probabiliste vs baseline)")
    print("=" * 72)
    print(f"{'Service':>8} | {'Fill rate base':>14} | {'Fill rate modèle':>16} | "
          f"{'Δ pts':>6} | {'Gain coût':>10} | {'Δ stock €':>10}")
    print("-" * 72)
    for service in SERVICE_LEVELS:
        s = summary["by_service_level"][f"{service:.2f}"]
        print(f"{int(service*100):>7}% | {s['baseline_fill_rate']*100:>13.1f}% | "
              f"{s['model_fill_rate']*100:>15.1f}% | {s['fill_rate_gain_pts']:>6.1f} | "
              f"{s['cost_reduction_pct']:>9.1f}% | {s['stock_value_reduction_pct']:>9.1f}%")
    print("-" * 72)
    fxs = summary["fx_sensitivity"]
    print(f"Sensibilité FX (+10% MGA/EUR) : surcoût détention stocks importés "
          f"= +{fxs['extra_holding_cost_imported_plus10_mga']/1e6:.1f} M MGA "
          f"(+{fxs['extra_holding_cost_imported_plus10_pct']:.0f}%) ; "
          f"imports = {fxs['imported_share_of_holding_pct']:.0f}% du stock "
          f"-> coût de détention TOTAL +{fxs['extra_holding_cost_total_plus10_pct']:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
