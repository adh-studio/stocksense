-- StockSense — Feature engineering -> table ml_features
-- Une ligne = (date, store_id, product_id). Voir config.FEATURE_COLUMNS.
--
-- Anti-fuite temporelle :
--   * lag_N        = valeur d'il y a N jours (LAG), jamais le jour courant.
--   * roll_mean/std_7/28 = agrégats sur les jours STRICTEMENT ANTERIEURS
--                    (ROWS BETWEEN N PRECEDING AND 1 PRECEDING).
--   * roll_mean_7_lag7 = moyenne 7 j se terminant il y a 7 jours
--                    (ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING).
-- Les fenêtres ROWS supposent des lignes journalières contiguës par série,
-- ce que garantit la génération (couverture pleine produit x magasin x jour).
--
-- roll_std_7 : SQLite n'a pas STDDEV. On calcule la variance de population
--   E[x^2] - E[x]^2 sur la fenêtre puis sqrt (fonction sqrt garantie par
--   build_database.py qui l'enregistre si absente du build SQLite).
-- days_since_promo : nb de jours depuis la dernière promo (0 si promo le jour
--   même). Le promo_flag étant une décision planifiée connue, l'inclusion du
--   jour courant n'introduit pas de fuite. 999 tant qu'aucune promo observée.

DROP TABLE IF EXISTS ml_features;

CREATE TABLE ml_features AS
WITH base AS (
    SELECT
        s.date,
        s.store_id,
        s.product_id,
        s.units_sold,
        s.price_mga,
        s.promo_flag,
        d.is_holiday,
        d.is_weekend,
        d.dow,
        d.month,
        d.weekofyear,
        fx.mga_per_eur,
        p.unit_cost_eur,
        p.is_imported,
        p.lead_time_days,
        p.category_id
    FROM fct_sales s
    JOIN dim_date    d  ON s.date       = d.date
    JOIN dim_product p  ON s.product_id = p.product_id
    JOIN dim_fx      fx ON s.date       = fx.date
),
feat AS (
    SELECT
        base.*,
        LAG(units_sold, 1)  OVER w AS lag_1,
        LAG(units_sold, 7)  OVER w AS lag_7,
        LAG(units_sold, 14) OVER w AS lag_14,
        LAG(units_sold, 28) OVER w AS lag_28,
        AVG(units_sold)                 OVER w7     AS roll_mean_7,
        AVG(units_sold)                 OVER w28    AS roll_mean_28,
        AVG(units_sold)                 OVER w7lag7 AS roll_mean_7_lag7,
        AVG(units_sold * units_sold)    OVER w7     AS m2_7,
        AVG(units_sold)                 OVER w7     AS m1_7,
        MAX(CASE WHEN promo_flag = 1 THEN date END) OVER wall AS last_promo_date
    FROM base
    WINDOW
        w      AS (PARTITION BY product_id, store_id ORDER BY date),
        w7     AS (PARTITION BY product_id, store_id ORDER BY date
                   ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING),
        w28    AS (PARTITION BY product_id, store_id ORDER BY date
                   ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING),
        w7lag7 AS (PARTITION BY product_id, store_id ORDER BY date
                   ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING),
        wall   AS (PARTITION BY product_id, store_id ORDER BY date
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT
    date,
    store_id,
    product_id,
    units_sold,
    -- features "jour courant" (connues au moment de la planification)
    price_mga,
    promo_flag,
    is_holiday,
    is_weekend,
    dow,
    month,
    weekofyear,
    mga_per_eur,
    unit_cost_eur,
    is_imported,
    lead_time_days,
    category_id,
    -- lags / rolling (jours antérieurs uniquement)
    lag_1,
    lag_7,
    lag_14,
    lag_28,
    roll_mean_7,
    roll_mean_28,
    sqrt(MAX(m2_7 - m1_7 * m1_7, 0.0)) AS roll_std_7,
    roll_mean_7_lag7,
    CASE
        WHEN last_promo_date IS NULL THEN 999
        ELSE CAST(julianday(date) - julianday(last_promo_date) AS INTEGER)
    END AS days_since_promo
FROM feat;
