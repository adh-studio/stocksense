-- StockSense — DDL du star schema (SQLite)
-- Voir COORDINATION.md §"Schéma de la base".
-- Ce script est idempotent : il supprime puis recrée toutes les tables.

DROP TABLE IF EXISTS fct_sales;
DROP TABLE IF EXISTS dim_fx;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_store;
DROP TABLE IF EXISTS dim_product;

-- Dimension produit ---------------------------------------------------------
CREATE TABLE dim_product (
    product_id       INTEGER PRIMARY KEY,
    sku              TEXT    NOT NULL,
    category         TEXT    NOT NULL,
    category_id      INTEGER NOT NULL,
    unit_cost_eur    REAL    NOT NULL,
    is_imported      INTEGER NOT NULL,     -- 0/1
    lead_time_days   INTEGER NOT NULL,
    supplier_country TEXT    NOT NULL
);

-- Dimension magasin ---------------------------------------------------------
CREATE TABLE dim_store (
    store_id INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    region   TEXT NOT NULL,
    city     TEXT NOT NULL,
    channel  TEXT NOT NULL              -- retail / wholesale
);

-- Dimension date ------------------------------------------------------------
CREATE TABLE dim_date (
    date         TEXT PRIMARY KEY,       -- 'YYYY-MM-DD'
    year         INTEGER NOT NULL,
    month        INTEGER NOT NULL,
    day          INTEGER NOT NULL,
    dow          INTEGER NOT NULL,       -- 0=lundi ... 6=dimanche
    weekofyear   INTEGER NOT NULL,
    is_weekend   INTEGER NOT NULL,       -- 0/1
    is_holiday   INTEGER NOT NULL,       -- 0/1
    holiday_name TEXT
);

-- Dimension taux de change (innovation : coût d'import) ---------------------
CREATE TABLE dim_fx (
    date        TEXT PRIMARY KEY,        -- 'YYYY-MM-DD'
    mga_per_eur REAL NOT NULL            -- Ariary par Euro
);

-- Faits de ventes -----------------------------------------------------------
CREATE TABLE fct_sales (
    date        TEXT    NOT NULL,
    store_id    INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    units_sold  INTEGER NOT NULL,
    price_mga   REAL    NOT NULL,
    promo_flag  INTEGER NOT NULL,        -- 0/1
    revenue_mga REAL    NOT NULL,
    PRIMARY KEY (date, store_id, product_id),
    FOREIGN KEY (store_id)   REFERENCES dim_store(store_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
    FOREIGN KEY (date)       REFERENCES dim_date(date)
);

CREATE INDEX IF NOT EXISTS idx_sales_prod_store_date
    ON fct_sales (product_id, store_id, date);
