
-- SQLite schema for demand forecasting pipeline
CREATE TABLE IF NOT EXISTS skus (
    sku_id TEXT PRIMARY KEY,
    sku_name TEXT
);

CREATE TABLE IF NOT EXISTS sales (
    sku_id TEXT,
    ds DATE,
    y REAL,
    PRIMARY KEY (sku_id, ds),
    FOREIGN KEY (sku_id) REFERENCES skus(sku_id)
);

CREATE TABLE IF NOT EXISTS forecasts (
    sku_id TEXT,
    model TEXT,
    ds DATE,
    yhat REAL,
    yhat_lower REAL,
    yhat_upper REAL,
    is_ensemble INTEGER DEFAULT 0,
    PRIMARY KEY (sku_id, model, ds, is_ensemble)
);

CREATE TABLE IF NOT EXISTS anomalies (
    sku_id TEXT,
    ds DATE,
    y REAL,
    resid REAL,
    zscore REAL,
    is_anom INTEGER,
    PRIMARY KEY (sku_id, ds)
);

CREATE TABLE IF NOT EXISTS metrics (
    sku_id TEXT,
    model TEXT,
    metric TEXT,
    value REAL,
    PRIMARY KEY (sku_id, model, metric)
);
