import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from contextlib import contextmanager

@contextmanager
def engine_ctx(db_url: str):
    eng = create_engine(db_url, future=True)
    try:
        yield eng
    finally:
        eng.dispose()

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where((np.abs(y_true) + np.abs(y_pred)) == 0, 1e-8, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

def naive_seasonal_forecast(series: pd.Series, horizon: int) -> pd.Series:
    # simple mix of last-7 and last-365 if available
    y = series.values
    out = []
    for h in range(1, horizon + 1):
        v7 = y[-7] if len(y) >= 7 else y[-1]
        v365 = y[-365] if len(y) >= 365 else y[-1]
        out.append((v7 + v365) / 2.0 if len(y) >= 365 else v7)
    return pd.Series(out, index=pd.RangeIndex(horizon))

def to_daily(df, date_col="ds", y_col="y"):
    # forward-fills missing dates per sku
    idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    out = df.set_index(date_col).reindex(idx).rename_axis(date_col).reset_index()
    out[y_col] = out[y_col].interpolate(limit_direction="both")
    return out
