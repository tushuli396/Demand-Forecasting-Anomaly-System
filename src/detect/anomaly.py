import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def stl_zscore_anomalies(df: pd.DataFrame, date_col="ds", y_col="y", period=7, z_thresh=3.0):
    # expects daily data
    s = df.sort_values(date_col)[y_col].astype(float).values
    if len(s) < period * 3:
        # not enough history, return none
        out = df[[date_col, y_col]].copy()
        out["resid"] = np.nan
        out["zscore"] = 0.0
        out["is_anom"] = 0
        return out
    res = STL(s, period=period, robust=True).fit()
    resid = res.resid
    z = (resid - np.nanmean(resid)) / (np.nanstd(resid) + 1e-8)
    out = df[[date_col, y_col]].copy()
    out["resid"] = resid
    out["zscore"] = z
    out["is_anom"] = (np.abs(z) >= z_thresh).astype(int)
    return out
