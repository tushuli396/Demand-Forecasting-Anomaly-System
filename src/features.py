import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    df = df.copy()
    ds = pd.to_datetime(df[date_col])
    df["dow"] = ds.dt.dayofweek
    df["dom"] = ds.dt.day
    df["month"] = ds.dt.month
    df["week"] = ds.dt.isocalendar().week.astype(int)
    return df

def make_supervised(series: pd.Series, lag: int = 30, horizon: int = 1):
    X, y = [], []
    vals = series.values.astype(float)
    for i in range(lag, len(vals) - horizon + 1):
        X.append(vals[i - lag:i])
        y.append(vals[i:i + horizon])
    return np.array(X), np.array(y)
