import pandas as pd
from .models.prophet_model import fit_prophet, forecast_prophet
from .models.lstm_model import fit_global_lstm, forecast_lstm

def run_prophet_forecast(ts: pd.DataFrame, horizon: int):
    m = fit_prophet(ts.rename(columns={"ds": "ds", "y": "y"}))
    fcst = forecast_prophet(m, periods=horizon)
    return fcst

def run_lstm_forecast(series_dict, series_per_sku, horizon: int, lag: int = 30, epochs: int = 10):
    model, scaler, lag = fit_global_lstm(series_dict, lag=lag, horizon=1, epochs=epochs)
    out = {}
    for sku, s in series_per_sku.items():
        yhat = forecast_lstm(model, scaler, lag, s, horizon=horizon)
        df = pd.DataFrame({"ds": pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"),
                           "yhat": yhat, "yhat_lower": yhat * 0.9, "yhat_upper": yhat * 1.1})
        out[sku] = df
    return out
