import pandas as pd
from prophet import Prophet

def fit_prophet(df: pd.DataFrame):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(df)
    return m

def forecast_prophet(model, periods: int):
    future = model.make_future_dataframe(periods=periods, freq="D")
    fcst = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst.tail(periods)
