import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def fit_global_lstm(series_dict, lag=30, horizon=1, epochs=10, lr=1e-3, batch_size=32, verbose=0, seed=42):
    # series_dict: {sku_id: pd.Series of y}
    Xs, ys = [], []
    scaler = StandardScaler()
    for s in series_dict.values():
        vals = s.values.astype(float).reshape(-1, 1)
        vals = scaler.fit_transform(vals).flatten()
        for i in range(lag, len(vals) - horizon + 1):
            Xs.append(vals[i - lag:i])
            ys.append(vals[i:i + horizon])
    X = np.array(Xs).reshape(-1, lag, 1)
    y = np.array(ys).reshape(-1, horizon)
    model = Sequential([
        LSTM(32, input_shape=(lag, 1)),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(lr), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, scaler, lag

def forecast_lstm(model, scaler, lag, series: pd.Series, horizon: int):
    vals = series.values.astype(float).reshape(-1, 1)
    vals_scaled = scaler.transform(vals).flatten()
    window = vals_scaled[-lag:].tolist()
    preds = []
    cur = window.copy()
    for _ in range(horizon):
        x = np.array(cur[-lag:]).reshape(1, lag, 1)
        yhat = model.predict(x, verbose=0).flatten()[0]
        preds.append(yhat)
        cur.append(yhat)
    preds = np.array(preds).reshape(-1, 1)
    inv = scaler.inverse_transform(preds).flatten()
    return inv
