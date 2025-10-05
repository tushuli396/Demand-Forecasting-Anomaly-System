import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dateutil.relativedelta import relativedelta

from .utils import mape, smape, naive_seasonal_forecast
from .detect.anomaly import stl_zscore_anomalies
from .models.prophet_model import fit_prophet, forecast_prophet
from .models.lstm_model import fit_global_lstm, forecast_lstm

def main(args):
    eng = create_engine(args.db, future=True)
    with eng.begin() as conn:
        sales = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["ds"])
    # train/val split (per SKU)
    horizon = args.horizon
    val_days = args.val_days

    all_metrics = []
    all_fcsts_rows = []
    all_anoms_rows = []

    # prepare for global LSTM
    series_dict = {}
    series_per_sku = {}

    for sku, df in sales.groupby("sku_id"):
        df = df.sort_values("ds").reset_index(drop=True)
        series = df.set_index("ds")["y"]
        series_per_sku[sku] = series
        series_dict[sku] = series

    # Fit global LSTM once
    lstm_model, scaler, lag = fit_global_lstm(series_dict, lag=30, horizon=1, epochs=args.epochs, verbose=0)

    for sku, series in series_per_sku.items():
        df = series.reset_index().rename(columns={"index": "ds", "y": "y"})
        train = df.iloc[:-val_days] if len(df) > val_days + 30 else df.iloc[:-7]
        val = df.iloc[len(train):]

        # Prophet (per SKU)
        m = fit_prophet(train.rename(columns={"ds": "ds", "y": "y"}))
        p_fcst_val = m.predict(val[["ds"]])[ ["ds", "yhat" ] ]
        p_fcst_fut = forecast_prophet(m, periods=horizon)

        # LSTM (global)
        l_fcst_vals = forecast_lstm(lstm_model, scaler, lag, train.set_index("ds")["y"], horizon=len(val))
        l_fcst_val = pd.DataFrame({"ds": val["ds"].values, "yhat": l_fcst_vals})
        l_fcst_future_vals = forecast_lstm(lstm_model, scaler, lag, df.set_index("ds")["y"], horizon=horizon)
        l_fcst_fut = pd.DataFrame({"ds": pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"),
                                   "yhat": l_fcst_future_vals})
        # Baseline (naive seasonal) for validation only
        nb = naive_seasonal_forecast(train["y"], len(val))
        nb = pd.Series(nb.values, index=val.index)

        # Validation metrics
        p_mape = mape(val["y"], p_fcst_val["yhat"])
        l_mape = mape(val["y"], l_fcst_val["yhat"])
        b_mape = mape(val["y"], nb.values)

        # Record metrics
        all_metrics += [
            {"sku_id": sku, "model": "prophet", "metric": "mape_val", "value": float(p_mape)},
            {"sku_id": sku, "model": "lstm", "metric": "mape_val", "value": float(l_mape)},
            {"sku_id": sku, "model": "baseline", "metric": "mape_val", "value": float(b_mape)},
        ]

        # Choose best (lower is better)
        best_model = "prophet" if p_mape <= l_mape else "lstm"

        # Ensemble weight by inverse MAPE (clip to avoid div by 0)
        p_w = 1.0 / max(p_mape, 1e-6)
        l_w = 1.0 / max(l_mape, 1e-6)
        p_w, l_w = p_w / (p_w + l_w), l_w / (p_w + l_w)

        # Save future forecasts (both models + ensemble) to DB rows
        p_fut = p_fcst_fut.copy()
        p_fut["model"] = "prophet"
        l_fut = l_fcst_fut.copy()
        l_fut["yhat_lower"] = l_fut["yhat"] * 0.9
        l_fut["yhat_upper"] = l_fut["yhat"] * 1.1
        l_fut["model"] = "lstm"

        ens = p_fut[["ds", "yhat"]].copy()
        ens["yhat"] = p_w * p_fut["yhat"].values + l_w * l_fut["yhat"].values
        ens["yhat_lower"] = ens["yhat"] * 0.9
        ens["yhat_upper"] = ens["yhat"] * 1.1
        ens["model"] = "ensemble"

        for df_f, model, is_ens in [(p_fut, "prophet", 0), (l_fut, "lstm", 0), (ens, "ensemble", 1)]:
            for _, r in df_f.iterrows():
                all_fcsts_rows.append({
                    "sku_id": sku,
                    "model": model,
                    "ds": pd.to_datetime(r["ds"]).date(),
                    "yhat": float(r["yhat"]),
                    "yhat_lower": float(r.get("yhat_lower", r["yhat"]*0.9)),
                    "yhat_upper": float(r.get("yhat_upper", r["yhat"]*1.1)),
                    "is_ensemble": is_ens
                })

        # Anomalies on **history** (last 180 days)
        hist = df.tail(180)
        an = stl_zscore_anomalies(hist, date_col="ds", y_col="y", period=7, z_thresh=args.z_thresh)
        for _, r in an.iterrows():
            all_anoms_rows.append({
                "sku_id": sku,
                "ds": pd.to_datetime(r["ds"]).date(),
                "y": float(r["y"]),
                "resid": float(r["resid"]) if pd.notnull(r["resid"]) else None,
                "zscore": float(r["zscore"]),
                "is_anom": int(r["is_anom"]),
            })

        # improvement over baseline
        imp = (b_mape - min(p_mape, l_mape)) / (b_mape + 1e-8) * 100
        all_metrics.append({"sku_id": sku, "model": "selected", "metric": "improvement_vs_baseline_pct", "value": float(imp)})

    # Write to DB
    eng = create_engine(args.db, future=True)
    with eng.begin() as conn:
        # clear & write
        conn.execute(text("DELETE FROM forecasts"))
        conn.execute(text("DELETE FROM anomalies"))
        conn.execute(text("DELETE FROM metrics"))
        pd.DataFrame(all_fcsts_rows).to_sql("forecasts", conn, if_exists="append", index=False)
        pd.DataFrame(all_anoms_rows).to_sql("anomalies", conn, if_exists="append", index=False)
        pd.DataFrame(all_metrics).to_sql("metrics", conn, if_exists="append", index=False)

    print("Training complete. Forecasts, anomalies, and metrics saved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--horizon", type=int, default=90)
    p.add_argument("--val_days", type=int, default=90)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--z_thresh", type=float, default=3.0)
    main(p.parse_args())
