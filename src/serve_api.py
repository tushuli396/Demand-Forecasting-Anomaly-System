from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import os
import pandas as pd

DB_URL = os.environ.get("DF_DB_URL", "sqlite:///data/retail.db")
app = FastAPI(title="Demand Forecasting & Anomaly Detection API")

def df_from_sql(q: str, **params):
    eng = create_engine(DB_URL, future=True)
    with eng.begin() as conn:
        df = pd.read_sql(text(q), conn, params=params)
    eng.dispose()
    return df

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast/{sku_id}")
def forecast(sku_id: str):
    df = df_from_sql("""SELECT ds, yhat, yhat_lower, yhat_upper FROM forecasts
                       WHERE sku_id=:sku AND is_ensemble=1 ORDER BY ds""", sku=sku_id)
    if df.empty:
        raise HTTPException(404, detail="No forecast found for sku")
    return df.to_dict(orient="records")

@app.get("/anomalies/{sku_id}")
def anomalies(sku_id: str):
    df = df_from_sql("SELECT ds, y, resid, zscore, is_anom FROM anomalies WHERE sku_id=:sku ORDER BY ds", sku=sku_id)
    if df.empty:
        raise HTTPException(404, detail="No anomalies found for sku")
    return df.to_dict(orient="records")
