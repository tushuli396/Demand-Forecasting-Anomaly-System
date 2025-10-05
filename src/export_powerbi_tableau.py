import argparse
import os
import pandas as pd
from sqlalchemy import create_engine

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    eng = create_engine(args.db, future=True)
    with eng.begin() as conn:
        forecasts = pd.read_sql("SELECT * FROM forecasts", conn)
        anomalies = pd.read_sql("SELECT * FROM anomalies", conn)
        metrics = pd.read_sql("SELECT * FROM metrics", conn)
    forecasts.to_csv(os.path.join(args.outdir, "forecasts.csv"), index=False)
    anomalies.to_csv(os.path.join(args.outdir, "anomalies.csv"), index=False)
    metrics.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)
    print(f"Wrote CSVs to {args.outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--outdir", default="outputs")
    main(p.parse_args())
