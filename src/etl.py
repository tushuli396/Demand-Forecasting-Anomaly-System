import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

def main(args):
    csv = pd.read_csv(args.csv, parse_dates=["ds"])
    csv["ds"] = csv["ds"].dt.date
    eng = create_engine(args.db, future=True)
    # create schema
    schema_sql = Path("sql/schema.sql").read_text()
    with eng.begin() as conn:
        for stmt in schema_sql.split(";\n"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        # insert skus
        skus = csv[["sku_id"]].drop_duplicates().assign(sku_name=lambda d: d["sku_id"])
        skus.to_sql("skus", conn, if_exists="append", index=False)
        # insert sales
        csv[["sku_id", "ds", "y"]].to_sql("sales", conn, if_exists="append", index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--db", required=True)
    main(p.parse_args())
