import argparse
import numpy as np
import pandas as pd
from numpy.random import default_rng

def gen_series(start, end, trend=0.01, weekly_amp=0.15, yearly_amp=0.3, noise=0.1, level=100.0, seed=42):
    rng = default_rng(seed)
    idx = pd.date_range(start, end, freq="D")
    t = np.arange(len(idx))
    # trend
    y = level * (1 + trend * t / len(idx))
    # weekly seasonality
    y *= (1 + weekly_amp * np.sin(2 * np.pi * (t % 7) / 7))
    # yearly seasonality (approximate)
    y *= (1 + yearly_amp * np.sin(2 * np.pi * t / 365))
    # promo spikes
    promos = rng.choice([0, 1], size=len(idx), p=[0.98, 0.02])
    y *= (1 + 0.5 * promos)
    # noise
    y *= (1 + noise * rng.standard_normal(len(idx)))
    y = np.maximum(0, y)
    return pd.DataFrame({"ds": idx, "y": y})

def main(args):
    rng = default_rng(123)
    rows = []
    for i in range(args.n_skus):
        sku = f"SKU-{i+1:03d}"
        df = gen_series(args.start, args.end,
                        trend=rng.uniform(0.0, 0.05),
                        weekly_amp=rng.uniform(0.05, 0.3),
                        yearly_amp=rng.uniform(0.1, 0.5),
                        noise=rng.uniform(0.05, 0.2),
                        level=rng.uniform(50, 500),
                        seed=1000 + i)
        df["sku_id"] = sku
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)[["sku_id", "ds", "y"]]
    out.to_csv(args.out, index=False)
    print(f"Wrote sample to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_skus", type=int, default=10)
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--out", type=str, default="data/sample_sales.csv")
    args = p.parse_args()
    main(args)
