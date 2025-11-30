# src/modeling/inference.py
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def infer_single(model, row: pd.Series, feature_cols):
    X = row[feature_cols].fillna(0).values.reshape(1, -1)
    t0 = time.perf_counter()
    p = model.predict_proba(X)[0, 1]
    dt_ms = (time.perf_counter() - t0) * 1000
    return p, dt_ms

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/artifacts/model_xgb.joblib")
    parser.add_argument("--data", type=str, default="data/processed/dataset.parquet")
    parser.add_argument("--n_events", type=int, default=1000)
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_parquet(args.data).tail(args.n_events)
    feature_cols = [
        "amount","sender_account_age_days","internal_risk_score","max_credit_limit",
        "historical_default_rate","time_since_last_login","change_of_device",
        "is_first_time_recipient","num_pix_last_1h","avg_amount_last_7d",
        "std_amount_last_30d","num_unique_recipients_24h"
    ]
    latencies = []
    for _, row in df.iterrows():
        _, dt_ms = infer_single(model, row, feature_cols)
        latencies.append(dt_ms)

    print(f"Latency avg: {np.mean(latencies):.2f} ms | p95: {np.percentile(latencies,95):.2f} ms")
