# src/data/simulate_transactions.py
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

def simulate(n_rows: int = 200_000, fraud_rate: float = 0.008) -> pd.DataFrame:
    ts0 = pd.Timestamp("2025-11-01")
    timestamps = ts0 + pd.to_timedelta(rng.integers(0, 60*24*30, size=n_rows), unit="m")
    user_ids = rng.integers(1, 50_000, size=n_rows)
    amounts = rng.uniform(10, 5000, size=n_rows).round(2)

    # transacionais adicionais
    sender_account_age_days = rng.integers(1, 3650, size=n_rows)
    recipient_bank = rng.choice(["001","237","260","290","341","033"], size=n_rows)
    is_first_time_recipient = rng.choice([0,1], size=n_rows, p=[0.9, 0.1])

    # risco interno
    internal_risk_score = rng.normal(600, 80, size=n_rows).clip(300, 900)
    max_credit_limit = rng.uniform(500, 50000, size=n_rows)
    historical_default_rate = rng.uniform(0.0, 0.2, size=n_rows)

    # comportamento login/device (simples)
    time_since_last_login = rng.integers(5, 60*60*24, size=n_rows)  # em segundos
    change_of_device = rng.choice([0,1], size=n_rows, p=[0.97, 0.03])

    # fraude desbalanceada
    is_fraud = rng.choice([0,1], size=n_rows, p=[1 - fraud_rate, fraud_rate])

    df = pd.DataFrame({
        "transaction_id": rng.integers(1e9, 1e10, size=n_rows).astype(str),
        "user_id": user_ids,
        "timestamp": timestamps,
        "amount": amounts,
        "sender_account_age_days": sender_account_age_days,
        "recipient_bank": recipient_bank,
        "is_first_time_recipient": is_first_time_recipient,
        "internal_risk_score": internal_risk_score,
        "max_credit_limit": max_credit_limit,
        "historical_default_rate": historical_default_rate,
        "time_since_last_login": time_since_last_login,
        "change_of_device": change_of_device,
        "is_fraud": is_fraud
    }).sort_values("timestamp").reset_index(drop=True)

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=200_000)
    parser.add_argument("--fraud_rate", type=float, default=0.008)
    parser.add_argument("--out", type=str, default="data/raw/transactions.parquet")
    args = parser.parse_args()

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = simulate(args.n_rows, args.fraud_rate)
    df.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}, rows={len(df)}")
