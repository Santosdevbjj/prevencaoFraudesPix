# src/modeling/inference.py
import time
import joblib
import numpy as np
import pandas as pd
# Removido: from pathlib import Path (F401 - não utilizado)

def infer_single(model, row: pd.Series, feature_cols):
    """
    Realiza a inferência e mede a latência total (pré-processamento + previsão).
    """
    t0 = time.perf_counter()  # <-- Medição começa aqui para incluir o pré-processamento
    
    # Pré-processamento e formato (NumPy 2D)
    X = row[feature_cols].fillna(0).values.reshape(1, -1)
    
    # Previsão do modelo
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

    # Carregamento de Modelo e Dados
    model = joblib.load(args.model)
    df = pd.read_parquet(args.data).tail(args.n_events)
    
    # Lista de Features (corrigida pelo Black/PEP 8)
    feature_cols = [
        "amount", "sender_account_age_days", "internal_risk_score", 
        "max_credit_limit", "historical_default_rate", "time_since_last_login",
        "change_of_device", "is_first_time_recipient", "num_pix_last_1h",
        "avg_amount_last_7d", "std_amount_last_30d", "num_unique_recipients_24h"
    ]
    
    latencies = []
    
    # Simulação da Inferência (Iteração lenta, mas fiel ao evento único)
    print(f"Simulando latência em {args.n_events} eventos...")
    for _, row in df.iterrows():
        _, dt_ms = infer_single(model, row, feature_cols)
        latencies.append(dt_ms)

    # Resultados (Latência Média e P95)
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print(f"\n✨ Resultados da Inferência ({args.n_events} eventos):")
    print(f"  - Latência Média: {avg_lat:.2f} ms")
    print(f"  - Latência P95: {p95_lat:.2f} ms")
