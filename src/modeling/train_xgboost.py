# src/modeling/train_xgboost.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier
from src.modeling.evaluate import evaluate # Importando a função de avaliação aprimorada

def prepare(df: pd.DataFrame):
    """
    Prepara o DataFrame para o treinamento, selecionando e codificando features.
    """
    # Lista de features numéricas e de agregação
    cols = [
        "amount",
        "sender_account_age_days",
        "internal_risk_score",
        "max_credit_limit",
        "historical_default_rate",
        "time_since_last_login",
        "change_of_device",
        "is_first_time_recipient",
        "num_pix_last_1h",
        "avg_amount_last_7d",
        "std_amount_last_30d",
        "num_unique_recipients_24h",
    ]
    
    df = df.copy()
    
    # Encoding categórico simples (One-Hot Encoding)
    # Assumimos que 'recipient_bank' ainda está presente no dataset.parquet
    df = pd.get_dummies(df, columns=["recipient_bank"], drop_first=True)

    # Criação final de X e y
    X = df[
        cols + [c for c in df.columns if c.startswith("recipient_bank_")]
    ].fillna(0)
    y = df["is_fraud"].astype(int)
    return X, y

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/processed/dataset.parquet"
    )
    parser.add_argument(
        "--out_model",
        type=str,
        default="models/artifacts/model_xgb.joblib",
    )
    parser.add_argument(
        "--out_report",
        type=str,
        default="models/reports/metrics_xgb.json",
    )
    args = parser.parse_args()

    # Criação dos diretórios de saída
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("models/reports").mkdir(parents=True, exist_ok=True)

    # 1. Preparação dos Dados
    df = pd.read_parquet(args.data)
    X, y = prepare(df)

    # 2. Divisão Temporal de Dados: último 10% como teste (simulação de futuro)
    n = len(df)
    cut = int(n * 0.9)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_test, y_test = X.iloc[cut:], y.iloc[cut:]

    # Lidar com desbalanceamento (Taxa Negativos / Taxa Positivos)
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # 3. Treinamento do Modelo
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=pos_weight, # Tratamento de desbalanceamento
    )

    print("Iniciando treinamento do XGBoost...")
    model.fit(X_train, y_train)

    # 4. Avaliação e Otimização do Threshold
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # Busca o melhor threshold (Trade-off entre Recall e FPR)
    thresholds = np.linspace(0.5, 0.99, 20)
    best_metrics = {"threshold": 0.0, "recall": 0.0, "fpr": 1.0, "auc": float(auc)}
    
    for t in thresholds:
        metrics = evaluate(model, X_test, y_test, threshold=t)
        
        # Otimiza o threshold: prioriza Recall mais alto, depois o menor FPR
        if metrics["recall"] >= best_metrics["recall"] and metrics["fpr"] <= best_metrics["fpr"]:
            best_metrics.update({
                "threshold": metrics["threshold"], 
                "recall": metrics["recall"], 
                "fpr": metrics["fpr"],
                "auc": metrics["auc"],
            })

    # 5. Salvamento dos Artefatos
    joblib.dump(model, args.out_model)
    with open(args.out_report, "w") as f:
        # Garante que os valores numéricos sejam floats nativos para JSON
        json.dump({k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                   for k, v in best_metrics.items()}, f, indent=2)

    print("\n✨ Treinamento Concluído!")
    print(f"  - AUC: {best_metrics['auc']:.4f}")
    print(f"  - Best Threshold: {best_metrics['threshold']:.4f}")
    print(f"  - Melhor Recall: {best_metrics['recall']:.3f}")
    print(f"  - FPR (Bloqueios Incorretos): {best_metrics['fpr']:.4f}")
    print(f"Modelo salvo em: {args.out_model}")
