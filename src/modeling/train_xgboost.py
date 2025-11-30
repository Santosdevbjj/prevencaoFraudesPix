# src/modeling/train_xgboost.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

def prepare(df: pd.DataFrame):
    # Selecionar features úteis
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
    # Encoding simples do bank
    df = df.copy()
    df = pd.get_dummies(df, columns=["recipient_bank"], drop_first=True)

    X = df[cols + [c for c in df.columns if c.startswith("recipient_bank_")]].fillna(0)
    y = df["is_fraud"].astype(int)
    return X, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/dataset.parquet")
    parser.add_argument("--out_model", type=str, default="models/artifacts/model_xgb.joblib")
    parser.add_argument("--out_report", type=str, default="models/reports/metrics_xgb.json")
    args = parser.parse_args()

    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("models/reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    X, y = prepare(df)

    # split temporal (exemplo simplificado): último 10% como teste
    n = len(df)
    cut = int(n * 0.9)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_test, y_test = X.iloc[cut:], y.iloc[cut:]

    # lidar com desbalanceamento via scale_pos_weight
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

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
        scale_pos_weight=pos_weight
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)

    # escolher threshold visando recall alto com FPR baixo (exemplo: varrer thresholds)
    thresholds = np.linspace(0.5, 0.99, 20)
    best = {"threshold": None, "recall": 0, "fpr": 1, "auc": auc}
    y_true = y_test.values
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        recall = recall_score(y_true, y_pred)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fpr = fp / max(fp + tn, 1)
        if recall >= best["recall"] and fpr <= best["fpr"]:
            best.update({"threshold": float(t), "recall": float(recall), "fpr": float(fpr), "auc": float(auc)})

    joblib.dump(model, args.out_model)
    with open(args.out_report, "w") as f:
        json.dump(best, f, indent=2)

    print(f"AUC: {auc:.4f} | Best threshold: {best['threshold']} | Recall: {best['recall']:.3f} | FPR: {best['fpr']:.4f}")
