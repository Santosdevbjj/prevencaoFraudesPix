import argparse
import joblib
import pandas as pd
from pathlib import Path

# Importação dos módulos do projeto
from src.modeling.train_xgboost import prepare # Prepara X e y
from src.modeling.evaluate import evaluate     # Avaliação completa (AUC, Recall, FPR, CM)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def train_and_evaluate(data_path: str, model_name: str):
    """
    Orquestra o treinamento do modelo (XGBoost ou LogReg) e a avaliação completa.
    """
    
    # 1. Configuração de IO
    model_artifact_path = Path(f"models/artifacts/model_{model_name}.joblib")
    report_path = Path(f"models/reports/metrics_{model_name}.json")
    
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("models/reports").mkdir(parents=True, exist_ok=True)

    # 2. Carregamento e Preparação dos Dados
    print(f"Lendo dados de: {data_path}")
    df = pd.read_parquet(data_path)
    X, y = prepare(df)

    # 3. Divisão Temporal (Split de 90% Treino / 10% Teste)
    n = len(df)
    cut = int(n * 0.9)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_test, y_test = X.iloc[cut:], y.iloc[cut:]
    
    # Cálculo do peso de posição para lidar com o desbalanceamento
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # 4. Inicialização do Modelo
    if model_name == "xgboost":
        # Configurações otimizadas do XGBoost, incluindo scale_pos_weight
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, n_jobs=-1, tree_method="hist",
            scale_pos_weight=pos_weight
        )
    elif model_name == "logreg":
        # Logistic Regression para baseline
        model = LogisticRegression(solver='liblinear', random_state=42, 
                                   class_weight='balanced')
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")

    print(f"Iniciando treinamento do modelo: {model_name}...")
    model.fit(X_train, y_train)

    # 5. Avaliação e Geração de Relatório
    # Usamos o threshold default de 0.5 apenas para a avaliação básica (função evaluate)
    metrics = evaluate(model, X_test, y_test, threshold=0.5)
    
    # Salvamento do Modelo e Métricas
    joblib.dump(model, model_artifact_path)
    with open(report_path, "w") as f:
        # Serializamos as métricas em JSON
        json_metrics = {k: v.tolist() if isinstance(v, list) else v for k, v in metrics.items()}
        import json # Necessário para dump, não queremos ele no topo para F401
        json.dump(json_metrics, f, indent=2)

    # 6. Logging
    print("\n✨ Treinamento e Avaliação Concluídos!")
    print(f"  - Modelo Salvo em: {model_artifact_path}")
    print(f"  - AUC: {metrics['auc']:.4f}")
    print(f"  - Recall (@0.5): {metrics['recall']:.3f}")
    print(f"  - Matriz de Confusão salva em: {report_path}")

if __name__ == "__main__":
    # Define o parser para receber argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Treina e avalia um modelo (XGBoost ou LogReg) no dataset processado."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dataset.parquet",
        help="Caminho para o dataset processado."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "logreg"],
        default="xgboost",
        help="Modelo a ser treinado (xgboost ou logreg)."
    )
    
    args = parser.parse_args()
    
    train_and_evaluate(
        data_path=args.data,
        model_name=args.model
    )
