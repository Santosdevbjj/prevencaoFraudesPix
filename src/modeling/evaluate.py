# src/modeling/evaluate.py
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Avalia um modelo de classificação binária em um conjunto de teste.

    Calcula métricas-chave para detecção de fraude, permitindo definir
    um ponto de corte (threshold) flexível para a probabilidade prevista.

    Args:
        model: Modelo treinado com método `predict_proba`.
        X_test (array-like): Features de teste.
        y_test (array-like): Labels verdadeiros.
        threshold (float): Limite de decisão para classificar como fraude.

    Returns:
        dict: Dicionário com métricas:
            - auc: Área sob a curva ROC.
            - threshold: Limite usado para classificação.
            - recall: Sensibilidade (taxa de detecção de fraude).
            - fpr: Taxa de falsos positivos.
            - confusion_matrix: Matriz de confusão em formato lista.
    """
    # 1. Previsão de probabilidades e classificação binária
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # 2. Cálculo das métricas
    auc = roc_auc_score(y_test, proba)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 3. Cálculo do FPR (False Positive Rate)
    # cm: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    fpr = 0.0 if (FP + TN) == 0 else FP / (FP + TN)

    # 4. Retorno dos resultados
    return {
        "auc": auc,
        "threshold": threshold,
        "recall": recall,
        "fpr": fpr,
        "confusion_matrix": cm.tolist(),
    }
