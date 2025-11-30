from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

def evaluate(model, X_test, y_test, threshold: float = 0.5):
    """
    Avalia o modelo com base em métricas-chave para detecção de fraude, 
    permitindo um ponto de corte flexível.
    """
    
    # 1. Previsão de Probabilidades e Classificação Binária
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    
    # 2. Cálculo das Métricas
    auc = roc_auc_score(y_test, proba)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Cálculo do FPR (Falso Positivo Rate)
    # cm: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()
    
    # Evita divisão por zero
    if (FP + TN) == 0:
        fpr = 0.0
    else:
        fpr = FP / (FP + TN)
    
    # 4. Retorno dos Resultados
    return {
        "auc": auc, 
        "threshold": threshold,
        "recall": recall, 
        "fpr": fpr,
        "confusion_matrix": cm.tolist()
    }
