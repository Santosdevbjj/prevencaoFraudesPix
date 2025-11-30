import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    recall = recall_score(y_test, (proba > 0.5).astype(int))
    cm = confusion_matrix(y_test, (proba > 0.5).astype(int))
    return {"auc": auc, "recall": recall, "confusion_matrix": cm.tolist()}
