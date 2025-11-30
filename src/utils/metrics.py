from sklearn.metrics import roc_auc_score, recall_score

def auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)
