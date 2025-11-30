import pandas as pd
from src.modeling.train_xgboost import prepare
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("data/processed/dataset.parquet")
X, y = prepare(df)

n = len(df)
cut = int(n*0.9)
X_train, y_train = X.iloc[:cut], y.iloc[:cut]
X_test, y_test = X.iloc[cut:], y.iloc[cut:]

model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:,1]
print("AUC:", roc_auc_score(y_test, proba))
