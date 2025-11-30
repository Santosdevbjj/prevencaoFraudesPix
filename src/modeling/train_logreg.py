import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/processed/dataset.parquet")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:,1]
print("AUC:", roc_auc_score(y_test, proba))
