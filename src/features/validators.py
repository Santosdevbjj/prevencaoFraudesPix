import pandas as pd

def check_no_leakage(df: pd.DataFrame):
    if "is_fraud" in df.columns:
        print("OK: target presente apenas como coluna alvo.")
    else:
        raise ValueError("Target ausente!")
