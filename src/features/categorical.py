import pandas as pd

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["recipient_bank"], drop_first=True)
    return df
