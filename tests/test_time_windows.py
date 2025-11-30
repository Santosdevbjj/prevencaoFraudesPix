import pandas as pd
from src.features.time_windows import features_time_based

def test_shift_prevents_leakage():
    df = pd.DataFrame({"user_id":[1,1], "timestamp":pd.date_range("2025-01-01", periods=2, freq="H"), "amount":[100,200]})
    df = features_time_based(df)
    assert df["num_pix_last_1h"].iloc[1] == 1
