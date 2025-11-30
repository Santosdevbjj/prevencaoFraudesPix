import pandas as pd
from src.features.validators import check_no_leakage

def test_target_present():
    df = pd.DataFrame({"is_fraud":[0,1]})
    check_no_leakage(df)
