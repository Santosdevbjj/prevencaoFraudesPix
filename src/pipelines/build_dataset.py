import pandas as pd
from src.data.simulate_transactions import simulate
from src.features.time_windows import features_time_based
from src.features.categorical import encode_categorical
from src.utils.io import write_parquet

df = simulate(200000, 0.008)
df = features_time_based(df)
df = encode_categorical(df)

write_parquet(df, "data/processed/dataset.parquet")
print("Dataset salvo em data/processed/dataset.parquet")
