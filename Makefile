.PHONY: setup simulate features train inference test lint format

setup:
	poetry install

simulate:
	poetry run python src/data/simulate_transactions.py --n_rows 200000 --fraud_rate 0.008

features:
	poetry run python src/pipelines/build_dataset.py --input data/raw/transactions.parquet --output data/processed/dataset.parquet

train:
	poetry run python src/pipelines/train_and_eval.py --data data/processed/dataset.parquet --model xgboost

inference:
	poetry run python src/modeling/inference.py --model models/artifacts/model_xgb.joblib --n_events 1000

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run black src tests
