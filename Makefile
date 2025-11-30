.PHONY: setup all pipeline simulate features train inference test lint format quality clean

# Definindo variáveis para caminhos e comandos comuns
PYTHON_RUN := poetry run python
DATA_PROCESSED := data/processed/dataset.parquet
MODEL_ARTIFACT := models/artifacts/model_xgb.joblib

# ==============================================================================
# FLUXO DE TRABALHO PRINCIPAL
# ==============================================================================
all: simulate features train test quality  ## Executa o pipeline completo (Simulação -> Treinamento -> Testes -> Qualidade)

simulate:
	$(PYTHON_RUN) src/data/simulate_transactions.py --n_rows 200000 --fraud_rate 0.008

features:
	$(PYTHON_RUN) src/pipelines/build_dataset.py --input data/raw/transactions.parquet --output $(DATA_PROCESSED)

train:
	$(PYTHON_RUN) src/pipelines/train_and_eval.py --data $(DATA_PROCESSED) --model xgboost

inference:
	$(PYTHON_RUN) src/modeling/inference.py --model $(MODEL_ARTIFACT) --n_events 1000

# ==============================================================================
# QUALIDADE E LIMPEZA
# ==============================================================================

quality: format lint test ## Executa formatação, lint e testes

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run black src tests

setup:
	poetry install

clean:
	@echo "Limpando artefatos gerados (datasets e modelos)..."
	rm -rf data/processed/*
	rm -rf models/artifacts/*
