.PHONY: format setup clean simulate features train test lint quality all

# ==============================================================================
# QUALIDADE E MANUTENÇÃO
# ==============================================================================
format:
	poetry run black src tests
	@echo "Código formatado com sucesso! (Black)"

setup:
	poetry install

clean:
	@echo "Limpando artefatos gerados (datasets e modelos)..."
	rm -rf data/processed/*
	rm -rf models/artifacts/*

lint:
	poetry run ruff check .

# Target de Orquestração de Qualidade
# Garante que a formatação é executada antes do lint e dos testes.
quality: format lint test
	@echo "✅ Pipeline de Qualidade concluído."

# ==============================================================================
# PIPELINE DE MLOPS
# ==============================================================================
simulate:
	poetry run python src/data/simulate_transactions.py

features:
	poetry run python src/pipelines/build_dataset.py

train:
	poetry run python src/pipelines/train_and_eval.py --model xgboost

test:
	poetry run pytest -q

# Target principal para rodar todo o fluxo
all: simulate features train quality
