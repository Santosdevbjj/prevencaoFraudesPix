uformat:
	poetry run black src tests

setup:
	poetry install

clean:
	@echo "Limpando artefatos gerados (datasets e modelos)..."
	rm -rf data/processed/*
	rm -rf models/artifacts/*
