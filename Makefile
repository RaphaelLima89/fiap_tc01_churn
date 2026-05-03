.PHONY: install lint format test run train clean help

help:
	@echo "Targets disponíveis:"
	@echo "  install   - poetry install (deps + pacote editavel)"
	@echo "  lint      - ruff check src tests"
	@echo "  format    - ruff format src tests"
	@echo "  test      - pytest -v"
	@echo "  run       - uvicorn na porta 8000 com reload"
	@echo "  train     - executa notebooks 01_eda e 02_modelos"
	@echo "  clean     - remove caches"

install:
	poetry install

lint:
	poetry run ruff check src tests

format:
	poetry run ruff format src tests

test:
	poetry run pytest

run:
	poetry run uvicorn churn_predictor.api.main:app --reload --port 8000

train:
	poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelos.ipynb

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__