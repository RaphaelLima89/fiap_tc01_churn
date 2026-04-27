### Configurações ###

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

# Carrega as variáveis do ambiente virtual
load_dotenv()

# Raiz do projeto
# Obs.: Este arquivo está em src/churn_predictor/config.py, então precisa subir 3 níveis para chegar a raiz
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Diretórios Principais
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset alvo do projeto
RAW_FILENAME = "telco_churn.csv"
RAW_PATH = RAW_DIR / RAW_FILENAME