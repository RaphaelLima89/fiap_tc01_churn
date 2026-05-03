# Churn Predictor — Telco

Modelo de classificação binária para prever cancelamento de clientes de uma operadora de telecom, treinado em cima do dataset público Telco Customer Churn (IBM, distribuído via Kaggle), com 7.043 clientes e 21 features. Projeto Tech Challenge utilizado na Fase 01 - Produtização de Modelos

A entrega cobre o ciclo inteiro do problema: exploração e limpeza, baselines comparados via cross-validation estratificada, uma rede neural (MLP em PyTorch) treinada com early stopping, tracking de experimentos com MLflow, API de inferência em FastAPI com logging estruturado, e testes automatizados. O modelo escolhido para produção foi um Random Forest balanceado, e a decisão final levou em conta não apenas PR-AUC, mas também o custo assimétrico de errar.

---

## Quick start

Pré-requisitos: Python 3.12 e Poetry 2.

```bash
git clone https://github.com/RaphaelLima89/fiap_tc01_churn churn-predictor
cd churn-predictor
poetry install
```

Para subir a API com hot reload:

```bash
make run                    # Linux/Mac/WSL
.\tasks.ps1 run             # Windows PowerShell (sem GNU make)
```

A API sobe em `http://127.0.0.1:8000`. O Swagger fica em `/docs`.

Para rodar os testes:

```bash
make test                   # ou .\tasks.ps1 test
```

São 3 testes — um smoke do pipeline de dados, um de schema (com pandera) validando o parquet limpo, e um end-to-end da API usando o `TestClient` do FastAPI. Os três cobrem as três categorias que o enunciado pede explicitamente.

Para lint:

```bash
make lint                   # ou .\tasks.ps1 lint
```

## Deploy live

API pública em <https://fiap-tc01-churn.onrender.com>. Swagger em [/docs](https://fiap-tc01-churn.onrender.com/docs).

---


---

## Estrutura

```
.
├── data/
│   ├── raw/                # Telco Customer Churn original (CSV)
│   └── interim/            # Parquet limpo apos EDA
├── notebooks/
│   ├── 01_eda.ipynb        # Exploração, limpeza, persistência do parquet
│   └── 02_modelos.ipynb    # Baselines, MLP, comparação, analise de custo
├── src/churn_predictor/
│   ├── data/loader.py      # Carga + train/test split estratificado
│   ├── features/           # Pipeline sklearn (OneHot + StandardScaler)
│   ├── models/             # Baselines sklearn e MLP em PyTorch
│   ├── api/                # FastAPI: schemas, inferência, middleware, main
│   ├── utils/logging.py    # structlog
│   └── config.py           # Constantes (paths, seeds, threshold)
├── models/final.joblib     # Pipeline sklearn serializado (RF balanceado)
├── tests/                  # pytest: smoke, schema, API
├── docs/                   # Model Card, ML Canvas, arquitetura, monitoramento
├── mlruns/                 # Tracking local do MLflow (gerado, ignorado)
├── Makefile                # Targets: install, lint, format, test, run, train
├── tasks.ps1               # Equivalente em PowerShell para Windows
└── pyproject.toml          # Single source of truth: deps, ruff, pytest
```

---


## API

Três endpoints: `GET /` (redireciona ao Swagger), `GET /health` (status + `model_loaded`, sempre 200) e `POST /predict` (probabilidade, predição binária no threshold 0.08, latência, `request_id`, versão do modelo). Dois middlewares cobrem cada request: um gera UUID por request (ou ecoa `X-Request-ID` do cliente), outro mede latência total e expõe em `X-Response-Time-Ms`. Logs via `structlog` em formato chave-valor, correlacionáveis por `request_id`. Se `models/final.joblib` faltar no startup, a API sobe em modo degraded — `/health` retorna 200 com `status: "degraded"`, `/predict` devolve 503. Topologia em `docs/architecture.md`.

Exemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.50, "TotalCharges": 1026.00
  }'
```

---

## Reprodutibilidade

Seed `42` em todos os pontos não-determinísticos (`train_test_split`, `RandomForestClassifier`, `KFold`, `numpy`, `torch.manual_seed`). Tracking do MLflow em `mlruns/` (filesystem backend); cada run tem tags, parâmetros, métricas (CV mean/std + test) e modelo serializado como artifact. Abrir a UI: `poetry run mlflow ui` → `http://127.0.0.1:5000`.

---


## Para mais detalhes

- `docs/model_card.md` — Model Card completo, com performance por subgrupo, vieses identificados e cenários de falha conhecidos
- `docs/ml_canvas.md` — ML Canvas com stakeholders, métricas de negócio e SLOs
- `docs/architecture.md` — Decisão de deploy (real-time vs batch) e justificativa
- `docs/monitoring.md` — Plano de monitoramento, métricas, alertas, playbook de resposta a incidentes