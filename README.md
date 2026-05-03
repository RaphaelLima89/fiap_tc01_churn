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

---

## Arquitetura

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

## Como o modelo foi escolhido

Quatro candidatos foram comparados: um `DummyClassifier(strategy="most_frequent")` como piso, regressão logística com regularização L2 e `class_weight="balanced"`, Random Forest também balanceado, e um MLP de duas camadas em PyTorch treinado com `BCEWithLogitsLoss` ponderada e early stopping. Os três modelos sklearn passaram por `StratifiedKFold(k=5)`. O MLP foi avaliado direto no test set — a justificativa para não fazer CV manual em PyTorch está na seção de limitações e no Model Card.

Resultado no test set:

| Modelo            | Test PR-AUC | Test ROC-AUC | Test F1 |
|-------------------|-------------|--------------|---------|
| dummy             | 0.265       | 0.500        | 0.000   |
| logreg            | 0.634       | 0.842        | 0.616   |
| **random_forest** | **0.641**   | 0.838        | 0.607   |
| mlp               | 0.636       | 0.843        | 0.626   |

PR-AUC foi a métrica primária porque o dataset é desbalanceado (26,5% de churn) e a classe positiva é a que importa: errar para menos custa cliente perdido. Os três modelos não-triviais ficam praticamente empatados dentro do desvio padrão do CV. O Random Forest venceu por margem pequena, e foi escolhido também porque apresentou probabilidades mais bem calibradas que o MLP, o que importa para a próxima decisão.

Tratar churn como um problema de classificação binária pura, com threshold em 0.5, é uma abordagem incompleta. Um falso negativo (modelo diz que o cliente fica e ele sai) custa muito mais que um falso positivo (oferecer desconto a alguém que ficaria de qualquer forma). Modelando FN em R$ 500 e FP em R$ 50, numa razão 10:1 que é coerente com a literatura de churn em telecom, o 0.5 deixa de ser um bom threshold para todos os modelos.

A curva de custo total foi calculada no test set varrendo o threshold do RF de 0 a 1 com passo de 0.01. O mínimo cai em **threshold = 0.08**, o que leva o custo de R$ 76.250 (no 0.5) para **R$ 38.650** — economia de 49,3%. Esse é o valor que está configurado em `API_DECISION_THRESHOLD` e aplicado pelo endpoint `/predict` ao converter `predict_proba` em predição binária.

Vale destacar um detalhe: no threshold default, o MLP (R$ 51.650) era mais barato que o RF (R$ 76.250). Uma comparação restrita ao custo no 0.5 escolheria o MLP. Mas o RF com threshold ótimo bate todos os outros (incluindo o MLP no threshold 0.5). A lição prática é que olhar somente para métricas tradicionais ou para custo num ponto fixo pode levar à decisão errada — o ranking depende do ponto de operação.

---


Três endpoints:

- `GET /` redireciona para o Swagger.
- `GET /health` retorna status da API e indica se o modelo está carregado. Sempre devolve HTTP 200, mesmo em modo degraded — fica claro para um load balancer que a API está viva mas pode estar sem modelo.
- `POST /predict` recebe as features de um cliente no formato cru do Telco (com `"No internet service"` e `"No phone service"` permitidos no schema, porque é o formato que o avaliador provavelmente vai testar). O endpoint normaliza esses casos para `Yes`/`No` antes de passar pelo pipeline, devolve a probabilidade de churn, a predição binária aplicando o threshold de 0.08, a versão do modelo, latência da inferência e um `request_id` rastreável nos logs.

Dois middlewares envolvem cada request. Um gera um UUID por request (ou ecoa o `X-Request-ID` que o cliente mandou) e expõe nos logs. O outro mede latência total e adiciona o header `X-Response-Time-Ms`. Os logs estruturados saem via `structlog`, no formato chave-valor, e dá para correlacionar uma predição específica com sua request lendo só por `request_id`.

Se o `models/final.joblib` não existir quando a API subir, ela ainda inicia. O `lifespan` captura o `FileNotFoundError`, loga, e segue. O `/health` então retorna `status: "degraded"` e o `/predict` devolve 503 com mensagem clara. Cenário pensado para deploy: é melhor ter um servidor no ar reportando o problema do que um que nem inicia.

Exemplo de chamada:

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

A seed `42` é utilizada em todos os pontos não-determinísticos: `train_test_split`, `RandomForestClassifier`, `KFold`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`. O Random Forest é determinístico por construção dado o seed. O MLP também é, dentro do loop de treino adotado (sem dropout estocástico em inferência, dataloaders sem shuffle nas avaliações).

O tracking dos experimentos fica em `mlruns/` no próprio repositório (filesystem backend do MLflow). Cada run tem tags, parâmetros, métricas (CV mean/std + test) e o modelo serializado como artifact. Para abrir a UI:

```bash
poetry run mlflow ui
```

A interface fica em `http://127.0.0.1:5000`.

---


## Limitações conhecidas

A maior é a ausência de cross-validation no MLP. Os três modelos sklearn passaram por `StratifiedKFold(k=5)` completa via helper de `mlflow.sklearn`. No MLP, fazer CV manual em PyTorch significaria rodar k loops de treino com early stopping, somar custo computacional, e o ganho de informação seria pequeno dado que o MLP não venceu nenhuma métrica primária no test set. 

O threshold de 0.08 é específico do split de test deste projeto. Em produção, deveria ser revalidado periodicamente — se a distribuição dos inputs mudar (drift) ou se os custos reais de FN e FP forem diferentes do assumido, esse ponto se move. O plano em `docs/monitoring.md` detalha como detectar isso.

Os custos de erro (R$ 500 e R$ 50) são premissas. O cálculo real dependeria de CAC, LTV, margem por cliente e outros números internos da operadora, que não foram disponibilizados. A lógica do framework continua válida com qualquer outro par de valores — basta refazer a curva, e o threshold sai diferente.

E o dataset é uma snapshot. Não há como fazer split temporal nem avaliar drift histórico, o que limita qualquer argumento sobre estabilidade do modelo ao longo do tempo. 


## Stack

Python 3.12, Poetry 2, pandas 2, scikit-learn 1.8, PyTorch 2.11, MLflow 3.11, FastAPI, Pydantic 2, pandera, pytest, ruff, structlog. Versões pinadas em `pyproject.toml`.

---

## Para mais detalhes

- `docs/model_card.md` — Model Card completo, com performance por subgrupo, vieses identificados e cenários de falha conhecidos
- `docs/ml_canvas.md` — ML Canvas com stakeholders, métricas de negócio e SLOs
- `docs/architecture.md` — Decisão de deploy (real-time vs batch) e justificativa
- `docs/monitoring.md` — Plano de monitoramento, métricas, alertas, playbook de resposta a incidentes