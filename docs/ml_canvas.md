# ML Canvas — Churn Predictor

Visão sistêmica do projeto, com stakeholders, métricas de negócio e SLOs requisitados pelo Tech Challenge. Serve para alinhamento entre time técnico, produto e operações.

---

## 1. Proposta de valor

Reduzir perdas financeiras associadas ao cancelamento de clientes em uma operadora de telecom, antecipando quem está em risco antes da decisão de cancelar. Recuperar cliente custa significativamente menos do que adquirir um novo equivalente — a saída do modelo viabiliza priorização e personalização das ações de retenção.

---

## 2. Stakeholders

A diretoria comercial é sponsor (define metas e budget). A equipe de retenção consome as predições e opera as ações. BI/Analytics avalia performance ex-post e reporta. Engenharia de ML mantém modelo, API, monitoramento e ciclo de retreino. Compliance/Jurídico cobre LGPD e ausência de viés problemático. O cliente final é sujeito da predição, sem acesso direto, mas afetado pelas consequências — daí a exigência de tratamento equitativo e revisão humana em ações ativas.

---

## 3. Decisões suportadas

- **Quem contatar primeiro em campanhas.** Lista ordenada por probabilidade, corte definido pelo budget. Frequência semanal ou quinzenal.
- **Que tipo de oferta enviar.** Combinada com regras de negócio (ticket, plano, histórico), ajusta o investimento de retenção.
- **Análise de carteira e forecasting.** Agregada por segmento, projeta churn esperado em 30/60/90 dias.

Decisões automáticas que afetam materialmente o cliente exigem revisão humana (ver Model Card §7).

---

## 4. Tarefa de predição

Classificação binária supervisionada. Target `Churn ∈ {Yes, No}` codificado como `1` quando o cliente cancela. Entrada: 19 features de um cliente individual. Saída primária: probabilidade em `[0, 1]`; saída derivada: predição binária aplicando o threshold de decisão de `0.08`. Predição é por cliente, não por evento — não responde "quando vai cancelar".

---

## 5. Fontes de dados

Treino: Telco Customer Churn (IBM, via Kaggle), 7.043 clientes, snapshot única. Limitações no Model Card §3.

Em operação real, o payload viria de CRM (perfil, contrato, dependentes), sistema de billing (charges, método de pagamento) e catálogo de serviços (internet, telefone, streaming, suporte).

---

## 6. Features

19 features divididas em demográficas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), contratuais e de serviços (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) e financeiras (`MonthlyCharges`, `TotalCharges`).

Mais discriminantes na EDA (Cramér's V com `Churn`): `Contract` (0,41), `InternetService` (0,32), `PaymentMethod` (0,30); `tenure` na correlação numérica (-0,35). `gender` é não-discriminante (0,01); `customerID` é descartado antes do treino.

---

## 7. Construção do modelo

Pipeline `ColumnTransformer` (OneHot + StandardScaler) → `RandomForestClassifier` balanceado, serializado em `models/final.joblib`. Comparação com `StratifiedKFold(k=5)` nos baselines sklearn (Dummy, LogReg, RF) e avaliação direta no test para o MLP em PyTorch. Tracking no MLflow local (experimento `churn-prediction`). Cadência de retreino trimestral; antecipado em drift (PSI > 0.2) ou queda de PR-AUC > 5pp. Recalibração de threshold acoplada ao retreino.

---

## 8. Avaliação offline

| Métrica                  | Por que medir                                                                  |
|--------------------------|--------------------------------------------------------------------------------|
| PR-AUC                   | Primária. Captura performance na classe positiva; tolerante a desbalanceamento |
| ROC-AUC                  | Comparação com literatura; menos sensível a desbalanceamento                   |
| F1 / Precision / Recall  | Operacionais — precision = eficiência de campanha; recall = cobertura          |
| Custo total esperado     | Métrica de negócio, integra FN e FP com pesos econômicos                       |

Threshold é tratado como hiperparâmetro de operação, otimizado por custo no test, não por F1.

---

## 9. Inferência

Real-time online via FastAPI/Uvicorn, requisição síncrona HTTP/JSON em `POST /predict`. Throughput esperado em escala TC: dezenas de RPS, single worker. Latência típica em dev: inferência ~5–10ms, request total ~20–40ms. Ausência de modelo coloca a API em modo degraded — `/health` 200 com `status: degraded`, `/predict` 503.

---

## 10. Métricas de negócio

| Métrica                       | Fórmula                                                                 | Frequência   |
|-------------------------------|-------------------------------------------------------------------------|--------------|
| Custo de churn evitado        | (churn esperado sem modelo − churn realizado) × LTV − custo das ofertas | Mensal       |
| ROI das campanhas de retenção | (receita preservada − custo das campanhas) / custo das campanhas        | Por campanha |
| Taxa de conversão da retenção | clientes recuperados / clientes contatados                              | Por campanha |
| Churn rate da base            | clientes que cancelaram / base ativa no início do período               | Mensal       |
| Custo por cliente recuperado  | custo total da campanha / número de recuperações                        | Por campanha |

Métrica direta de valor: **custo de churn evitado**, calculável ex-post quando o ciclo de retenção fecha.

---

## 11. SLOs

| Indicador                                 | Objetivo                          | Janela  |
|-------------------------------------------|-----------------------------------|---------|
| Disponibilidade de `/predict`             | ≥ 99,0%                           | Mensal  |
| Latência p95 do `/predict`                | ≤ 200 ms                          | Diário  |
| Latência p99 do `/predict`                | ≤ 500 ms                          | Diário  |
| Taxa de erros 5xx                         | ≤ 0,5%                            | Diário  |
| PR-AUC offline em janela móvel de 30 dias | ≥ 0,60 (com label real após D+30) | Mensal  |
| Divergência custo realizado vs esperado   | ≤ 15%                             | Mensal  |
| PSI máximo por feature crítica            | ≤ 0,2                             | Semanal |

---

## 12. Avaliação em produção

- **Label real:** disponível ~30–60 dias após a predição, atribuído via `request_id`.
- **Métricas técnicas:** PR-AUC, ROC-AUC, F1 em janela móvel de 30 dias.
- **Operacionais:** latência p50/p95/p99, throughput, taxa de erro (logs + `X-Response-Time-Ms`).
- **Drift:** PSI e Kolmogorov-Smirnov por feature.
- **Fairness:** disparidade de recall/precision em `gender` e `SeniorCitizen`.

Detalhamento operacional em `docs/monitoring.md`.