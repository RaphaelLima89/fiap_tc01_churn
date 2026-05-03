# ML Canvas — Churn Predictor

Visão sistêmica do projeto, organizada nos blocos com métricas de negócio e SLOs conforme requisitos do Tech Challenge. A proposta aqui é o que o material sirva para alinhamento entre time técnico, produto e operações.

---

## 1. Proposta de valor

Reduzir perdas financeiras associadas ao cancelamento de clientes em uma operadora de telecomunicações, antecipando quem está em risco antes da decisão de cancelar. A premissa é que um cliente recuperado por ação proativa de retenção custa significativamente menos do que adquirir um novo cliente equivalente. O modelo entrega uma estimativa quantificável desse risco para cada cliente da base, viabilizando priorização e personalização de ações.

---

## 2. Stakeholders

| Stakeholder            | Papel                                                                                              | Interesse direto                                                  |
|------------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| Diretoria comercial    | Sponsor do projeto, define metas de retenção e budget de campanhas                                 | Redução do churn rate; ROI das ações de retenção                  |
| Equipe de retenção     | Consumidor primário das predições; opera as ações de contato e oferta                              | Listas priorizadas, com sinal acionável e baixo ruído             |
| BI / Analytics         | Avalia performance ex-post, calcula custo evitado, gera relatórios                                 | Dados estruturados de inferência, rastreabilidade, métricas claras |
| Time de engenharia ML  | Mantém modelo, API, monitoramento e plano de retreino                                              | Estabilidade, observabilidade, ciclo de vida bem definido         |
| Compliance / Jurídico  | Garante conformidade com LGPD e políticas internas de uso de dados                                 | Documentação, auditabilidade, ausência de viés problemático       |
| Cliente final          | Sujeito da predição (sem acesso direto ao modelo, mas afetado por suas consequências)              | Tratamento equitativo, comunicação respeitosa em ações ativas     |

---

## 3. Decisões suportadas

A predição alimenta três tipos de decisão, em ordem decrescente de criticidade:

- **Quem contatar primeiro em campanhas de retenção.** Lista ordenada por probabilidade de churn, com corte definido pelo budget da campanha. Decisão operacional, frequência semanal ou quinzenal.
- **Que tipo de oferta enviar.** Combinada com regras de negócio (ticket atual, tipo de plano, histórico), a probabilidade ajusta o investimento de retenção — clientes mais arriscados podem receber ofertas mais agressivas.
- **Análise de carteira e forecasting.** Agregada por segmento, a saída do modelo alimenta projeções de churn esperado em janelas de 30/60/90 dias.

Decisões automáticas

## 4. Tarefa de predição

- **Tipo:** classificação binária supervisionada.
- **Target:** `Churn ∈ {Yes, No}`, codificado como `1` para clientes que cancelaram no período de referência.
- **Entrada:** vetor de 19 features de um cliente individual (perfil contratual, serviços contratados, métricas de cobrança).
- **Saída primária:** probabilidade de churn em `[0, 1]`.
- **Saída derivada:** predição binária, obtida ao aplicar o threshold de decisão configurado (`0.08`, calibrado por análise de custo).

A predição é por cliente, não por evento. O modelo não responde "quando vai cancelar", apenas "qual a probabilidade de cancelar dado o estado atual".

---


## 5. Fontes de dados

### Treino e validação

Telco Customer Churn (IBM), distribuído via Kaggle. 7.043 clientes, 21 colunas, snapshot única. Limitações conhecidas — ausência de carimbo temporal, mercado norte-americano implícito, sem dados de uso real — estão detalhadas no Model Card.

### Inferência (em uma operação real)

A API espera receber as 19 features em formato JSON. Em integração com sistemas internos, esses dados viriam de:

- CRM (perfil, contrato, dependentes);
- Sistema de billing (charges, método de pagamento, paperless);
- Catálogo de serviços (internet, telefone, streaming, suporte).

---


## 6. Features

19 features de entrada, distribuídas em três grupos:

- **Demográficas:** `gender`, `SeniorCitizen`, `Partner`, `Dependents` (4).
- **Contratuais e de serviços:** `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` (13).
- **Financeiras:** `MonthlyCharges`, `TotalCharges` (2).

Da EDA, as features mais discriminantes (Cramér's V com `Churn`) são `Contract` (0,41), `InternetService` (0,32), `PaymentMethod` (0,30) e `tenure` na correlação numérica (-0,35). `gender` é não-discriminante (Cramér's V 0,01) e `customerID` é descartado antes do treino.

---


## 7. Construção do modelo

- **Pipeline:** `ColumnTransformer` (OneHot para categóricas, StandardScaler para numéricas) → `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`. Tudo serializado em `models/final.joblib`.
- **Comparação:** quatro modelos avaliados com `StratifiedKFold(k=5)` nos baselines sklearn (Dummy, Regressão Logística, Random Forest) e avaliação direta no test para o MLP em PyTorch.
- **Tracking:** MLflow local (`mlruns/`), experimento `churn-prediction`, com runs nomeados por modelo e artefatos versionados.
- **Cadência de retreino:** trimestral em condições normais; antecipado em caso de drift (PSI > 0.2) ou queda de PR-AUC > 5pp.
- **Recalibração de threshold:** acoplada ao retreino. Uma curva de custo é refeita com dados frescos e o ponto ótimo é atualizado.

---

## 8. Avaliação offline

| Métrica            | Por que medir                                                                                  |
|--------------------|------------------------------------------------------------------------------------------------|
| PR-AUC             | Métrica primária. Captura performance na classe positiva (churn), tolerante a desbalanceamento |
| ROC-AUC            | Comparação com literatura e baseline; menos sensível a desbalanceamento                        |
| F1 / Precision / Recall | Operacionais — precision indica eficiência da campanha, recall indica cobertura            |
| Custo total esperado | Métrica de negócio, integra FN e FP com pesos econômicos (R$ 500 e R$ 50)                    |

Threshold é tratado como hiperparâmetro de operação, otimizado por custo total no test, não por F1 ou outras métricas de classificação direta.



---

## 9. Inferência

- **Modo:** real-time, online, requisição síncrona.
- **Canal:** HTTP/JSON via FastAPI. Endpoint `POST /predict` em `http://<host>:8000/predict`.
- **Throughput esperado (escala TC):** dezenas de requests por segundo, single worker. Para volumes industriais, escalonamento horizontal via uvicorn workers + load balancer.
- **Latência típica** (medida em ambiente local de desenvolvimento): inferência interna ~5–10ms, request total ~20–40ms incluindo overhead de framework.
- **Fallback:** ausência de modelo coloca a API em modo degraded. `/health` retorna `200 + status: degraded`, `/predict` retorna `503` com mensagem explícita.

---

## 10. Métricas de negócio

| Métrica                                  | Fórmula                                                                              | Frequência   |
|------------------------------------------|--------------------------------------------------------------------------------------|--------------|
| Custo de churn evitado                   | (churn esperado sem modelo − churn realizado) × LTV médio − custo das ofertas        | Mensal       |
| ROI das campanhas de retenção            | (receita preservada − custo das campanhas) / custo das campanhas                     | Por campanha |
| Taxa de conversão da retenção            | clientes recuperados / clientes contatados                                           | Por campanha |
| Churn rate da base                       | clientes que cancelaram no período / base ativa no início do período                 | Mensal       |
| Custo por cliente recuperado             | custo total da campanha / número de recuperações                                     | Por campanha |

A métrica mais direta para avaliar o valor do modelo é o **custo de churn evitado**. Ela só pode ser calculada ex-post, depois que o ciclo de retenção fecha.

---

## 11. SLOs

Service Level Objectives da API em produção:

| Indicador                                 | Objetivo                                  | Janela de avaliação |
|-------------------------------------------|-------------------------------------------|---------------------|
| Disponibilidade do endpoint `/predict`    | ≥ 99,0%                                   | Mensal              |
| Latência p95 do `/predict`                | ≤ 200 ms                                  | Diário              |
| Latência p99 do `/predict`                | ≤ 500 ms                                  | Diário              |
| Taxa de erros 5xx                         | ≤ 0,5% das requisições                    | Diário              |
| PR-AUC offline em janela móvel de 30 dias | ≥ 0,60 (com label real após D+30)         | Mensal              |
| Divergência custo realizado vs esperado   | ≤ 15%                                     | Mensal              |
| PSI máximo por feature crítica            | ≤ 0,2                                     | Semanal             |


## 12. Avaliação em produção

- **Coleta de label real:** depende de ciclo de cancelamento da operadora. Tipicamente, 30 a 60 dias após a predição é possível confirmar se o cliente cancelou ou não, e atribuir o resultado à predição original via `request_id`.
- **Métricas técnicas em produção:** PR-AUC, ROC-AUC e F1 calculados em janela móvel de 30 dias com labels coletados.
- **Métricas operacionais:** latência (p50/p95/p99), throughput, taxa de erro — capturadas via headers (`X-Response-Time-Ms`) e logs estruturados.
- **Detecção de drift:** Population Stability Index (PSI) e Kolmogorov-Smirnov por feature, comparando distribuição em janela recente vs distribuição de treino.
- **Fairness:** disparidade de recall e precision entre subgrupos sensíveis (`gender`, `SeniorCitizen`), com alerta em divergência sustentada.
- **A/B testing:** não previsto neste escopo. Em evolução, comparação de campanhas com vs sem priorização do modelo permitiria isolar o impacto.

Detalhamento operacional em `docs/monitoring.md`.

---