# Plano de monitoramento

Define o que monitorar em produção, com quais limiares, e como responder. Estrutura baseada na Aula 05 do ciclo de vida e ancorada nos SLOs de `docs/ml_canvas.md`.

---



## Tipos de drift

- **Data drift.** Mudança na distribuição dos inputs. Detectado via PSI e Kolmogorov-Smirnov por feature.
- **Concept drift.** Mudança na relação entre input e target. Visível apenas via performance offline com label real (atraso de D+30 a D+60).
- **Prediction drift.** Mudança na distribuição da saída do modelo. Detectável imediatamente, sem depender de label.

---

## Métricas e limiares

| Sinal                                    | Threshold                       | Janela   | Severidade |
|------------------------------------------|---------------------------------|----------|------------|
| Disponibilidade `/predict`               | < 99,0%                         | 24h      | Alta       |
| Latência p99                             | > 500 ms                        | 1h       | Média      |
| Taxa de erros 5xx                        | > 0,5%                          | 1h       | Alta       |
| PSI por feature crítica                  | > 0,2 sustentado                | Semanal  | Média      |
| Drift de prediction (KS vs baseline)     | p-valor < 0,01 por 3 dias       | Diária   | Média      |
| PR-AUC offline em janela móvel           | < 0,60                          | 30 dias  | Alta       |
| Divergência custo realizado vs esperado  | > 15%                           | Mensal   | Alta       |
| Disparidade de recall por subgrupo       | > 0,1 absoluto entre extremos   | Mensal   | Média      |

Features críticas para PSI: `Contract`, `InternetService`, `MonthlyCharges`, `tenure`, `PaymentMethod`. Subgrupos sensíveis para fairness: `gender`, `SeniorCitizen`.

---

## Instrumentação

Já implementado: logs estruturados via `structlog` com `request_id`, `model_version`, `latency_ms`, `churn_probability` e `churn_prediction` por request, mais o header `X-Response-Time-Ms`.

Evoluções esperadas em produção: persistência das requests em data lake para análise de drift e performance offline; endpoint `/metrics` em formato Prometheus; relatórios periódicos de Evidently AI ou WhyLogs.

---

## Resposta a alertas

- **Operacional (latência, 5xx, disponibilidade):** restart do worker, rollback do deploy recente se houver correlação, escalonamento horizontal.
- **Drift de input ou predição:** investigar fonte; retreino agendado se sustentado por mais de dois ciclos.
- **Performance offline abaixo do limiar:** retreino com dados frescos e recalibração de threshold.
- **Divergência de custo:** revisar premissas de FN/FP; recalibrar threshold; retreinar se persistir.
- **Disparidade por subgrupo:** validar amostra; investigar viés de coleta; considerar reponderação ou retreino corretivo.

Em qualquer retreino: nova run no MLflow → comparação com produção → atualização de `MODEL_VERSION` → deploy. Rollback é o caminho inverso.

---

## Cadência

Diário: operacional e prediction drift. Semanal: PSI. Mensal: PR-AUC offline, custo realizado, fairness. Trimestral: retreino programado, antecipado em alerta de severidade alta sustentado.

---
