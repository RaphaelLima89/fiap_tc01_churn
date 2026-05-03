# Arquitetura de deploy

Documento que justifica a estratégia escolhida e descreve o que foi implementado. Segue os conceitos observados na Aula 04 do ciclo de vida (batch, real-time, edge, streaming).

---

## Decisão

Quatro estratégias foram consideradas:

| Estratégia       | Adequação ao caso                                                                           |
|------------------|---------------------------------------------------------------------------------------------|
| Batch            | Resolveria a geração de listas, mas não a consulta interativa durante atendimento           |
| Real-time online | Atende ambos os padrões; lista priorizada vira um cliente HTTP iterando sobre a base        |
| Edge / on-device | Não aplicável — não há cliente final consumindo o modelo localmente                         |
| Streaming        | Excede a complexidade necessária; não há fluxo contínuo de eventos a processar              |

Escolha: **real-time online via API REST** (FastAPI). Cobre o uso interativo das equipes de retenção e o uso em lote (cliente HTTP em loop), reaproveita o ferramental exigido pelo enunciado e mantém a versão do modelo simétrica ao tracking do MLflow.


---

## Topologia

```mermaid
flowchart LR
    Client[Cliente HTTP<br/>Swagger / cURL / Integracao] --> API[FastAPI / Uvicorn]
    subgraph Processo
      API --> MW1[request_id middleware]
      MW1 --> MW2[latency middleware]
      MW2 --> Endpoint[/predict]
      Endpoint --> Schema[Pydantic CustomerFeatures]
      Schema --> Inference[inference layer]
      Inference --> Model[Pipeline sklearn<br/>final.joblib em memoria]
    end
    Logs[(structlog)] -.-> Endpoint
    Logs -.-> MW2
```

Três camadas em `src/churn_predictor/api/`:

- **Borda HTTP.** Uvicorn ASGI, opcionalmente atrás de um load balancer em produção.
- **Cross-cutting.** Dois middlewares: `request_id` (UUID por request, ecoa em `X-Request-ID`) e `latency` (mede via `perf_counter`, expõe `X-Response-Time-Ms`, loga estruturadamente). A ordem de registro coloca `request_id` mais externo, para que `latency` leia o ID já populado.
- **Domínio.** `schemas.py` valida o payload via Pydantic 2. `inference.py` mantém o modelo cacheado num singleton populado no `lifespan`. `main.py` aplica o threshold de 0.08 e trata o cenário degraded (HTTP 503 em `/predict` quando o modelo não carrega; `/health` continua respondendo 200).

---

## Versionamento e rollback

A versão em produção tem identidade em três pontos: o arquivo `models/final.joblib`, a constante `MODEL_VERSION` em `config.py` (ecoada na resposta), e a run correspondente no MLflow.

Fluxo de retreino: nova run → comparação com produção → exportação do `final.joblib` + atualização de `MODEL_VERSION` + commit + redeploy. Rollback é o caminho inverso. Para escala maior, evolução natural é o **MLflow Model Registry** com transições de estágio.

---


## Escalabilidade

Em ordem de complexidade:

1. Vertical (mais CPU);
2. Workers do uvicorn (cada um carrega ~22 MB do modelo);
3. Réplicas horizontais atrás de load balancer;
4. Model Server dedicado (MLflow Serving, BentoML, Triton) quando houver múltiplos modelos ou hot-swap.

Cache de predição é uma evolução barata se houver consultas repetidas em janela curta (chave: hash do payload normalizado).

---

## Segurança

Para o escopo acadêmico: validação estrita via Pydantic, `422` em payload inválido, rastreabilidade por `request_id` em logs estruturados. Em produção real, evoluções esperadas são autenticação (API key, OAuth2 ou JWT), TLS, rate limiting, auditoria opcional de inferências (com retenção limitada e respeito à LGPD) e sanitização de logs.

A superfície de exposição atual é pequena por construção: a API não armazena PII nem expõe identificadores fora dos logs.

