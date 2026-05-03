# Model Card — Churn Predictor (Random Forest balanceado)

Define o que monitorar em produção, com quais limiares, e como responder. Estrutura baseada na Aula 05 do ciclo de vida e ancorada nos SLOs de `docs/ml_canvas.md`.

---

## 1. Detalhes do modelo

| Campo | Valor |
|---|---|
| Nome | `random_forest_balanced_v1` |
| Tipo | Random Forest Classifier (binário) |
| Versão | 1.0 |
| Framework | scikit-learn 1.8 |
| Artefato | `models/final.joblib` (~22 MB) |
| Tracking | MLflow local, run `random_forest_balanced` |

Pipeline: `ColumnTransformer` (`OneHotEncoder(handle_unknown="ignore")` + `StandardScaler`) → `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`. O colapso de `"No internet service"`/`"No phone service"` para `"No"` é aplicado dentro do endpoint `/predict`, antes da inferência.

---

## 2. Uso pretendido

Estimar a probabilidade de cancelamento de um cliente individual da operadora. Alimenta priorização de campanhas de retenção, segmentação por risco e forecasting agregado. Consumidor primário: equipes de Customer Success, Retenção e BI, via API REST.

**Fora de escopo:** decisão automática sem revisão humana; populações fora do perfil Telco IBM (B2B, móvel pura, outros mercados); inferência causal; latência sub-10ms.

---

## 3. Dados de treinamento

Telco Customer Churn (IBM, via Kaggle). 7.043 clientes × 21 colunas (1 ID, 19 features, 1 target binário). Pré-processamento (notebook `01_eda.ipynb`, salvo em `data/interim/telco_clean.parquet`):

- `TotalCharges` veio como `object` por 11 linhas com espaço em branco (clientes `tenure=0`). Convertidas para `0.0`.
- 7 colunas com valores redundantes (ex.: `OnlineSecurity="No internet service"` ↔ `InternetService="No"`). Colapsados para `"No"`.

Split estratificado por `Churn` (`test_size=0.2`, `random_state=42`): 5.634 treino, 1.409 teste, ~26,5% positiva em ambos. Limitações dos dados: snapshot temporal única, mercado norte-americano residencial, atributos sensíveis presentes, sem dados de uso real.

---

## 4. Métricas de avaliação

PR-AUC é primária pelo desbalanceamento (26,5% positiva). Resultados no test set (n=1.409):

| Modelo            | CV PR-AUC (k=5)   | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
|-------------------|-------------------|-------------|--------------|---------|----------------|-------------|
| dummy             | 0.265 ± 0.000     | 0.265       | 0.500        | 0.000   | 0.000          | 0.000       |
| logreg            | 0.660 ± 0.020     | 0.634       | 0.842        | 0.616   | 0.527          | 0.741       |
| **random_forest** | **0.644 ± 0.028** | **0.641**   | 0.838        | 0.607   | 0.519          | 0.729       |
| mlp               | (sem CV)          | 0.636       | 0.843        | 0.626   | 0.541          | 0.745       |

**Custo total esperado** com `FN = R$ 500` e `FP = R$ 50` (razão 10:1):

| Modelo / threshold       | Custo total   | vs ótimo do RF |
|--------------------------|---------------|----------------|
| dummy / 0.5              | R$ 187.000    | +384%          |
| logreg / 0.5             | R$ 54.300     | +40%           |
| random_forest / 0.5      | R$ 76.250     | +97%           |
| mlp / 0.5                | R$ 51.650     | +34%           |
| **random_forest / 0.08** | **R$ 38.650** | baseline       |

---

## 5. Análise quantitativa por subgrupo

Performance do RF no threshold 0.08, desagregada por atributos sensíveis e categorias de negócio.

| gender  | n   | Churn rate | Recall | Precision |
|---------|-----|-----------|--------|-----------|
| Female  | 695 | 26,9%     | 0,73   | 0,52      |
| Male    | 714 | 26,1%     | 0,72   | 0,52      |

`gender` é não-discriminante (Cramér's V 0,01); performance equivalente, como esperado.

| SeniorCitizen  | n     | Churn rate | Recall | Precision |
|----------------|-------|-----------|--------|-----------|
| 0 (não-senior) | 1.171 | 23,7%     | 0,71   | 0,52      |
| 1 (senior)     | 238   | 41,6%     | 0,76   | 0,57      |

Seniors têm churn ~75% maior, mas o efeito vem majoritariamente de variáveis correlacionadas (`MonthlyCharges`, ausência de `Partner`/`Dependents`, contrato mensal). Sem viés problemático identificável.

| InternetService | n   | Churn rate | Recall |
|-----------------|-----|-----------|--------|
| No              | 305 | 7,2%      | 0,55   |
| DSL             | 487 | 18,9%     | 0,69   |
| Fiber optic     | 617 | 41,8%     | 0,77   |

Fibra concentra o risco; modelo recupera bem (Cramér's V 0,32, feature mais informativa).

| Contract       | n   | Churn rate | Recall |
|----------------|-----|-----------|--------|
| Month-to-month | 776 | 42,9%     | 0,79   |
| One year       | 305 | 11,1%     | 0,53   |
| Two year       | 328 | 2,7%      | 0,33   |

Mensal é o segmento de maior risco e o modelo prioriza acertos lá. Recall cai em bianual porque a classe positiva é raríssima — informação escassa, não falha de modelo.

---

## 6. Limitações

- **Sem CV no MLP.** Os três modelos sklearn passaram por `StratifiedKFold(k=5)`. CV manual em PyTorch teria custo alto e ganho pequeno dado que o MLP não venceu nenhuma métrica primária. Decisão pragmática.
- **Threshold específico do split.** O 0.08 vale para este test. Em produção, revalidar mensalmente; drift ou mudança nos custos reais move o ponto ótimo. Plano em `docs/monitoring.md`.
- **Premissas de custo.** R$ 500 (FN) e R$ 50 (FP) são ilustrativos. A lógica vale com qualquer outro par; CAC/LTV/margem reais redesenham a curva.
- **Sem variável temporal.** Snapshot única. Inviabiliza split temporal, análise de coorte e drift histórico.
- **Calibração.** RF tende a probabilidades concentradas em valores intermediários. Para threshold fixo, irrelevante; se a probabilidade virar insumo direto, aplicar `CalibratedClassifierCV`.
- **Segmentos pequenos.** Métricas em subgrupos com n<30 têm IC largo, sinal fraco.

---

## 7. Considerações éticas

`gender` e `SeniorCitizen` estão entre as features. `gender` é não-discriminante e a performance disagregada é equivalente. `SeniorCitizen` tem churn genuinamente maior e o modelo o usa — tratar a probabilidade como insumo, não gatilho automático de oferta por idade. Monitoramento inclui fairness em ambos.

Ações de retenção devem ter revisão humana, especialmente em alto valor. A API expõe probabilidade, predição, threshold e versão — auditável caso a caso. `customerID` é descartado antes do treino. Em produção, seguir política de retenção e LGPD. RF é caixa cinza: importâncias globais nativas; SHAP é evolução natural para explicabilidade local.