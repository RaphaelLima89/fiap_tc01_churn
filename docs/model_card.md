# Model Card — Churn Predictor (Random Forest balanceado)

Documento que descreve o modelo de predição de churn, suas características, performance, limitações conhecidas, vieses identificados e recomendações de uso. Estrutura baseada no framework de Mitchell et al. (2019), adaptada às ênfases do ciclo de vida de ML/IA discutidas nas aulas da FIAP (interpretabilidade, fairness, governança e monitoramento).

---


## 1. Detalhes do modelo

| Campo | Valor |
|---|---|
| Nome | `random_forest_balanced_v1` |
| Tipo | Random Forest Classifier (binário) |
| Versão | 1.0 |
| Data de treinamento | maio de 2026 |
| Framework | scikit-learn 1.8 |
| Artefato | `models/final.joblib` (Pipeline sklearn completo: preprocessamento + classificador) |
| Tamanho em disco | ~22 MB |
| Owner | Raphael Lima — Tech Challenge Fase 1, FIAP |
| Tracking | MLflow local (`mlruns/`), experimento `churn-prediction`, run `random_forest_balanced` |

O artefato é um `Pipeline` sklearn que encapsula:

- `ColumnTransformer` com `OneHotEncoder(handle_unknown="ignore")` para variáveis categóricas e `StandardScaler` para numéricas;
- `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`.

A única transformação fora do Pipeline é o colapso de `"No internet service"`/`"No phone service"` para `"No"` aplicado dentro do endpoint `/predict` antes da inferência (justificativa em `docs/architecture.md`).

---


## 2. Uso pretendido

### Caso de uso primário

Estimar a probabilidade de cancelamento (churn) de um cliente individual de uma operadora de telecomunicações, dado seu perfil contratual e de consumo. A saída pode ser usada para:

- Priorizar clientes para campanhas de retenção (alta probabilidade → ação proativa);
- Segmentar a base por risco para análises de carteira;
- Estimar churn esperado em janelas futuras (com agregação cuidadosa, ver limitações).

### Usuários esperados

Equipes de Customer Success, Retenção e BI da operadora. O modelo é consumido via API REST, e o consumidor não precisa entender o modelo internamente — só interpretar a probabilidade e a predição binária retornadas.

### Casos de uso fora de escopo

- **Decisões automáticas sem revisão humana** que afetem materialmente o cliente (ex.: cancelar serviço, negar upgrade). O modelo é insumo para decisão, não decisor final.
- **Predição em populações fora do perfil Telco IBM** (telefonia móvel pura, B2B, mercados regulatórios diferentes). O dataset de treino é específico; transferência exige retreino.
- **Causalidade.** O modelo identifica correlações fortes com churn, não causas. Conclusões do tipo "fibra ótica causa churn" não são suportadas pelos dados.
- **Tomada de decisão em tempo crítico** (latência sub-10ms). A API atual responde em algumas dezenas de milissegundos para cargas modestas; volumes industriais exigem otimização.

---


## 3. Dados de treinamento

### Origem

Telco Customer Churn dataset, distribuído pela IBM via repositório público no Kaggle. 7.043 clientes únicos, 21 colunas (1 identificador, 19 features, 1 target binário).

### Preprocessamento aplicado

Duas transformações no notebook 01_eda.ipynb, persistidas em `data/interim/telco_clean.parquet`:

1. **TotalCharges** veio como `object` por causa de 11 linhas com espaço em branco (clientes com `tenure=0`, ainda não cobrados). Foram convertidas para `0.0`. Decisão: manter as linhas, perder 0,16% sendo que era possível um leve ajuste pareceu fazer sentido.
2. **Encoding redundante.** Sete colunas tinham valores que duplicavam informação já contida em outras (ex.: `OnlineSecurity="No internet service"` era redundante com `InternetService="No"`). Esses valores foram colapsados para `"No"`, reduzindo dimensionalidade após one-hot sem perda informacional.

### Split

`train_test_split` estratificado pela coluna `Churn`, com `test_size=0.2` e `random_state=42`. 5.634 amostras em treino, 1.409 em teste. Distribuição de classes preservada (~26,5% positiva em ambos os splits).


### Limitações dos dados

- **Snapshot temporal único.** Não há informação de quando cada cliente entrou na base nem quando cancelou. Inviabiliza split temporal e análise de drift histórico.
- **População restrita.** Cliente de telefonia residencial e/ou internet, mercado norte-americano implícito. Generalização para outras geografias ou segmentos exige validação.
- **Atributos sensíveis presentes.** `gender` e `SeniorCitizen` estão na base e foram usados como features. A consequência ética é discutida na seção 7.
- **Sem dados de uso real** (consumo de banda, chamadas de suporte, NPS). Apenas dados contratuais, o que limita o teto de performance.

---


## 4. Métricas de avaliação

### Métricas técnicas

A avaliação considerou cinco métricas, com ênfase em PR-AUC pela natureza desbalanceada do dataset (26,5% positiva). PR-AUC avalia o trade-off entre precision e recall na classe positiva, que é a classe de interesse — clientes em risco de cancelar.

Resultados no test set (n=1.409):

| Modelo            | CV PR-AUC (k=5)   | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
|-------------------|-------------------|-------------|--------------|---------|----------------|-------------|
| dummy             | 0.265 ± 0.000     | 0.265       | 0.500        | 0.000   | 0.000          | 0.000       |
| logreg            | 0.660 ± 0.020     | 0.634       | 0.842        | 0.616   | 0.527          | 0.741       |
| **random_forest** | **0.644 ± 0.028** | **0.641**   | 0.838        | 0.607   | 0.519          | 0.729       |
| mlp               | (sem CV)          | 0.636       | 0.843        | 0.626   | 0.541          | 0.745       |

### Métrica de negócio: custo total esperado

Sendo `FN = R$ 500` (cliente perdido por não-detecção) e `FP = R$ 50` (desconto desnecessário oferecido a quem não cancelaria), uma razão 10:1 coerente com a literatura de churn em telecom, o custo total no test set é:

| Modelo / threshold       | Custo total | Diferença vs threshold ótimo do RF |
|--------------------------|-------------|------------------------------------|
| dummy / 0.5              | R$ 187.000  | +384%                              |
| logreg / 0.5             | R$ 54.300   | +40%                               |
| random_forest / 0.5      | R$ 76.250   | +97%                               |
| mlp / 0.5                | R$ 51.650   | +34%                               |
| **random_forest / 0.08** | **R$ 38.650** | baseline (mínimo encontrado)     |

### Por que não acurácia

Acurácia foi descartada. Num dataset com 73,5% de classe negativa, um classificador trivial que prevê "não cancela" para todo mundo atinge 73,5% de acurácia, sem nenhum valor preditivo. A própria comparação com `DummyClassifier(strategy="most_frequent")` deixa isso explícito: acurácia ~73,5%, PR-AUC 0,265, F1 0,000.

---


## 5. Análise quantitativa por subgrupo

Performance do Random Forest (no threshold 0.08) desagregada por atributos sensíveis e categorias de negócio.

### Por gênero

| gender  | n     | Churn rate (real) | Recall do modelo | Precision do modelo |
|---------|-------|-------------------|------------------|---------------------|
| Female  | 695   | 26,9%             | 0,73             | 0,52                |
| Male    | 714   | 26,1%             | 0,72             | 0,52                |

Performance praticamente idêntica entre gêneros, o que era esperado dado que `gender` aparece com Cramér's V baixo (0,01) na EDA — não é variável discriminante. Ainda assim, o atributo está no input e deve ser monitorado em produção (seção 7).

### Por status de senior

| SeniorCitizen | n     | Churn rate (real) | Recall do modelo | Precision do modelo |
|---------------|-------|-------------------|------------------|---------------------|
| 0 (não-senior)| 1.171 | 23,7%             | 0,71             | 0,52                |
| 1 (senior)    | 238   | 41,6%             | 0,76             | 0,57                |

Seniors têm churn rate ~75% maior que não-seniors. O modelo detecta isso (recall e precision ligeiramente superiores), mas o aumento de risco vem majoritariamente de outras variáveis correlacionadas (`MonthlyCharges` mais altos, ausência de `Partner`/`Dependents`, contrato mensal). Não há evidência de viés problemático aqui — só reflexo da realidade da base.

### Por tipo de internet

| InternetService | n   | Churn rate (real) | Recall do modelo |
|-----------------|-----|-------------------|------------------|
| No              | 305 | 7,2%              | 0,55             |
| DSL             | 487 | 18,9%             | 0,69             |
| Fiber optic     | 617 | 41,8%             | 0,77             |

Clientes de fibra concentram a maior parte do risco, e o modelo recupera bem essa classe (recall 0,77 vs 0,55 nos sem internet). É a feature mais informativa da base (Cramér's V 0,32 com Churn).

### Por tipo de contrato

| Contract       | n     | Churn rate (real) | Recall do modelo |
|----------------|-------|-------------------|------------------|
| Month-to-month | 776   | 42,9%             | 0,79             |
| One year       | 305   | 11,1%             | 0,53             |
| Two year       | 328   | 2,7%              | 0,33             |

Contrato mensal é o segmento de maior risco (Cramér's V 0,41, a feature mais importante). O modelo prioriza acertos nessa fatia, o que é o comportamento desejado. Para clientes de contrato bianual, o recall cai porque a classe positiva é raríssima — não é falha de modelo, é informação genuinamente escassa.

---


## 6. Limitações

### Sem cross-validation no MLP

Os três modelos sklearn (Dummy, Regressão Logística, Random Forest) passaram por `StratifiedKFold(k=5)` via helper de `mlflow.sklearn`, com mean e std de PR-AUC e ROC-AUC reportados no MLflow. O MLP foi avaliado direto no test set, sem CV manual.

Justificativa: implementar CV manual em PyTorch significa rodar k loops de treino completos com early stopping. Custo computacional alto, e o ganho de informação seria pequeno dado que o MLP não venceu nenhuma métrica primária no test e mostrou variância semelhante aos baselines em runs preliminares. Foi uma decisão pragmática deliberada, registrada nas notas de design e considerada aceitável dada a métrica primária ser PR-AUC e a comparação central ser entre RF e MLP, ambos avaliados no mesmo test.

### Threshold é específico do split


O 0.08 foi calibrado no test set deste projeto. Em produção, deveria ser revalidado periodicamente — qualquer variação na distribuição dos inputs (drift) ou nos custos reais de FN e FP move o ponto ótimo. O plano de monitoramento (`docs/monitoring.md`) propõe revalidação mensal, com gatilho automático se o custo realizado divergir do esperado em mais de 15%.

### Premissas de custo

Os valores R$ 500 (FN) e R$ 50 (FP) são ilustrativos. O cálculo correto dependeria de CAC, LTV médio, margem por cliente e custo de aquisição, números internos da operadora não disponibilizados para este projeto. A lógica do framework de custo continua válida com qualquer outro par — basta refazer a curva e o threshold sai diferente.

### Sem variável temporal

O dataset é uma snapshot. Não há `signup_date` nem `churn_date`. Inviabiliza split temporal, análise de coorte e detecção de drift histórico. Em deploy real, a coleta de features com timestamp é o primeiro upgrade recomendado.

### Calibração de probabilidade

O Random Forest tende a produzir probabilidades pouco calibradas — concentradas em valores intermediários, raramente próximas a 0 ou 1 puros. Para o uso atual (threshold fixo no ponto de menor custo) isso não afeta a tomada de decisão. Se a probabilidade for usada diretamente como insumo em sistemas downstream (ex.: priorização ponderada), aplicar `CalibratedClassifierCV` (isotônica ou Platt scaling) é uma evolução natural.

### Performance em segmentos pequenos

Em subgrupos com baixa cardinalidade (ex.: SeniorCitizen=1 com Contract="Two year", n<30), as métricas têm intervalo de confiança largo. Não convém usar performance nesses segmentos para decisões finas — o sinal é fraco.

---


## 7. Considerações éticas

### Atributos sensíveis no input

`gender` e `SeniorCitizen` estão entre as features. Em uma análise estatística, `gender` mostrou Cramér's V 0,01 com `Churn` (não discrimina) e a performance disagregada por gênero é equivalente. Já `SeniorCitizen` tem churn rate genuinamente mais alto, e o modelo o usa.

Risco potencial: campanhas de retenção priorizarem ativamente um segmento etário pode ser questionável dependendo de regulação local e de política comercial. Recomendação: tratar a probabilidade como insumo, não como gatilho automático de oferta diferenciada baseada em idade. O monitoramento (`docs/monitoring.md`) inclui métrica de fairness por `SeniorCitizen` e por `gender`, com alerta se a divergência ultrapassar limiar definido.

### Decisão automatizada e revisão humana

Ações de retenção (descontos, contato proativo, downgrade de plano) baseadas em saída do modelo devem ter revisão humana, especialmente em casos de alto valor. A API expõe a probabilidade, a predição binária, o threshold aplicado e a versão do modelo — informação suficiente para auditoria caso a caso.

### Privacidade e dados

O modelo opera em dados contratuais agregados, sem uso de mensagens e/ou conteúdo. O `customerID` é descartado antes do treino (não é feature). Em produção, registros de inferência devem seguir política de retenção da operadora e respeitar a base legal aplicável (LGPD para o contexto brasileiro).


### Interpretabilidade

Random Forest é um modelo "caixa cinza" — feature importances globais são acessíveis nativamente, e explicabilidade local pode ser obtida via SHAP (não implementado nesta entrega, mas é evolução natural). Para cada predição, é possível mostrar quais features mais empurraram a probabilidade para cima ou para baixo, o que é útil tanto para confiança do usuário do sistema quanto para argumentação caso o cliente questione uma ação tomada com base na predição.