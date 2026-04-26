# CurvantML — Framework Multi-Caracterização de Risco em Curvas

**Data:** 2026-04-25
**Status:** Aprovado pelo usuário — pronto para implementação

---

## Contexto e Motivação

O curvantML prevê condução de risco em curvas usando dados OBD + GPS, momentos antes de o veículo entrar na curva. O pipeline atual produzia um único alvo binário (`manobra`) derivado de heurísticas (Li et al., 2016), com teto de ~70% F1 em todos os modelos — sinal claro de que o problema é o rótulo, não o modelo.

Três problemas centrais identificados:

1. **Rótulo único e ruidoso:** `manobra` combina três critérios heterogêneos em um binário, perdendo informação e criando ruído de label.
2. **Avaliação otimista:** split aleatório por amostra permite que dados da mesma rota apareçam em treino e teste — métricas infladas.
3. **Sem separação de modos:** o pipeline não distingue se a geometria da curva seguinte está disponível (rota conhecida) ou não (rota desconhecida), perdendo um diferencial importante.

---

## Objetivos do Redesign

1. Definir uma **taxonomia explícita de caracterizações de risco** dentro da curva
2. Implementar **dois modos de pipeline** (rota conhecida / desconhecida) com o mesmo modelo base
3. Adotar **ISL contínuo como alvo principal de regressão**, com classes derivadas por threshold
4. Migrar deep learning de Keras para **PyTorch** (multi-task, mais flexível)
5. Corrigir a avaliação para **split e cross-validation por `id_route`**

---

## Seção 1 — Taxonomia de Risco

### 1.1 Caracterizações

```
Caracterizações de risco dentro da curva
│
├── Grupo A — Independentes de geometria (OBD-only)
│   ├── manobra_accel       : variação acumulada de velocidade > limiar (Li et al.)
│   └── manobra_ziguezague  : ≥3 mudanças bruscas de bearing + accel_lateral > limiar
│
├── Grupo B — Dependem de R (derivado do GPS do trace)
│   ├── isl_value           : ISL contínuo = v² / (R · g · μ)  [alvo principal]
│   ├── isl_class           : baixo (ISL<0.5) / médio (0.5–0.8) / alto (≥0.8)
│   ├── manobra_lateral     : |accel_y|_max > limiar em curva DNIT ≥ média
│   └── v_excess            : v_entry > v_safe(R) = √(R · g · μ) · 3.6
│
└── manobra_combinado        : Perigosa se qualquer um dos 5 binários abaixo for True:
                               manobra_accel, manobra_ziguezague, isl_alto (isl_value≥0.8),
                               manobra_lateral, v_excess
                               (mantido apenas para comparação com literatura)
```

### 1.2 Embasamento em literatura

| Caracterização | Referência |
|---|---|
| manobra_accel, manobra_lateral, manobra_ziguezague | Li et al. (2016) |
| isl_value, isl_class | Física do veículo: ISL = v²/(Rgμ); Jessen et al. (2010) |
| v_excess | DNIT (2010), normas de velocidade segura por raio |
| manobra_combinado | Li et al. (2016) — baseline de comparação |

---

## Seção 2 — Dois Modos de Pipeline

### 2.1 Definição

**Modo 1 — Rota conhecida (trace GPS completo disponível):**
O trace GPS do percurso completo está disponível antes ou durante a viagem. A geometria da curva seguinte (raio real, classe DNIT) pode ser calculada com precisão e usada como feature de input.

**Modo 2 — Rota desconhecida (apenas dados até posição atual):**
O veículo está em posição (a,b) sem conhecimento do percurso seguinte. A geometria da curva deve ser estimada a partir da curvatura observada nos metros imediatamente anteriores à curva.

### 2.2 Diferença nos inputs

| Bloco de features | Modo 1 | Modo 2 |
|---|---|---|
| F1–F3, F5 (OBD + geometria estimada) | ✓ | ✓ |
| F4 (geometria real da curva seguinte) | ✓ | ✗ (NaN → imputado) |
| `modo_rota_conhecida` (0/1) | 1 | 0 |

Um único modelo é treinado com ambos os modos misturados. A feature `modo_rota_conhecida` permite que o modelo aprenda a diferença de confiança entre os dois cenários.

---

## Seção 3 — Feature Engineering

### 3.1 Sensores universais (presentes em todos os datasets)

- `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y`, `accel_z`, `lat`, `lon`

Throttle, rotation_rate, fuel_rate e demais sensores são ausentes no dataset Eletronuclear e **não são incluídos** no vetor de features.

### 3.2 Blocos de features

**F1 — Dinâmica pré-curva** (janela 10s antes da curva, ou `janela_distancia` metros):

Para cada sensor em {`vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y`, `accel_z`}:
- `{sensor}_mean`, `{sensor}_std`, `{sensor}_median`, `{sensor}_max`, `{sensor}_min`
- `{sensor}_slope` (tendência linear na janela completa)
- `{sensor}_cv` (coeficiente de variação)
- `{sensor}_mean_tarde`, `{sensor}_slope_tarde` (segunda metade da janela — captura frenagem pré-curva)

**F2 — Perfil de entrada na curva:**
- `v_entry`: velocidade no primeiro ponto da curva (km/h)
- `v_speed_drop`: v_max_janela − v_entry
- `v_speed_drop_pct`: v_speed_drop / v_max_janela
- `v_entry_vs_mean`: v_entry / vehicle_speed_mean
- `jerk_x_max`, `jerk_x_std`: taxa de variação de accel_x (Δaccel_x / Δt)
- `jerk_y_max`, `jerk_y_std`: taxa de variação de accel_y
- `jerk_z_max`, `jerk_z_std`: taxa de variação de accel_z

**F3 — Geometria estimada via GPS (sem leakage):**
- `janela_raio_min`, `janela_raio_mean`, `janela_raio_last`: raio de curvatura na janela pré-curva
- `v_entry_sq_over_raio_est`: v_entry² / janela_raio_min — proxy físico de ISL na entrada (feature física explícita)

**F4 — Geometria real da curva seguinte (Modo 1 apenas):**
- `curve_raio_min`, `curve_raio_mean`: raio mínimo e médio dentro da curva
- `curve_dnit_num`: classe DNIT codificada (0=suave … 4=muito_fechada)

**F5 — Contexto histórico do trajeto:**
- `n_curvas_antes`: número de curvas já percorridas naquele trajeto
- `prop_perigosas_antes`: proporção de curvas perigosas anteriores
- `prev_raio_min`, `prev_raio_mean`: geometria da curva anterior
- `prev_dnit_num`: classe DNIT da curva anterior

### 3.3 Janela pré-curva

- Padrão: `janela_tempo = 10` segundos
- Alternativa configurável: `janela_distancia` metros (parâmetro em `config.yaml`)
- Os dois modos são mutuamente exclusivos; `janela_distancia != null` tem precedência

---

## Seção 4 — Modelos

### 4.1 Modelos clássicos (baseline e produção)

Cada caracterização de risco tem seus próprios modelos treinados independentemente:

- **Classificação binária** (manobra_accel, manobra_lateral, manobra_ziguezague, v_excess): LogisticRegression, RandomForest, XGBoost
  - `isl_alto` (isl_value ≥ 0.8) é derivado de `isl_class` — não treinado como alvo separado
- **Classificação 3 classes** (isl_class): LogisticRegression, RandomForest, XGBoost
- **Regressão** (isl_value): Ridge, RandomForestRegressor, XGBRegressor

**Tuning com Optuna:** XGBClassifier, XGBRegressor, RandomForestClassifier, RandomForestRegressor.
**Sem tuning (baselines):** LogisticRegression, SVM, DecisionTree, Ridge, SVR — defaults como piso de comparação.

**SMOTE:** removido do binário balanceado (manobra_accel, manobra_lateral, manobra_ziguezague, isl_alto). Mantido para `v_excess` (muito desbalanceado: ~107 vs ~2526).

### 4.2 Deep learning — PyTorch

Keras/TensorFlow é substituído por PyTorch.

**Arquitetura: MLP Multi-Tarefa**

```
Input → Encoder compartilhado → Múltiplos heads
```

Encoder:
```
BatchNorm1d(n_features)
→ Linear(256) → ReLU → Dropout(0.3)
→ Linear(128) → ReLU → Dropout(0.3)
```

Heads (cada um com Linear(64) → ReLU → head-específico):
- `isl_value_head`:        → Linear(1)          [regressão, loss: MSELoss]
- `isl_class_head`:        → Linear(3)          [3 classes, loss: CrossEntropyLoss]
- `manobra_accel_head`:    → Linear(1) + Sigmoid [binário, loss: BCELoss]
- `manobra_lateral_head`:  → Linear(1) + Sigmoid [binário, loss: BCELoss]
- `manobra_zz_head`:       → Linear(1) + Sigmoid [binário, loss: BCELoss]

Loss total:
```
L = λ₁·MSE_isl + λ₂·CE_isl_class + λ₃·BCE_accel + λ₄·BCE_lateral + λ₅·BCE_zz
```

Os pesos λ são tunados via Optuna ou definidos por validação.

**Arquiteturas sequenciais (GRU/LSTM):**
Reescritas em PyTorch. `janela_tempo` explorável (ex: sequência das últimas 3 curvas como input). Unidades mínimas: GRU(64), LSTM(64) — os valores anteriores de 3 e 2 unidades eram claramente insuficientes.

### 4.3 Estrutura de arquivos novos

```
src/
  characterization.py   — substitui driving_analysis.py (taxonomia explícita de risco)
  features.py           — atualizado: accel_z, jerk_z, v_entry_sq_over_raio_est, modo_rota_conhecida
  models_pytorch.py     — novo: MLP multi-task, GRU, LSTM em PyTorch
  models.py             — mantido para clássicos; Optuna adicionado
  pipeline.py           — expõe modo_1 e modo_2, split por id_route
scripts/
  run.py                — flags --modo1 / --modo2 adicionadas
```

---

## Seção 5 — Avaliação

### 5.1 Split por `id_route`

```python
rotas = df['id_route'].unique()
train_rotas, test_rotas = train_test_split(rotas, test_size=0.3, random_state=42)
df_train = df[df['id_route'].isin(train_rotas)]
df_test  = df[df['id_route'].isin(test_rotas)]
```

Cross-validation: **GroupKFold(n_splits=5)** com `groups=id_route`. Nenhuma rota aparece em dois folds.

### 5.2 Métricas por tipo de alvo

| Tipo | Métricas |
|---|---|
| Binário | F1-weighted, Precisão, Recall, AUC-ROC |
| 3 classes (isl_class) | F1-macro, matriz de confusão |
| Regressão (isl_value) | R², MAE, RMSE |

### 5.3 Comparação Modo 1 vs Modo 2

Cada modelo é avaliado com e sem o bloco F4 (geometria real da curva). Isso produz a análise: "quanto a geometria da curva seguinte melhora a previsão?", que vira uma seção de resultados no paper.

---

## O que é descartado

| Item | Motivo |
|---|---|
| `manobra_combinado` como alvo principal de treino | Rótulo heurístico composto; mantido só como baseline/comparação |
| SMOTE em dados balanceados | Adiciona ruído em vez de ajudar |
| GRU(3,2) e LSTM pequenos | Claramente subpotentes |
| GridSearchCV com grade única | Substituído por Optuna |
| Split aleatório por amostra | Avaliação inflada; substituído por split por `id_route` |
| Keras/TensorFlow | Substituído por PyTorch |
| Features de sensores parciais (throttle, rotation_rate, fuel) | Ausentes em Eletronuclear; removidos para consistência universal |

---

## Critérios de sucesso

- ISL regressão: R² > 0.50 no Modo 1 (com geometria real)
- ISL classificação 3 classes: F1-macro > 0.65
- manobra_accel / manobra_ziguezague: F1 > 0.72 (OBD-only, sem depender de R)
- Avaliação honesta: split por `id_route` em todos os experimentos
- PyTorch multi-task treinando e convergindo sem NaN
