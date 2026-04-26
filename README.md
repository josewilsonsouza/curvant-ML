# CurvantML

Framework de machine learning para **prever condução de risco em curvas**, momentos antes do motorista entrar no trecho curvo, utilizando dados de sensores veiculares OBD (GPS, acelerômetro, velocidade, RPM).

## Instalação

```bash
pip install -e ".[dev]"
```

## Como rodar

```bash
# 1. Limpeza dos dados brutos (executar uma vez, ou ao mudar parâmetros de pré-processamento)
python scripts/preprocess_data.py --input data/eletro_rjdf_serra_rjmgba_janeiro.parquet

# 2. Pipeline completo
python scripts/run.py                            # modelos clássicos, Modo 1
python scripts/run.py --plot                     # + gráficos e matrizes de confusão
python scripts/run.py --isl                      # + ISL 3 classes + regressão P1
python scripts/run.py --optuna                   # + Optuna (XGBoost e RandomForest)
python scripts/run.py --pytorch                  # + MLP multi-tarefa PyTorch
python scripts/run.py --mlp                      # + MLP sklearn (GridSearchCV)
python scripts/run.py --modo modo2               # Modo 2: sem geometria da curva seguinte
python scripts/run.py --isl --pytorch --plot     # combinação completa
```

## Dois modos de pipeline

| Modo | Features F4 (geometria da curva) | Caso de uso |
|------|----------------------------------|-------------|
| `modo1` (padrão) | ✓ incluídas (`f4_raio_min`, `f4_raio_mean`, `f4_dnit_num`) | Rota conhecida — trace GPS completo disponível |
| `modo2` | ✗ zeradas (NaN → 0) | Rota desconhecida — apenas OBD + posição atual |

Um único modelo é treinado com `modo_rota_conhecida` (0/1) como feature, permitindo que aprenda a diferença de confiança entre os dois cenários.

## Taxonomia de risco

O pipeline produz **cinco caracterizações independentes** por curva, substituindo o rótulo único `manobra`:

| Caracterização | Grupo | Tipo | Critério |
|---|---|---|---|
| `manobra_accel` | OBD-only | Binário | Variação acumulada de velocidade > limiar (Li et al., 2016) |
| `manobra_ziguezague` | OBD-only | Binário | ≥ 3 mudanças bruscas de bearing + accel_lateral > limiar |
| `isl_value` / `isl_class` | GPS | Regressão / 3 classes | ISL = v² / (R·g·μ) — Jessen et al. (2010) |
| `manobra_lateral` | GPS | Binário | \|accel_y\|_max > limiar em curva DNIT ≥ média |
| `v_excess` | GPS | Binário | v_entry > v_safe(R) = √(R·g·μ) · 3,6 |
| `manobra_combinado` | — | Binário | True se qualquer um dos 5 acima for True (baseline comparativo) |

## ISL — Índice de Segurança Lateral

$$\text{ISL} = \frac{v^2}{R \cdot g \cdot \mu}$$

onde $v$ é a velocidade (m/s), $R$ o raio de curvatura (m), $g = 9{,}81\ \text{m/s}^2$ e $\mu = 0{,}6$ (asfalto seco).

| Classe | ISL | Interpretação |
|---|---|---|
| `baixo` | $< 0{,}5$ | Ampla margem de segurança |
| `medio` | $0{,}5 \leq \text{ISL} < 0{,}8$ | Atenção recomendada |
| `alto` | $\geq 0{,}8$ | Próximo ao limite de aderência |

## Pipeline

```
Dados brutos (OBD + GPS)
    │
    ▼
preprocess_data.py       clip de acelerômetro, filtro de velocidade,
                         thinning de paradas, divisão por gaps > 30 s
    │
    ▼
Detecção de curvas       curvatura de Frenet (B-spline), classificação DNIT,
                         colunas: curva, raio_curvatura, classe_dnit
    │
    ▼
Análise de condução      janelas de 15 s → taxonomia por critério:
  (characterization.py)    • manobra_accel   (variação de velocidade)
                           • manobra_lateral  (accel_y gated por DNIT)
                           • manobra_ziguezague (bearing + accel_lateral)
                           • manobra_combinado (OR dos três acima)
    │
    ▼
Extração de features     janela pré-curva (15 s ou distância configurável):
  (features.py)
  F1 — Dinâmica          mean/std/median/max/min/slope/cv de
                         vehicle_speed, engine_rpm, accel_x, accel_y, accel_z
                         + mean_tarde / slope_tarde (segunda metade da janela)
  F2 — Entrada na curva  v_entry, v_speed_drop, jerk_x/y/z_max/std
                         v_entry_sq_over_raio_est (proxy físico de ISL)
  F3 — Geometria estimada janela_raio_min/mean/last (sem leakage)
  F4 — Geometria real    f4_raio_min/mean, f4_dnit_num  ← Modo 1 apenas
  F5 — Contexto histórico n_curvas_antes, prop_perigosas_antes,
                         prev_raio_min/mean, prev_dnit_num
  + modo_rota_conhecida  (1 = Modo 1, 0 = Modo 2)
    │
    ▼
Avaliação honesta        split e cross-validation por id_route
  (models.py)            GroupKFold(n_splits=5) — nenhuma rota aparece
                         em dois folds
    │
    ▼
Modelos
  ├── Clássicos (padrão)   LogisticRegression, SVM, DecisionTree,
  │                        RandomForest, XGBoost (+ Optuna via --optuna)
  │
  ├── --isl
  │   ├── ISL 3 classes    target: isl_class (baixo/medio/alto)
  │   └── P1 regressão     target: curve_accel_y_max / curve_abs_accel_max
  │
  └── --pytorch            MultiTaskMLP (encoder 256→128 + 5 heads)
      ├── head isl_value   regressão  (MSELoss)
      ├── head isl_class   3 classes  (CrossEntropyLoss)
      ├── head manobra_accel_curva    (BCELoss)
      ├── head manobra_lateral_curva  (BCELoss)
      └── head manobra_zz_curva       (BCELoss)
```

## Modelos e tuning

### Clássicos (padrão)
LogisticRegression, SVM, DecisionTree, RandomForest, XGBoost, MLP sklearn.
Pipeline: `SMOTE → StandardScaler → [PCA] → modelo` com `GroupKFold` por rota.
SMOTE ativo apenas em alvos desbalanceados (`v_excess`); removido para alvos ~50/50.

### Optuna (`--optuna`)
Tuning bayesiano de XGBoost e RandomForest via Optuna. Parâmetros em `config.yaml`:
```yaml
optuna:
  n_trials: 50
  timeout: 300    # segundos por otimização
```

### PyTorch multi-tarefa (`--pytorch`)
```
Input → BatchNorm → Linear(256) → ReLU → Dropout(0.3)
      → Linear(128) → ReLU → Dropout(0.3)
      → 5 heads independentes
```
Loss total: $L = \sum_i \lambda_i \cdot L_i$, com $\lambda_i = 1{,}0$ por padrão.
Curva de loss salva automaticamente em `results/pytorch_loss_multitask_mlp.pdf`.

## Configuração (`config.yaml`)

| Parâmetro | Seção | Padrão | Descrição |
|---|---|---|---|
| `limite_raio` | `curve_detection` | `150` | Raio máximo (m) para marcar `curva=True` |
| `janela_tempo` | `driving_analysis` | `15` | Segundos por janela de classificação |
| `limiar_accel_lateral` | `driving_analysis` | `2.0` | \|accel_y\|_max mínimo para `manobra_lateral` (m/s²) |
| `janela_tempo` | `features` | `15` | Segundos da janela pré-curva |
| `janela_distancia` | `features` | `null` | Metros antes da curva; tem precedência sobre `janela_tempo` |
| `isl_max_cap_percentil` | `ml` | `99` | Remove outliers extremos de ISL antes de treinar |
| `pca_n_components` | `ml` | `null` | PCA após SMOTE+Scaler (`null` = desativado) |
| `epochs` | `neural_networks.multitask_mlp` | `200` | Épocas de treino PyTorch |
| `lr` | `neural_networks.multitask_mlp` | `0.0005` | Learning rate Adam |
| `n_trials` | `optuna` | `50` | Trials Optuna por modelo/target |

## Dados

Dataset público no HuggingFace: [`jwsouza13/routes_ML_inmetro`](https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro)

| Conjunto | Veículo | Trecho | `loc_coleta` |
|---|---|---|---|
| ELETRONUCLEAR | Spin / Van | Rio de Janeiro | `eletronuclear` |
| RJ-DF | Nivus | Rio de Janeiro → Brasília | `rjdf` |
| SERRA | Jetta | Trecho serrano | `serra` |
| RJMGBA / JANEIRO | — | Rotas adicionais | `rjmgba`, `janeiro` |

**Features universais:** apenas sensores presentes em todos os datasets — `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y`, `accel_z`, `lat`, `lon`. Sensores ausentes no Eletronuclear (throttle, rotation_rate, fuel_rate) são excluídos.
