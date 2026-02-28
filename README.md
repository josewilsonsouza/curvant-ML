# CurvantML

Framework de machine learning para **prever condução de risco em curvas**, momentos antes do motorista entrar no trecho curvo, utilizando dados de sensores veiculares OBD (GPS, acelerômetro, velocidade, RPM).

## Objetivo

Classificar manobras como **Segura** ou **Perigosa** com base em features extraídas da janela de tempo imediatamente anterior à entrada em cada curva, permitindo alertas preventivos ao motorista.

## Instalação

```bash
pip install -e ".[dev]"
```

## Uso rápido

```bash
# 1. Limpeza dos dados brutos (executar uma vez)
python preprocess_data.py

# 2. Pipeline completo de experimentos
python run_experiments.py                    # modelos clássicos
python run_experiments.py --plot             # + gráficos
python run_experiments.py --mlp              # + MLP sklearn (GridSearchCV)
python run_experiments.py --keras            # + Keras MLP / GRU / LSTM
```

## Pipeline

```
Dados brutos (OBD)
    │
    ▼
preprocess_data.py          limpeza de ruídos:
    │                         • clip de spikes do acelerômetro (|a| > 5 m/s²)
    │                         • remoção de velocidades impossíveis (> 150 km/h)
    │                         • thinning de pontos parados consecutivos
    │                         • divisão de trajetos em gaps temporais > 30 s
    ▼
eletro_rjdf_serra_clean.parquet
    │
    ▼
Detecção de curvas          curvatura de Frenet via B-spline cúbica
    │                         • classificação DNIT (muito_fechada → suave)
    │                         • sigma de suavização adaptativo por densidade GPS
    ▼
Análise de condução         janelas de 10 s rotuladas como Segura / Perigosa:
    │                         • aceleração/frenagem anormal
    │                         • direção perigosa (gated por risco DNIT ≥ média)
    │                         • zigue-zague
    ▼
Extração de features        janela de 10 s PRÉ-curva:
    │                         mean / std / median / max / min de
    │                         vehicle_speed, engine_rpm, accel_x, accel_y
    ▼
Modelos de ML
    ├── Clássicos: Regressão Logística, SVM, Árvore de Decisão,
    │             Floresta Aleatória, MLP sklearn
    └── Deep learning: MLP Keras, GRU, LSTM
```

## Dados

Dataset público no HuggingFace: [`jwsouza13/routes_ML_inmetro`](https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro)

Três conjuntos de coleta:

| Conjunto | Veículo | Trecho |
|---|---|---|
| ELETRONUCLEAR | Spin / Van | Rio de Janeiro |
| RJ-DF | Nivus | Rio de Janeiro → Brasília |
| SERRA | Jetta | Trecho serrano |

## Avaliação dos modelos

Para evitar data leakage, o pipeline de ML aplica:
1. Split estratificado nos dados brutos (antes de SMOTE / normalização)
2. SMOTE contido dentro de cada fold via `ImbPipeline` — amostras sintéticas nunca cruzam para a validação
3. Métricas reportadas: CV Acurácia, CV F1 (weighted), Acurácia, F1, Precisão e Recall no conjunto de teste

## Configuração (`config.yaml`)

Todos os parâmetros do pipeline estão centralizados em `config.yaml`. Não é necessário alterar código-fonte para ajustar limiares ou hiperparâmetros.

### `preprocessing`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `accel_limite` | `5.0` m/s² | Clip de spikes do acelerômetro |
| `vel_max` | `150.0` km/h | Velocidade máxima plausível (acima disso = erro de GPS) |
| `vel_min_parado` | `2.0` km/h | Abaixo disso o veículo é considerado parado |
| `max_parados_consecutivos` | `3` | Máximo de pontos parados consecutivos mantidos |
| `max_gap` | `30.0` s | Gap temporal que divide um trajeto em sub-trajetos |
| `min_pontos_segmento` | `10` | Sub-trajetos menores que isso são descartados |

### `curve_detection`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `sigma` | `2` | Suavização gaussiana da curvatura. Use `'auto'` para sigma adaptativo por densidade de pontos GPS |
| `limite_raio` | `100` m | Raio máximo para marcar um ponto como curva (`curva=True`). Aumentar captura curvas mais suaves; diminuir restringe a curvas fechadas |
| `dnit.muito_fechada` | `50` m | R ≤ 50 m → grau > 22,9° |
| `dnit.fechada` | `100` m | R ≤ 100 m → grau > 11,5° |
| `dnit.media` | `200` m | R ≤ 200 m → grau > 5,7° |
| `dnit.aberta` | `500` m | R ≤ 500 m → grau > 2,3° |

### `driving_analysis`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `janela_tempo` | `10` s | Duração de cada janela de classificação |
| `var_velocidade_max` | `15` km/h | Variação acumulada de velocidade que indica aceleração/frenagem anormal |
| `velocidade_max_direcao` | `30` km/h | Velocidade mínima para aplicar o critério de direção perigosa |
| `angulo_max_direcao` | `40.0°` | Variação líquida de bearing abaixo da qual a direção é considerada perigosa (Li et al., 2016: 0,7 rad ≈ 40°) |
| `zigue_zague.limiar_bearing` | `15°` | Mudança mínima de bearing entre pontos para contar como evento de zigue-zague |
| `zigue_zague.limiar_accel_lateral` | `0.3` m/s² | Aceleração lateral mínima para contar como evento de zigue-zague |
| `zigue_zague.min_mudancas` | `3` | Número mínimo de eventos para classificar a janela como zigue-zague |

> O critério de direção perigosa só é aplicado quando a classe DNIT da curva for ≥ `media` (R ≤ 200 m). Em curvas suaves ou abertas, alta velocidade sem grande variação de bearing é considerada normal.

### `features`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `janela_tempo` | `10` s | Janela de tempo antes do início da curva usada para extração de features |

### `ml`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `random_state` | `42` | Semente para reprodutibilidade |
| `test_size` | `0.3` | Proporção do conjunto de teste |
| `cv_folds` | `5` | Número de folds na validação cruzada estratificada |
| `use_smote` | `true` | Aplica SMOTE dentro de cada fold para balancear as classes |
| `pca_n_components` | `null` | PCA antes dos modelos: `null` = desativado, `0.95` = manter 95% da variância, `10` = 10 componentes fixos |

### `neural_networks`

Hiperparâmetros para MLP Keras, GRU e LSTM (camadas, dropout, epochs, batch size). Altere diretamente nesta seção sem modificar `src/models.py`.

## Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `data/eletro_rjdf_serra_clean.parquet` | Dataset limpo (gerado por `preprocess_data.py`) |
| `results/tab_result.tex` | Tabela LaTeX com critérios de condução por classe |
| `results/ml_resultados.tex` | Tabela LaTeX com métricas por modelo |
| `results/matriz_confusao_<modelo>.pdf` | Matriz de confusão (com flag `--plot`) |
