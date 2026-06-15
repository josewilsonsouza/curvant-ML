# CurvantML

Framework de machine learning para prever condução de risco em curvas, momentos antes do motorista entrar no trecho curvo, utilizando dados de sensores veiculares OBD (GPS, acelerômetro, velocidade, RPM).

## Instalação

```bash
pip install -e
```

## Preparação dos dados

O primeiro passo é gerar um arquivo parquet limpo para alimentar a pipeline.
Use `scripts/preprocess_data.py` para aplicar as regras de limpeza configuradas em `config.yaml`:

- filtrar acelerações muito altas (`accel_limite`, padrão 5.0 m/s²)
- remover velocidades físicas impossíveis (`vel_max`, padrão 150 km/h)
- consolidar paradas curtas (até `max_parados_consecutivos`, padrão 3 pontos)
- cortar trajetos quando há gaps longos no sensor (`max_gap`, padrão 30 s)
- descartar segmentos muito curtos (`min_pontos_segmento`, padrão 10)

```bash
python scripts/preprocess_data.py --input data/eletro_rjdf_serra_rjmgba_janeiro.parquet
```

O arquivo de saída padrão é o mesmo nome de entrada com sufixo `_clean.parquet`.

## Cache de intermediários

O pipeline principal (`scripts/run.py`) salva intermediários em cache na pasta `data/`:

- `data/.cache_df_analysis.parquet` - resultado da detecção de curvas e caracterização de risco
- `data/.cache_features_df.parquet` - features extraídas por curva

Na execução normal, o script carrega esses caches quando eles existem e são mais recentes que o arquivo de dados de entrada. Use `--rebuild` recriar o cache.

## Como rodar

```bash
# 1. Limpeza dos dados brutos (executar uma vez, ou ao mudar parâmetros de pré-processamento)
python scripts/preprocess_data.py --input data/eletro_rjdf_serra_rjmgba_janeiro.parquet

# 2. Pipeline completo
python scripts/run.py                            # modelos clássicos (padrão)
python scripts/run.py --classical                # modelos clássicos (explícito)
python scripts/run.py --plot                     # + gráficos e matrizes de confusão
python scripts/run.py --isl                      # + ISL 3 classes + regressão P1
python scripts/run.py --optuna                   # + Optuna (XGBoost e RandomForest)
python scripts/run.py --pytorch                  # + MLP multi-tarefa PyTorch
python scripts/run.py --mlp                      # + MLP sklearn (GridSearchCV)
python scripts/run.py --fi                       # + importância de features XGBoost
python scripts/run.py --ts                       # + modelos de série temporal (dados brutos)
python scripts/run.py --ts --rebuild             # força reprocessamento das etapas 1–5
python scripts/run.py --isl --pytorch --ts --plot  # combinação completa
```

As etapas de pré-processamento (1–5) são cacheadas automaticamente em `data/.cache_*.parquet` após a primeira execução. Use `--rebuild` para invalidar o cache (necessário ao mudar parâmetros de `config.yaml` que afetam a detecção de curvas ou extração de features).

### O que são as features F4

As features F4 descrevem a **geometria real da curva que o veículo está prestes a entrar** - raio mínimo, raio médio e classe DNIT - extraídas do traço GPS completo do trajeto. Elas representam informação que só existe quando a rota foi percorrida antes (navegação, frotas com rotas fixas, análise offline).

Atualmente, o código extrai essas features diretamente da curva detectada e não implementa alternância de modos de execução. A documentação antiga mencionava um fluxo em que F4 era zerado para simular rota desconhecida, mas essa lógica não está presente no fluxo principal atual.

As três colunas F4 são:
  - `f4_raio_min`
  - `f4_raio_mean`
  - `f4_dnit_num`

## Métodos de caracterização de risco

**Targets (Alvos)**
- **Tipo por curva (binário)**: `manobra_accel_curva`, `manobra_lateral_curva`, `manobra_ziguezague_curva`, `manobra_combinado_curva` (alias `manobra`). Cada alvo é 1 se o critério disparar em qualquer ponto do trecho curvo, 0 caso contrário. Estes alvos são computados por segmento contíguo `curva=True`.
- **ISL (física) - classificação 3 classes**: `isl_class` (baixo/medio/alto). Valores contínuos relacionados: `isl_value`, `isl_mean`, `isl_max` (usados como targets de regressão/diagnóstico, mas excluídos das features de treino para evitar leakage).
- **Regressão de aceleração**: `curve_accel_y_max`, `curve_accel_y_mean`, `curve_abs_accel_max`, `curve_abs_accel_mean` - máximos e médias de acelerações dentro da curva (targets de regressão P1).
- **Velocidade / risco cinemático**: `v_entry`, `v_pred_kinematica`, `v_entry_sq_over_raio_est`, `v_safe_dnit`, `v_entry_ratio` - usados como features e/ou targets dependentes do experimento (p. ex. excesso de velocidade na entrada).

>Observação: todos os targets relacionados à curva são agregados por curva (não por ponto) e derivados dos pontos com `curva=True`. Colunas computadas dentro da curva (ex.: `isl_*`, `curve_*`) estão listadas em `_COLS_EXCLUIR` para evitar vazamento de informação ao treinar modelos que usem somente a janela pré-curva.

### Visão geral

O pipeline classifica o comportamento do motorista aplicando três critérios independentes a cada **segmento contíguo de `curva=True`** detectado. Para cada segmento, os critérios são avaliados sobre uma janela que inclui os pontos do segmento mais uma aproximação de $\tau_{\text{apx}} = 5\ \text{s}$ imediatamente anteriores à entrada. O rótulo resultante é atribuído **apenas aos pontos do segmento** e pontos entre curvas recebem `Segura` por definição.

### Critério 1 - Limite de aderência (Círculo de Kamm)

Detecta qualquer instante em que o vetor de aceleração total excede uma fração $\alpha$ do limite de aderência disponível, sem depender de GPS.

$$C_{\text{accel}} = \max_{t \in W} \sqrt{a_x(t)^2 + a_y(t)^2} > \alpha \cdot \mu \cdot g$$

onde $\mu = 0{,}6$ (atrito asfalto seco), $g = 9{,}81\ \text{m/s}^2$ e $\alpha = 0{,}7$ (margem de segurança), resultando num limiar de $\approx 4{,}12\ \text{m/s}^2$. A formulação decorre diretamente do **Círculo de Kamm**: a força total de atrito disponível é $\mu g$, compartilhada entre aceleração longitudinal ($a_x$) e lateral ($a_y$) - qualquer combinação que ultrapasse $\alpha \mu g$ indica operação próxima ao limite físico do pneu. O critério usa apenas `accel_x` e `accel_y` do OBD, sem geometria da curva.

### Critério 2 - Aceleração lateral excessiva

Detecta força centrífuga anormal percebida pelo acelerômetro lateral, condicionada à severidade geométrica da curva segundo a classificação DNIT.

$$C_{\text{lateral}} = \left(\max_{t \in W}|a_y(t)| > a_{y,\lim}\right) \wedge \left(\text{risco}_{\text{DNIT}} \geq 2\right)$$

com $a_{y,\lim} = 2{,}0\ \text{m/s}^2$. A condição DNIT $\geq 2$ restringe o critério a curvas de raio $R \leq 200\ \text{m}$ (classes *média*, *fechada* e *muito fechada*), evitando falsos positivos em retas ou curvas suaves onde a aceleração lateral é esperada.

**Classificação DNIT** (Manual de Projeto Geométrico de Rodovias Rurais, DNIT 2010):

| Classe | Raio | Grau de curva $D = 1145{,}92/R$ | $\text{risco}_{\text{DNIT}}$ |
|---|---|---|---|
| Muito fechada | $R \leq 50\ \text{m}$ | $D > 22{,}9°$ | 4 |
| Fechada | $R \leq 100\ \text{m}$ | $D > 11{,}5°$ | 3 |
| Média | $R \leq 200\ \text{m}$ | $D > 5{,}7°$ | 2 |
| Aberta | $R \leq 500\ \text{m}$ | $D > 2{,}3°$ | 1 |
| Suave | $R > 500\ \text{m}$ | - | 0 |

O raio $R$ é estimado pela curvatura de Frenet via ajuste de B-spline com suavização gaussiana adaptativa ($\sigma \approx 20\ \text{m} / \overline{d}$, clipado em $[1, 8]$).

### Critério 3 - Zigue-zague

Detecta oscilações laterais da trajetória combinando variação de direção (bearing GPS) com aceleração centrípeta, indicando instabilidade direcional ou desvios de faixa.

Seja $e_i = (\Delta\theta_i,\ |v^2/R|_i)$ o $i$-ésimo evento qualificado, onde $\Delta\theta_i$ é a variação de bearing **com sinal** (positivo = direita, negativo = esquerda) corrigida para a ambiguidade circular. Define-se evento qualificado quando:

$$|\Delta\theta_i| > \theta_{\lim} \;\wedge\; \left|\frac{v^2}{R}\right|_i > a_{c,\lim}$$

O critério de zigue-zague exige $n_{\min}$ eventos qualificados com **alternância de sinal** entre consecutivos:

$$C_{\text{zz}} = \#\left\{i : \text{evento qualificado} \wedge \mathrm{sgn}(\Delta\theta_i) \neq \mathrm{sgn}(\Delta\theta_{i-1})\right\} \geq n_{\min}$$

com $\theta_{\lim} = 15°$, $a_{c,\lim} = 0{,}3\ \text{m/s}^2$ e $n_{\min} = 3$. A exigência de alternância distingue zigue-zague (esquerda–direita–esquerda) de curvas contínuas no mesmo sentido. A aceleração centrípeta $v^2/R$ é calculada diretamente dos dados OBD+GPS sem assumir $\mu$.

### Rótulo composto e agregação por curva

O critério combinado é a disjunção lógica dos três, avaliado sobre a janela $W_k$ do segmento $k$:

$$C_{\text{combinado}}(W_k) = C_{\text{accel}}(W_k) \vee C_{\text{lateral}}(W_k) \vee C_{\text{zz}}(W_k)$$

O rótulo da curva é então:

$$y_{\text{curva}} = \begin{cases} \text{Risco} & \text{se } C_{\text{combinado}}(W_k) = \text{True} \\ \text{Segura} & \text{caso contrário} \end{cases}$$

A avaliação é conservadora - basta um critério disparar na janela do segmento para classificar a curva como Risco. Isso é adequado para aplicações de segurança, onde falsos negativos (curvas perigosas classificadas como seguras) são mais custosos que falsos positivos.

### ISL - Índice de Segurança Lateral

O ISL quantifica a proximidade ao limite de aderência lateral (Jessen et al., 2010):

$$\text{ISL} = \frac{v^2}{R \cdot g \cdot \mu}$$

onde $v$ é a velocidade (m/s), $R$ o raio de curvatura (m), $g = 9{,}81\ \text{m/s}^2$ e $\mu = 0{,}6$ (coeficiente de atrito asfalto seco). Raios menores que $R_{\min} = 5\ \text{m}$ são clipados para evitar ISL $\to \infty$ por ruído GPS/B-spline.

| Classe | ISL | Interpretação |
|---|---|---|
| `baixo` | $\text{ISL} < 0{,}5$ | Ampla margem de segurança |
| `medio` | $0{,}5 \leq \text{ISL} < 0{,}8$ | Atenção recomendada |
| `alto` | $\text{ISL} \geq 0{,}8$ | Próximo ao limite de aderência lateral |

O ISL é calculado nos pontos com `curva=True` dentro de cada curva detectada e excluído das features de entrada dos modelos para evitar leakage.

### Sumário das features

| Coluna | Fonte | Tipo | Critério |
|---|---|---|---|
| `manobra_accel_curva` | OBD | Binário | $\max\sqrt{a_x^2+a_y^2} > \alpha\mu g \approx 4{,}1\ \text{m/s}^2$ em alguma janela |
| `manobra_lateral_curva` | OBD + GPS | Binário | $\max|a_y| > 2{,}0\ \text{m/s}^2$ e DNIT $\geq$ média |
| `manobra_ziguezague_curva` | GPS | Binário | $\geq 3$ eventos de bearing + accel centrípeta |
| `manobra_combinado_curva` | - | Binário | OR dos três acima (target principal dos classificadores) |
| `isl_class` | OBD + GPS | 3 classes | ISL $= v^2/(R g \mu)$, classes baixo/médio/alto |
| `isl_max` | OBD + GPS | Regressão | Valor máximo de ISL dentro da curva (contínuo) |
| `v_excess` | OBD + GPS | Binário | $v_{\text{entry}} > v_{\text{safe}}(R) = \sqrt{Rg\mu}\cdot 3{,}6$ |

## Janelas de análise

### Caracterização ancorada ao segmento de curva

A análise de risco não usa partições temporais fixas. Para cada segmento contíguo de `curva=True` detectado, a avaliação dos três critérios é feita sobre a janela:

$$W_k = \left\{ t : t_k^{\text{início}} - \tau_{\text{apx}} \leq t < t_k^{\text{fim}} \right\}$$

onde $t_k^{\text{início}}$ e $t_k^{\text{fim}}$ são os limites temporais do segmento $k$ e $\tau_{\text{apx}} = 5\ \text{s}$ é a janela de aproximação (configurável). Os rótulos resultantes são atribuídos **apenas aos pontos dentro do segmento** - pontos entre curvas recebem `Segura` por definição. Isso garante que os critérios são sempre avaliados em contexto geometricamente relevante e que nenhuma curva é dividida por um limite de janela arbitrário.

### Janela pré-curva por distância de frenagem

As features são extraídas de uma janela espacial imediatamente anterior à entrada da curva. A extensão é calculada dinamicamente a partir da velocidade de aproximação $\bar{v}$ (média dos últimos 5 pontos antes da curva):

$$d_{\text{janela}} = \text{clip}\!\left(\frac{\bar{v}^2}{2\,a_c},\ d_{\min},\ d_{\max}\right)$$

com $a_c = 2{,}5\ \text{m/s}^2$ (desaceleração de conforto), $d_{\min} = 50\ \text{m}$ e $d_{\max} = 400\ \text{m}$. Todos os parâmetros são configuráveis em `config.yaml`.

| $\bar{v}$ (km/h) | $d_{\text{janela}}$ (m) |
|---|---|
| 40 | 50 (mín) |
| 60 | 56 |
| 80 | 99 |
| 100 | 154 |
| 120 | 222 |
| 160 | 395 |

A janela é adicionalmente filtrada para excluir pontos com `curva=True`, evitando que comportamento dentro de uma curva anterior contamine as features da curva seguinte.

### Contagem de eventos de risco na abordagem

Dois contadores de evento são computados diretamente dos sensores na janela pré-curva, sem depender dos rótulos de caracterização:

$$n_{\text{accel}} = \#\left\{t \in W : \sqrt{a_x(t)^2 + a_y(t)^2} > \alpha\mu g\right\}$$

$$n_{\text{lateral}} = \#\left\{t \in W : |a_y(t)| > a_{y,\lim}\right\}$$

com os mesmos limiares dos Critérios 1 e 2. Esses contadores capturam a intensidade e frequência do comportamento de risco na aproximação à curva, complementando as estatísticas escalares de F1.


## Modelos e tuning

### Clássicos (padrão)
LogisticRegression, SVM, DecisionTree, RandomForest, XGBoost, MLP sklearn.
Pipeline: `SMOTE -> StandardScaler -> [PCA] -> modelo` com `GroupKFold` por rota.
SMOTE ativo apenas em alvos desbalanceados (`v_excess`); removido para alvos ~50/50.

### Optuna (`--optuna`)
Tuning bayesiano de XGBoost e RandomForest via Optuna. Parâmetros em `config.yaml`:
```yaml
optuna:
  n_trials_xgb: 50    # trials para XGBoost
  n_trials_rf: 30     # trials para RandomForest
  timeout: 300        # segundos máximos por otimização
```

### Série temporal (`--ts`)

Opera sobre a sequência bruta de pontos GPS+OBD da janela pré-curva, sem reduzir para estatísticas escalares. A janela é reamostrada para `n_timesteps=50` pontos uniformes; cada ponto tem os canais definidos em `sensors` (ex. `vehicle_speed`, `accel_x/y`, `engine_rpm`, `raio_curvatura`). Features escalares de `features_df` (como `f3_janela_raio_min`) são adicionadas como canais constantes ao longo da sequência via `scalares_extras`. Um canal `prev_{target}` (valor do target na curva anterior da mesma rota) é adicionado automaticamente como entrada autorregressiva.

O target e o tipo de tarefa são configuráveis em `config.yaml`:

```yaml
time_series_regression:
  model: [rf, xgboost, linear]   # ou: gru | lstm | cnn1d | mlp
  task: auto                      # auto | regression | classification
  target: isl_mean
```

**Modelos disponíveis:**

| Modelo | Tipo | Detalhes |
|---|---|---|
| `gru` | Neural | GRU recorrente, `hidden_size` camadas, `n_layers` profundidade |
| `lstm` | Neural | LSTM recorrente, mesma configuração do GRU |
| `cnn1d` | Neural | Convoluções 1D com canais `[32, 64]` + pooling global |
| `mlp` | Neural | MLP simples sobre sequência achatada |
| `rf` | Clássico | RandomForest sobre features estatísticas da sequência |
| `xgboost` | Clássico | XGBoost sobre features estatísticas da sequência |
| `linear` | Clássico | Ridge (regressão) ou LogisticRegression (classificação) |

O split treino/teste é feito por rota base com estratificação pelo mediano do target por rota. Para regressão, o target é normalizado por rota (StandardScaler) antes do treino e desnormalizado após a predição, para reduzir shift de distribuição entre trajetos.

### PyTorch multi-tarefa (`--pytorch`)

A ideia central é que todas as tarefas de previsão compartilham a mesma representação interna da curva. Um encoder aprende features úteis para todas as tarefas simultaneamente; cada head especializa essa representação para seu objetivo específico.

Curva de loss salva automaticamente em `results/pytorch_loss_multitask_mlp.pdf`.

## Configuração (`config.yaml`)

| Parâmetro | Seção | Padrão | Descrição |
|---|---|---|---|
| `limite_raio` | `curve_detection` | `150` | Raio máximo (m) para marcar `curva=True` |
| `kamm_alpha` | `driving_analysis` | `0.7` | Fração do limite de aderência - limiar $= \alpha\mu g \approx 4{,}12\ \text{m/s}^2$ |
| `janela_aproximacao` | `driving_analysis` | `5` | Segundos de abordagem incluídos na avaliação de cada curva |
| `limiar_accel_lateral` | `driving_analysis` | `2.0` | \|accel_y\|_max mínimo para `manobra_lateral` (m/s²) |
| `janela_tempo` | `features` | `15` | Fallback temporal da janela pré-curva (s), usado se `distancia_acumulada` ausente |
| `janela_distancia` | `features` | `null` | Distância fixa (m) antes da curva; `null` = distância dinâmica por frenagem |
| `janela_acel_confort` | `features` | `2.5` | Desaceleração de conforto $a_c$ (m/s²) para o cálculo dinâmico |
| `janela_distancia_min` | `features` | `50` | Distância mínima da janela dinâmica (m) |
| `janela_distancia_max` | `features` | `400` | Distância máxima da janela dinâmica (m) |
| `isl_max_cap_percentil` | `ml` | `99` | Remove outliers extremos de ISL antes de treinar |
| `pca_n_components` | `ml` | `null` | PCA após SMOTE+Scaler (`null` = desativado) |
| `epochs` | `neural_networks.multitask_mlp` | `400` | Épocas de treino PyTorch |
| `lr` | `neural_networks.multitask_mlp` | `0.0005` | Learning rate Adam |
| `n_trials_xgb` | `optuna` | `50` | Trials Optuna para XGBoost |
| `n_trials_rf` | `optuna` | `30` | Trials Optuna para RandomForest |
| `timeout` | `optuna` | `300` | Tempo máximo por otimização (s) |
| `model` | `time_series_regression` | `[rf, xgboost, linear]` | Modelos a treinar - lista ou string: `gru`, `lstm`, `cnn1d`, `mlp`, `linear`, `rf`, `xgboost` |
| `task` | `time_series_regression` | `auto` | `auto` (detecta pelo target), `regression` ou `classification` |
| `target` | `time_series_regression` | `isl_mean` | Variável alvo - regressão: `isl_mean`, `isl_max`, `curve_accel_y_max`, `curve_accel_y_mean`, `curve_abs_accel_max`; classificação: `manobra_combinado_curva`, `isl_class`, etc. |
| `sensors` | `time_series_regression` | `[vehicle_speed, accel_x, accel_y, engine_rpm, raio_curvatura]` | Canais temporais da janela pré-curva (cada coluna vira uma dimensão da sequência) |
| `scalares_extras` | `time_series_regression` | `[f3_janela_raio_min, f3_janela_raio_mean]` | Colunas de `features_df` adicionadas como canais constantes ao longo da sequência |
| `target_cap_percentil` | `time_series_regression` | `99` | Remove amostras com target acima desse percentil antes de treinar (`null` = sem filtro) |
| `n_timesteps` | `time_series_regression` | `50` | Comprimento fixo após reamostagem da janela pré-curva |
| `epochs` | `time_series_regression` | `400` | Épocas de treino (modelos neurais) |
| `hidden_size` | `time_series_regression` | `64` | Dimensão oculta do GRU/LSTM |
| `n_layers` | `time_series_regression` | `1` | Número de camadas recorrentes (GRU/LSTM) |
| `channels` | `time_series_regression` | `[32, 64]` | Canais das camadas CNN1D |
| `dropout` | `time_series_regression` | `0.3` | Taxa de dropout (modelos neurais) |

## Dados

Dataset público no HuggingFace: [`jwsouza13/routes_ML_inmetro`](https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro)

| Conjunto | Veículo | Trecho | `loc_coleta` |
|---|---|---|---|
| ELETRONUCLEAR | Spin / Van | Rio de Janeiro | `eletronuclear` |
| RJ-DF | Nivus | Rio de Janeiro -> Brasília | `rjdf` |
| SERRA | Jetta | Trecho serrano | `serra` |
| RJMGBA / JANEIRO | - | Rotas adicionais | `rjmgba`, `janeiro` |

**Features universais:** apenas sensores presentes em todos os datasets - `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y`, `lat`, `lon`. Sensores ausentes no Eletronuclear (throttle, rotation_rate, fuel_rate) e `accel_z` (aceleração vertical - ruído de suspensão, sem relação com risco em curva horizontal) são excluídos.
