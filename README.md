# CurvantML

Framework de machine learning para prever condução de risco em curvas, momentos antes do motorista entrar no trecho curvo, utilizando dados de sensores veiculares OBD (GPS, acelerômetro, velocidade, RPM).

## Instalação

```bash
pip install -e ".[dev]"
```

## Como rodar

```bash
# 1. Limpeza dos dados brutos (executar uma vez, ou ao mudar parâmetros de pré-processamento)
python scripts/preprocess_data.py --input data/eletro_rjdf_serra_rjmgba_janeiro.parquet

# 2. Pipeline completo
python scripts/run.py                            # modelos clássicos (padrão), Modo 1
python scripts/run.py --classical                # modelos clássicos (explícito)
python scripts/run.py --plot                     # + gráficos e matrizes de confusão
python scripts/run.py --isl                      # + ISL 3 classes + regressão P1
python scripts/run.py --optuna                   # + Optuna (XGBoost e RandomForest)
python scripts/run.py --pytorch                  # + MLP multi-tarefa PyTorch
python scripts/run.py --mlp                      # + MLP sklearn (GridSearchCV)
python scripts/run.py --ts                       # + modelos de série temporal (dados brutos)
python scripts/run.py --ts --rebuild             # força reprocessamento das etapas 1–5
python scripts/run.py --modo modo2               # Modo 2: sem geometria da curva seguinte
python scripts/run.py --isl --pytorch --ts --plot  # combinação completa
```

As etapas de pré-processamento (1–5) são cacheadas automaticamente em `data/.cache_*.parquet` após a primeira execução. Use `--rebuild` para invalidar o cache (necessário ao mudar parâmetros de `config.yaml` que afetam a detecção de curvas ou extração de features).

## Dois modos de pipeline

O pipeline opera em dois modos que diferem na disponibilidade da geometria real da curva seguinte:

| Modo | Features F4 | Caso de uso |
|------|-------------|-------------|
| `modo1` (padrão) | ✓ incluídas: `f4_raio_min`, `f4_raio_mean`, `f4_dnit_num` | **Rota conhecida** — trace GPS completo pré-mapeado |
| `modo2` | ✗ zeradas (NaN → 0) | **Rota desconhecida** — apenas OBD + posição atual |

### O que são as features F4

As features F4 descrevem a **geometria real da curva que o veículo está prestes a entrar** — raio mínimo, raio médio e classe DNIT — extraídas do traço GPS completo do trajeto. Elas representam informação que só existe quando a rota foi percorrida antes (navegação, frotas com rotas fixas, análise offline).

No **Modo 2**, essas três colunas são zeradas antes do treinamento e da inferência: o modelo opera apenas com o comportamento do motorista na abordagem (features F1–F3 e F5) e a geometria estimada localmente (F3), sem conhecer o raio da próxima curva.

### Um modelo, dois contextos

Um único modelo é treinado com os dados de ambos os modos, usando `modo_rota_conhecida` (1 = Modo 1, 0 = Modo 2) como feature explícita. Isso permite que o modelo:

- **aprenda a explorar F4** quando a geometria está disponível (Modo 1), produzindo previsões mais precisas;
- **recaia graciosamente** sobre as features de comportamento quando F4 = 0 (Modo 2), sem precisar de um modelo separado.

O custo do Modo 2 é uma queda esperada no desempenho de classificação — o modelo passa a depender inteiramente de sinais como velocidade de entrada, jerk e histórico de curvas anteriores para antecipar o risco.

## Métodos de caracterização de risco

### Visão geral

O pipeline classifica o comportamento do motorista aplicando três critérios independentes a cada **segmento contíguo de `curva=True`** detectado. Para cada segmento, os critérios são avaliados sobre uma janela que inclui os pontos do segmento mais uma aproximação de $\tau_{\text{apx}} = 5\ \text{s}$ imediatamente anteriores à entrada. O rótulo resultante é atribuído **apenas aos pontos do segmento** — pontos entre curvas recebem `Segura` por definição.

### Critério 1 — Limite de aderência (Círculo de Kamm)

Detecta qualquer instante em que o vetor de aceleração total excede uma fração $\alpha$ do limite de aderência disponível, sem depender de GPS.

$$C_{\text{accel}} = \max_{t \in W} \sqrt{a_x(t)^2 + a_y(t)^2} > \alpha \cdot \mu \cdot g$$

onde $\mu = 0{,}6$ (atrito asfalto seco), $g = 9{,}81\ \text{m/s}^2$ e $\alpha = 0{,}7$ (margem de segurança), resultando num limiar de $\approx 4{,}12\ \text{m/s}^2$. A formulação decorre diretamente do **Círculo de Kamm**: a força total de atrito disponível é $\mu g$, compartilhada entre aceleração longitudinal ($a_x$) e lateral ($a_y$) — qualquer combinação que ultrapasse $\alpha \mu g$ indica operação próxima ao limite físico do pneu. O critério usa apenas `accel_x` e `accel_y` do OBD, sem geometria da curva.

### Critério 2 — Aceleração lateral excessiva

Detecta força centrífuga anormal percebida pelo acelerômetro lateral, condicionada à severidade geométrica da curva segundo a classificação DNIT.

$$C_{\text{lateral}} = \bigl(\max_{t \in W}|a_y(t)| > a_{y,\lim}\bigr) \;\wedge\; \bigl(\text{risco}_{\text{DNIT}} \geq 2\bigr)$$

com $a_{y,\lim} = 2{,}0\ \text{m/s}^2$. A condição DNIT $\geq 2$ restringe o critério a curvas de raio $R \leq 200\ \text{m}$ (classes *média*, *fechada* e *muito fechada*), evitando falsos positivos em retas ou curvas suaves onde a aceleração lateral é esperada.

**Classificação DNIT** (Manual de Projeto Geométrico de Rodovias Rurais, DNIT 2010):

| Classe | Raio | Grau de curva $D = 1145{,}92/R$ | $\text{risco}_{\text{DNIT}}$ |
|---|---|---|---|
| Muito fechada | $R \leq 50\ \text{m}$ | $D > 22{,}9°$ | 4 |
| Fechada | $R \leq 100\ \text{m}$ | $D > 11{,}5°$ | 3 |
| Média | $R \leq 200\ \text{m}$ | $D > 5{,}7°$ | 2 |
| Aberta | $R \leq 500\ \text{m}$ | $D > 2{,}3°$ | 1 |
| Suave | $R > 500\ \text{m}$ | — | 0 |

O raio $R$ é estimado pela curvatura de Frenet via ajuste de B-spline com suavização gaussiana adaptativa ($\sigma \approx 20\ \text{m} / \overline{d}$, clipado em $[1, 8]$).

### Critério 3 — Zigue-zague

Detecta oscilações laterais da trajetória combinando variação de direção (bearing GPS) com aceleração centrípeta, indicando instabilidade direcional ou desvios de faixa.

Seja $e_i = (\Delta\theta_i,\ |v^2/R|_i)$ o $i$-ésimo evento qualificado, onde $\Delta\theta_i$ é a variação de bearing **com sinal** (positivo = direita, negativo = esquerda) corrigida para a ambiguidade circular. Define-se evento qualificado quando:

$$|\Delta\theta_i| > \theta_{\lim} \;\wedge\; \left|\frac{v^2}{R}\right|_i > a_{c,\lim}$$

O critério de zigue-zague exige $n_{\min}$ eventos qualificados com **alternância de sinal** entre consecutivos:

$$C_{\text{zz}} = \#\bigl\{i : \text{evento qualificado} \;\wedge\; \operatorname{sgn}(\Delta\theta_i) \neq \operatorname{sgn}(\Delta\theta_{i-1})\bigr\} \geq n_{\min}$$

com $\theta_{\lim} = 15°$, $a_{c,\lim} = 0{,}3\ \text{m/s}^2$ e $n_{\min} = 3$. A exigência de alternância distingue zigue-zague (esquerda–direita–esquerda) de curvas contínuas no mesmo sentido. A aceleração centrípeta $v^2/R$ é calculada diretamente dos dados OBD+GPS sem assumir $\mu$.

### Rótulo composto e agregação por curva

O critério combinado é a disjunção lógica dos três, avaliado sobre a janela $W_k$ do segmento $k$:

$$C_{\text{combinado}}(W_k) = C_{\text{accel}}(W_k) \vee C_{\text{lateral}}(W_k) \vee C_{\text{zz}}(W_k)$$

O rótulo da curva é então:

$$y_{\text{curva}} = \begin{cases} \text{Risco} & \text{se } C_{\text{combinado}}(W_k) = \text{True} \\ \text{Segura} & \text{caso contrário} \end{cases}$$

A avaliação é conservadora — basta um critério disparar na janela do segmento para classificar a curva como Risco. Isso é adequado para aplicações de segurança, onde falsos negativos (curvas perigosas classificadas como seguras) são mais custosos que falsos positivos.

### ISL — Índice de Segurança Lateral

O ISL quantifica a proximidade ao limite de aderência lateral (Jessen et al., 2010):

$$\text{ISL} = \frac{v^2}{R \cdot g \cdot \mu}$$

onde $v$ é a velocidade (m/s), $R$ o raio de curvatura (m), $g = 9{,}81\ \text{m/s}^2$ e $\mu = 0{,}6$ (coeficiente de atrito asfalto seco). Raios menores que $R_{\min} = 5\ \text{m}$ são clipados para evitar ISL $\to \infty$ por ruído GPS/B-spline.

| Classe | ISL | Interpretação |
|---|---|---|
| `baixo` | $\text{ISL} < 0{,}5$ | Ampla margem de segurança |
| `medio` | $0{,}5 \leq \text{ISL} < 0{,}8$ | Atenção recomendada |
| `alto` | $\text{ISL} \geq 0{,}8$ | Próximo ao limite de aderência lateral |

O ISL é calculado nos pontos com `curva=True` dentro de cada curva detectada e excluído das features de entrada dos modelos para evitar leakage.

### Sumário das caracterizações

| Coluna | Fonte | Tipo | Critério |
|---|---|---|---|
| `manobra_accel_curva` | OBD | Binário | $\max\sqrt{a_x^2+a_y^2} > \alpha\mu g \approx 4{,}1\ \text{m/s}^2$ em alguma janela |
| `manobra_lateral_curva` | OBD + GPS | Binário | $\max|a_y| > 2{,}0\ \text{m/s}^2$ e DNIT $\geq$ média |
| `manobra_ziguezague_curva` | GPS | Binário | $\geq 3$ eventos de bearing + accel centrípeta |
| `manobra_combinado_curva` | — | Binário | OR dos três acima (target principal dos classificadores) |
| `isl_class` | OBD + GPS | 3 classes | ISL $= v^2/(R g \mu)$, classes baixo/médio/alto |
| `isl_max` | OBD + GPS | Regressão | Valor máximo de ISL dentro da curva (contínuo) |
| `v_excess` | OBD + GPS | Binário | $v_{\text{entry}} > v_{\text{safe}}(R) = \sqrt{Rg\mu}\cdot 3{,}6$ |

## Janelas de análise

### Caracterização ancorada ao segmento de curva

A análise de risco não usa partições temporais fixas. Para cada segmento contíguo de `curva=True` detectado, a avaliação dos três critérios é feita sobre a janela:

$$W_k = \{\,t : t_k^{\text{início}} - \tau_{\text{apx}} \leq t < t_k^{\text{fim}}\,\}$$

onde $t_k^{\text{início}}$ e $t_k^{\text{fim}}$ são os limites temporais do segmento $k$ e $\tau_{\text{apx}} = 5\ \text{s}$ é a janela de aproximação (configurável). Os rótulos resultantes são atribuídos **apenas aos pontos dentro do segmento** — pontos entre curvas recebem `Segura` por definição. Isso garante que os critérios são sempre avaliados em contexto geometricamente relevante e que nenhuma curva é dividida por um limite de janela arbitrário.

### Janela pré-curva por distância de frenagem

As features são extraídas de uma janela espacial imediatamente anterior à entrada da curva. A extensão é calculada dinamicamente a partir da velocidade de aproximação $\bar{v}$ (média dos últimos 5 pontos antes da curva):

$$d_{\text{janela}} = \text{clip}\!\left(\frac{\bar{v}^2}{2\,a_c},\ d_{\min},\ d_{\max}\right)$$

com $a_c = 2{,}5\ \text{m/s}^2$ (desaceleração de conforto), $d_{\min} = 50\ \text{m}$ e $d_{\max} = 400\ \text{m}$. Todos os parâmetros são configuráveis em `config.yaml`. A formulação garante que a janela cresce com o quadrado da velocidade — à mesma taxa que a distância de frenagem real — capturando o contexto de decisão relevante independentemente da velocidade.

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

$$n_{\text{accel}} = \#\bigl\{t \in W : \sqrt{a_x(t)^2 + a_y(t)^2} > \alpha\mu g\bigr\}$$

$$n_{\text{lateral}} = \#\bigl\{t \in W : |a_y(t)| > a_{y,\lim}\bigr\}$$

com os mesmos limiares dos Critérios 1 e 2. Esses contadores capturam a intensidade e frequência do comportamento de risco na aproximação à curva, complementando as estatísticas escalares de F1.

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
Análise de condução      por segmento de curva + 5 s de abordagem:
  (characterization.py)    • manobra_accel   (Círculo de Kamm)
                           • manobra_lateral  (accel_y gated por DNIT)
                           • manobra_ziguezague (bearing alternado + accel centrípeta)
                           • manobra_combinado (OR dos três acima)
    │
    ▼
Extração de features     janela pré-curva dinâmica (d = v²/2aₒ):
  (features.py)
  F1 — Dinâmica          mean/std/median/max/min/slope/cv de
                         vehicle_speed, engine_rpm, accel_x, accel_y
                         + mean_tarde / slope_tarde (segunda metade da janela)
  F2 — Entrada na curva  v_entry, v_speed_drop, jerk_x/y_max/std
                         v_entry_sq_over_raio_est (proxy físico de ISL)
  F3 — Geometria estimada janela_raio_min/mean/last (sem leakage)
  F4 — Geometria real    f4_raio_min/mean, f4_dnit_num  ← Modo 1 apenas
  F5 — Contexto histórico n_curvas_antes, prop_perigosas_antes,
                         prev_raio_min/mean, prev_dnit_num
  + n_perigo_accel/lateral_janela  (contagem direta dos sensores)
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
  ├── --pytorch            MultiTaskMLP (encoder 256->128 + 4 heads)
  │   ├── head isl_class          3 classes  (CrossEntropyLoss)
  │   ├── head manobra_accel_curva           (BCELoss)
  │   ├── head manobra_lateral_curva         (BCELoss)
  │   └── head manobra_zz_curva              (BCELoss)
  │
  └── --ts                 Série temporal (dados brutos da janela pré-curva)
      ├── Neurais          GRU, LSTM, CNN1D, MLP
      │                    n_timesteps=50 pontos reamostrados; alvo normalizado por rota
      └── Clássicos        RF, XGBoost, linear (features estatísticas da janela)
                           target e task configuráveis em config.yaml
```

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

**Arquitetura:**

**Encoder compartilhado** — $\mathbf{x} \in \mathbb{R}^{n}$ (features F1–F5 padronizadas):

$$
\mathbf{z} = f_{\mathrm{enc}}(\mathbf{x}) \in \mathbb{R}^{128}, \qquad
f_{\mathrm{enc}}(\mathbf{x}) = \mathrm{Drop}_{0.3}\!\Bigl(\mathrm{ReLU}\bigl(W_2\,\mathrm{Drop}_{0.3}(\mathrm{ReLU}(W_1\,\mathrm{BN}(\mathbf{x})))\bigr)\Bigr)
$$

com $W_1 \in \mathbb{R}^{256 \times n}$ e $W_2 \in \mathbb{R}^{128 \times 256}$.

**Heads independentes** — aplicados sobre a representação $\mathbf{z}$:

$$
\hat{y}_{\mathrm{isl\_class}} = W_{\mathrm{isl}}^{(2)}\,\mathrm{ReLU}\!\left(W_{\mathrm{isl}}^{(1)}\,\mathbf{z}\right) \in \mathbb{R}^{3}
$$

$$
\hat{y}_{k} = \sigma\!\left(W_{k}^{(2)}\,\mathrm{ReLU}\!\left(W_{k}^{(1)}\,\mathbf{z}\right)\right) \in (0,1), \qquad k \in \{\mathrm{accel},\,\mathrm{lateral},\,\mathrm{zz}\}
$$

com $W_{\cdot}^{(1)} \in \mathbb{R}^{64 \times 128}$ e $W_{\cdot}^{(2)} \in \mathbb{R}^{d_{\mathrm{out}} \times 64}$ ($d_{\mathrm{out}} = 3$ para `isl_class`, $d_{\mathrm{out}} = 1$ para os demais).

**Loss total:**

$$L = \lambda_1\,\mathcal{L}_{\text{CE}}(\hat{y}_{\text{isl\_class}},\,y_{\text{isl\_class}}) + \sum_{k \in \{\text{accel, lateral, zz}\}} \lambda_k\,\mathcal{L}_{\text{BCE}}(\hat{y}_k, y_k)$$

com $\lambda_i = 1{,}0$ para todas as tarefas (configurável via `lambdas`). Um único `backward()` por batch propaga o gradiente de todas as tarefas pelo encoder compartilhado.

**Scheduler:** `ReduceLROnPlateau(patience=50, factor=0.5)` — reduz a LR à metade quando a loss não melhora por 50 épocas consecutivas. O patience longo evita reduções prematuras causadas por ruído de batch. Prever ISL e zigue-zague ao mesmo tempo força o encoder a aprender representações mais gerais, reduzindo overfitting. Tarefas correlacionadas (ex. ISL alto ↔ `manobra_accel`) reforçam mutuamente o gradiente do encoder. Uma única passagem retorna todas as estimativas de risco simultaneamente.

Curva de loss salva automaticamente em `results/pytorch_loss_multitask_mlp.pdf`.

## Configuração (`config.yaml`)

| Parâmetro | Seção | Padrão | Descrição |
|---|---|---|---|
| `limite_raio` | `curve_detection` | `150` | Raio máximo (m) para marcar `curva=True` |
| `kamm_alpha` | `driving_analysis` | `0.7` | Fração do limite de aderência — limiar $= \alpha\mu g \approx 4{,}12\ \text{m/s}^2$ |
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
| `model` | `time_series_regression` | `[rf, xgboost, linear]` | Modelos a treinar — lista ou string: `gru`, `lstm`, `cnn1d`, `mlp`, `linear`, `rf`, `xgboost` |
| `task` | `time_series_regression` | `auto` | `auto` (detecta pelo target), `regression` ou `classification` |
| `target` | `time_series_regression` | `isl_mean` | Variável alvo — regressão: `isl_mean`, `isl_max`, `curve_accel_y_max`, `curve_accel_y_mean`, `curve_abs_accel_max`; classificação: `manobra_combinado_curva`, `isl_class`, etc. |
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
| RJMGBA / JANEIRO | — | Rotas adicionais | `rjmgba`, `janeiro` |

**Features universais:** apenas sensores presentes em todos os datasets — `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y`, `lat`, `lon`. Sensores ausentes no Eletronuclear (throttle, rotation_rate, fuel_rate) e `accel_z` (aceleração vertical — ruído de suspensão, sem relação com risco em curva horizontal) são excluídos.
