# CurvantML

Framework de machine learning para prever, **momentos antes** de o motorista entrar numa
curva, o quão arriscada ela vai ser — usando dados de sensores veiculares OBD (GPS,
acelerômetro, velocidade, RPM).

A previsão é **antecipada**: todas as features saem de uma janela *antes* da curva, e a
janela termina com uma folga (`lead_gap`, 30 m) da entrada — simulando um sistema que avisa
o motorista com antecedência.

## Instalação

```bash
pip install -e .
```

## Estrutura

```
curvant-ML/
├── run.py                  # ponto de entrada (todas as flags)
├── config.yaml             # todos os parâmetros, numa fonte só
├── curvant/
│   ├── constants.py        # constantes físicas (g, μ, limiares de ISL)
│   ├── pipeline.py         # orquestra as etapas
│   ├── utils/              # config, preprocessing, filters
│   ├── driving/            # curve_detection, risk_measures, features, isl
│   └── models/             # tabular, temporais, multitarefa, montecarlo
├── data/                   # parquet (brutos, *_clean e cache)
├── results/<alvo>/         # saídas, uma subpasta por alvo
└── docs/TARGETS_E_FEATURES.md   # referência detalhada de targets e features
```

## Como rodar

Cada flag escolhe **o que prever**; cada uma escreve seus resultados em `results/<alvo>/`.
Sem flag, o `run.py` mostra a ajuda.

```bash
# Limpeza dos dados brutos (auto-detecta o arquivo principal em data/)
python run.py --preprocessar

# Alvos
python run.py --risco          # classifica a curva em Segura/Risco
python run.py --risco --otimizar   # + ajuste de hiperparâmetros (Optuna)
python run.py --isl            # classifica o ISL (3 classes) + baseline físico
python run.py --velocidade     # prevê a velocidade crítica e deriva o ISL
python run.py --aceleracao     # regressão das acelerações dentro da curva
python run.py --multitarefa    # MLP PyTorch: ISL + manobras juntos
python run.py --importancia    # importância das features (XGBoost)

# Utilidades (combináveis)
python run.py --isl --plot         # + gráficos (matrizes de confusão, scatter)
python run.py --velocidade --rebuild   # ignora o cache e reprocessa as etapas 1–5
```

As etapas 1–5 (detecção de curvas, caracterização, features) são cacheadas em
`data/.cache_*.parquet`. Use `--rebuild` ao mudar parâmetros do `config.yaml` que afetam
detecção ou extração de features.

## O que cada flag prevê

| Flag | Alvo | Tipo |
|---|---|---|
| `--risco` | curva Segura/Risco (`manobra_combinado_curva`) | binário |
| `--isl` | classe de ISL (baixo/médio/alto) + baseline físico | 3 classes |
| `--velocidade` | velocidade crítica `v_critica` (e ISL derivado pela física) | regressão |
| `--aceleracao` | picos de aceleração dentro da curva | regressão |
| `--multitarefa` | `isl_max` + as 4 manobras, num modelo só | regressão + binários |

> **Direção atual do projeto:** o caminho mais promissor é o `--velocidade` — prever a
> velocidade crítica e calcular o ISL pela física. É onde o modelo supera de fato um chute
> simples (o erro cai de ~11 para ~7 km/h sobre o baseline de persistência).

A lista completa de alvos e das 49 features (agrupadas em F1–F5 + Monte Carlo), com o que
cada coluna significa, está em **[docs/TARGETS_E_FEATURES.md](docs/TARGETS_E_FEATURES.md)**.

## Pipeline

```
dados brutos (.parquet)
   → preprocessing      (utils/preprocessing.py)   limpa ruído, corta gaps
   → curve_detection    (driving/curve_detection.py) B-spline → curvatura → raio → DNIT
   → risk_measures      (driving/risk_measures.py)  rótulos de manobra (Segura/Risco)
   → features           (driving/features.py)       features por curva + Monte Carlo
   → models             (models/*.py)               treino e avaliação
```

### Janela pré-curva

As features saem de uma janela espacial logo antes da curva. O tamanho é fixo
(`features.janela_distancia`, 50 m) ou dinâmico pela distância de frenagem de conforto
`d = v̄²/(2·a_c)`, com clip em `[janela_distancia_min, janela_distancia_max]`. A janela
termina `lead_gap` metros **antes** da entrada (predição antecipada) e exclui pontos de uma
curva anterior.

### Caracterização de risco (3 critérios)

Para cada segmento contíguo de `curva=True`, três critérios independentes geram os rótulos:

| Critério | Coluna | Definição |
|---|---|---|
| Limite de aderência (Kamm) | `manobra_accel_curva` | `max √(aₓ²+a_y²) > α·μ·g` |
| Aceleração lateral | `manobra_lateral_curva` | `max \|a_y\| > 2.0` e curva DNIT ≥ média |
| Zigue-zague | `manobra_ziguezague_curva` | ≥ 3 mudanças de bearing alternadas com aceleração centrípeta |

`manobra_combinado_curva` é o OR dos três — o alvo padrão do `--risco`.

## Modelos

- **Tabular** (`--risco`, `--isl`, `--aceleracao`): LogisticRegression, SVM, DecisionTree,
  RandomForest, XGBoost, MLP. Pipeline `SMOTE → StandardScaler → [PCA] → modelo`, com split e
  validação cruzada **por rota** (`GroupKFold`/`StratifiedGroupKFold`) para não vazar entre
  gravações.
- **Optuna** (`--risco --otimizar`): tuning bayesiano de XGBoost, RandomForest e LogReg.
- **Temporais** (`--velocidade`): operam sobre a **sequência bruta** da janela pré-curva
  (não estatísticas). Modelos em `temporais.model`: `linear`, `rf`, `xgboost` (tabulares
  sobre estatísticas da sequência) e `mlp`, `lstm`, `gru`, `cnn1d` (neurais).
- **Multitarefa** (`--multitarefa`): MLP PyTorch com um encoder compartilhado e 5 heads.

### Baseline físico

Sob `--isl` e `--velocidade`, o pipeline reporta um **baseline sem aprendizado** ao lado dos
modelos — a simulação de Monte Carlo (argmax para `isl_class`) e a persistência (`v_critica`
≈ velocidade de aproximação). Serve para medir o quanto o ML realmente agrega.

## Configuração (`config.yaml`)

Tudo num arquivo só, com cada seção correspondendo a um módulo. As principais:

| Seção | Para que serve |
|---|---|
| `physics` | constantes do ISL: `mu` (atrito) e os limiares `isl_baixo`/`isl_alto` |
| `preprocessing` | limpeza dos dados brutos |
| `curve_detection` | B-spline, `limite_raio`, classes DNIT |
| `risk_measures` | limiares dos critérios de risco (Kamm, lateral, zigue-zague) |
| `features` | janela pré-curva, `lead_gap`, sensores |
| `montecarlo` | simulação de probabilidade de ISL |
| `ml` | parâmetros compartilhados dos modelos (split, CV, PCA, cap de outliers) |
| `optuna` | nº de trials do tuning |
| `multitarefa` | hiperparâmetros do MLP PyTorch |
| `temporais` | modelos, alvo e hiperparâmetros do `--velocidade` |

As constantes físicas (`g`, `μ`, limiares de ISL) ficam centralizadas em
[curvant/constants.py](curvant/constants.py), que lê `μ` e os limiares da seção `physics` do
config — então dá para variar o atrito (ex.: asfalto molhado) sem editar código.

## Dados

Dataset público no HuggingFace:
[`jwsouza13/routes_ML_inmetro`](https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro)

| Conjunto | Veículo | Trecho | `loc_coleta` |
|---|---|---|---|
| ELETRONUCLEAR | Spin / Van | Rio de Janeiro | `eletronuclear` |
| RJ-DF | Nivus | Rio de Janeiro → Brasília | `rjdf` |
| SERRA | Jetta | trecho serrano | `serra` |
| RJMGBA / JANEIRO | — | rotas adicionais | `rjmgba`, `janeiro` |

**Sensores universais** (presentes em todos os conjuntos): `vehicle_speed`, `engine_rpm`,
`accel_x`, `accel_y`, `lat`, `lon`. Sensores ausentes em parte dos dados (throttle,
rotation_rate, fuel_rate) e `accel_z` são excluídos.
