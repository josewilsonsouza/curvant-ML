<p align="center">
  <img src="figures/curvantML.png" alt="CurvantML" width="400">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-189FDD">
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-tuning-7B3FE4">
  <a href="https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro">
    <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-FFD21E">
  </a>
</p>

Queremos prever, momentos antes de o motorista entrar numa curva, se ele realizará uma condução segura ou de risco, a partir de dados de sensores veiculares OBD Link anteriores à curva.

## Instalação e execução
Clone este repositório:

```bash
git clone https://github.com/josewilsonsouza/curvant-ml.git
cd curvant-ml
```
Execute a instalação:

```powershell
pip install -e .
```

Agora execute a limpeza dos dados, executando

```bash
cvt --preprocess
```

Implementamos algumas variáveis que são de interesse prever antes de entrar na curva. As flags de predição abaixo escolhem o que prever; cada uma escreve seus resultados em `results/<alvo>/`. Sem flag, `cvt` mostra a ajuda.

```bash
cvt --risk           # classifica a condução na curva em Segura/Risco
cvt --velocity       # prevê a velocidade crítica e deriva o ISL
```

Os comandos abaixo **não** são predição — são utilidades combináveis com as flags de predição.

```bash
cvt --velocity --rebuild     # ignora o cache e reprocessa as etapas 1–5
cvt --velocity --no-plot     # pula os gráficos (mais rápido)
```

> Análises complementares do risco (importância de features e otimização Optuna) ficam em [`notebooks/analise_risco.ipynb`](notebooks/analise_risco.ipynb), fora da CLI.

> **📝Nota**. Use `--rebuild` ao mudar parâmetros de detecção de curvas (`config.yaml`) ou de extração de features (`features.yaml > extracao`).

As flags acima preveem o seguinte

| Flag | Alvo/Target | Tipo |
|---|---|---|
| `--risk` | curva Segura/Risco (`manobra_combinado_curva`) | binário |
| `--velocity` | velocidade crítica `v_critica` (e ISL derivado pela física) | regressão |

> **Direção atual do projeto:** o caminho mais promissor é o `--velocity`: prever a velocidade crítica e calcular o ISL pela física. É onde o modelo supera de fato um chute simples (o erro cai de ~11 para ~7 km/h sobre o baseline de persistência).

A lista completa de alvos e features (agrupadas por tipo), com o que cada coluna significa, está em **[TARGETS_E_FEATURES](docs/TARGETS_E_FEATURES.md)**. A seleção de quais features cada flag usa fica em **[features.yaml](features.yaml)** (whitelist por flag).

Veja na imagem a seguir o fluxo de exeução das flags e os targets.

```mermaid
graph LR
    A([cvt]) --> B{Modo de<br/>Execução}

    %% Preparação
    B -->|Preparação| C[--preprocess] --> D[(data/)]

    %% Modelagem Principal
    B -->|Modelagem| G((Alvos))

    G -->|--risk| H[Classificação Binária<br/>Segura/Risco]
    G -->|--velocity| J[Regressão Temporal<br/>v_critica]

    H & J --> M[(results/< alvo >/)]

    %% Modificadores
    N[[Flags Utilitárias]] -.->|--no-plot<br/>--rebuild| G
```

## Pipeline

O projeto segue o seguite pipeline.

```mermaid
graph LR
    A[(Dados Brutos)] --> B[utils/preprocessing.py]
    B --> C[driving/curve_detection.py]
    C --> D[driving/risk_measures.py]
    D --> E[driving/features.py]
    E --> F{{models/*.py}}
```


### Janela pré-curva

As features saem de uma janela espacial logo antes da curva. O tamanho é fixo (`features.janela_distancia`, 50 m) ou dinâmico pela distância de frenagem de conforto `d = v̄²/(2·a_c)`, com clip em ` janela_distancia_min, janela_distancia_max]`. A janela termina `lead_gap` metros **antes** da entrada (predição antecipada) e exclui pontos de uma curva anterior.

### Caracterização de risco (3 critérios)

Para cada segmento contíguo de `curva=True`, três critérios independentes geram os rótulos:

| Critério | Coluna | Definição |
|---|---|---|
| Limite de aderência (Kamm) | `manobra_accel_curva` | $\max_t \sqrt{a_x^2 + a_y^2} > \alpha\,\mu\,g$ |
| Aceleração lateral | `manobra_lateral_curva` | $\max_t \lvert a_y \rvert > 2{,}0$ e curva DNIT $\ge$ aberta |
| Zigue-zague | `manobra_ziguezague_curva` | $\ge 3$ mudanças de bearing alternadas com aceleração centrípeta |

`manobra_combinado_curva` é o OR dos três. Detalhes desses critérios estão em [RISK_MEASURES](docs/RISK_MEASURES.md).

## Modelos

- **Tabular** (`--risk`): LogisticRegression, SVM, DecisionTree, RandomForest, XGBoost, MLP. Pipeline `SMOTE -> StandardScaler -> [PCA] -> modelo`, com split e validação cruzada **por rota** (`GroupKFold`/`StratifiedGroupKFold`) para não vazar entre gravações.
- **Optuna** (`notebooks/analise_risco.ipynb`): tuning bayesiano de XGBoost, RandomForest e LogReg — análise complementar, fora da CLI.
- **Temporais** (`--velocity`): operam sobre a sequência bruta da janela pré-curva (não estatísticas). Modelos em `temporais.model`: `linear`, `rf`, `xgboost` (tabulares sobre estatísticas da sequência) e `mlp`, `lstm`, `gru`, `cnn1d` (neurais).

Sob `--velocity`, o pipeline reporta um baseline sem aprendizado ao lado dos modelos — a persistência (`v_critica` ~ velocidade de aproximação). Serve para medir o quanto o ML realmente agrega.

## Configuração

Dois arquivos na raiz: **`config.yaml`** (parâmetros de modelo/pipeline) e **`features.yaml`**
(quais features cada flag usa — whitelist por flag).

`config.yaml`, cada seção correspondendo a um módulo. As principais:

| Seção | Para que serve |
|---|---|
| `physics` | constantes do ISL: `mu` (atrito) e os limiares `isl_baixo`/`isl_alto` |
| `preprocessing` | limpeza dos dados brutos |
| `curve_detection` | B-spline, `limite_raio`, classes DNIT |
| `risk_measures` | limiares dos critérios de risco (Kamm, lateral, zigue-zague) |
| `montecarlo` | simulação de probabilidade de ISL |
| `ml` | parâmetros compartilhados dos modelos (split, CV, PCA) |
| `optuna` | nº de trials do tuning (usado pela análise em `notebooks/analise_risco.ipynb`) |
| `temporais` | modelos, alvo e hiperparâmetros do `--velocity` |

`features.yaml` tem `extracao:` (parâmetros da janela pré-curva: `janela_distancia`, `lead_gap`,
`vars_sensor`) e `flags:` com a lista de features por flag — `risk` (lista chapada),
`importance` (`herda: risk`) e `velocity` (`sensors` + `scalares_extras`). Whitelist explícita:
só entra no modelo o que estiver listado.

As constantes físicas (`g`, `μ`, limiares de ISL) ficam centralizadas em [curvant/constants.py](curvant/constants.py), que lê `μ` e os limiares da seção `physics` do config.

## Dados

Dataset público no HuggingFace: [`jwsouza13/routes_ML_inmetro`](https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro). Os dados foram coletados pela equipe Lainf do Inmetro.

| Conjunto | Veículo | Trecho | Local |
|---|---|---|---|
| ELETRONUCLEAR | Spin / Van | Rio de Janeiro | `eletronuclear` |
| RJ-DF | Nivus | Rio de Janeiro -> Brasília | `rjdf` |
| SERRA | Jetta | trecho serrano | `serra` |
| RJMGBA / JANEIRO | — | rotas adicionais | `rjmgba`, `janeiro` |