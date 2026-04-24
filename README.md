# CurvantML

Framework de machine learning para **prever condução de risco em curvas**, momentos antes do motorista entrar no trecho curvo, utilizando dados de sensores veiculares OBD (GPS, acelerômetro, velocidade, RPM).

## Instalação

```bash
pip install -e ".[dev]"
```

## Como rodar

```bash
# 1. Limpeza dos dados brutos (executar uma vez, ou ao mudar parâmetros de pré-processamento)
python scripts/preprocess_data.py

# 2. Pipeline completo
python scripts/run.py                        # modelos clássicos de condução
python scripts/run.py --plot                 # + gráficos/matrizes de confusão em results/
python scripts/run.py --isl                  # + 4 blocos de análise (veja abaixo)
python scripts/run.py --mlp                  # + MLP sklearn com GridSearchCV
python scripts/run.py --keras                # + Keras MLP / GRU / LSTM
python scripts/run.py --isl --plot --mlp --keras
```

### Flag `--isl` — 3 blocos de análise

| Bloco | Target | Tipo | Descrição |
|---|---|---|---|
| ISL | `isl_class` | Classificação 3 classes | Prediz a classe de risco lateral da curva: **baixo** (ISL < 0,5) / **médio** (0,5–0,8) / **alto** (≥ 0,8) |
| P1 — Aceleração | `curve_accel_y_max`, `curve_abs_accel_max` | Regressão | Pico de aceleração lateral e total medido pelo sensor (sem assumir μ) |
| P2 — Velocidade | `manobra_velocidade` | Classificação binária | Detecta excesso de velocidade na entrada: $v_{\text{entry}} > v_{\text{safe}}(R)$ |

**v_safe**: velocidade máxima segura para o raio detectado na entrada da curva, calculada como

$$v_{\text{safe}}(R) = \sqrt{R \cdot g \cdot \mu} \times 3{,}6 \quad [\text{km/h}]$$

onde $R$ é o raio de curvatura (m) no primeiro ponto da curva, $g = 9{,}81\ \text{m/s}^2$ e $\mu = 0{,}6$ (asfalto seco). Se $v_{\text{entry}} > v_{\text{safe}}$, `manobra_velocidade = 1`.

### Configuração (`config.yaml`)

| Parâmetro | Seção | Padrão | Descrição |
|---|---|---|---|
| `janela_tempo` | `features` | `10` | Segundos da janela pré-curva |
| `janela_distancia` | `features` | `null` | Metros antes da curva (P3); tem precedência sobre `janela_tempo` |
| `isl_max_cap_percentil` | `ml` | `99` | Remove outliers extremos de ISL antes de treinar |
| `pca_n_components` | `ml` | `null` | PCA após SMOTE+Scaler (`null` = desativado) |

## Pipeline

```
Dados brutos (OBD)
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
Análise de condução      janelas de 10 s → Segura / Perigosa:
                           • aceleração/frenagem anormal
                           • direção perigosa (gated por DNIT ≥ média)
                           • zigue-zague
    │
    ▼
Extração de features     janela pré-curva (tempo ou distância):
                           mean / std / median / max / min / slope / cv de
                           vehicle_speed, engine_rpm, accel_x, accel_y
                           + v_entry (velocidade de entrada na curva)
                           + n_curvas_antes, prop_perigosas_antes
                           + prev_raio_min/mean, prev_dnit_num (curva anterior)
    │
    ▼
Modelos de ML
    ├── Condução          target: manobra (maioria Perigosa nos pontos da curva)
    │   (padrão)          Clássicos: Reg. Logística, SVM, Árvore, Floresta,
    │                               XGBoost
    │                     Deep learning (--keras): MLP Keras, GRU, LSTM
    │
    └── --isl
        ├── ISL classif.  target: isl_alto (ISL ≥ 0,8)
        ├── ISL regressão target: isl_max
        ├── P1 regressão  target: curve_accel_y_max / curve_abs_accel_max
        └── P2 classif.   target: manobra_velocidade (v_entry > v_safe)
```

## ISL — Índice de Segurança Lateral

Mede o quão próximo o veículo está do limite de aderência lateral:

$$\text{ISL} = \frac{v^2}{R \cdot g \cdot \mu} = \frac{\text{ctp\_accel}}{g \cdot \mu}$$

onde $v$ é a velocidade (m/s), $R$ o raio de curvatura (m), $g = 9{,}81\ \text{m/s}^2$ e $\mu = 0{,}6$ (asfalto seco).

| Classe | ISL | Interpretação |
|---|---|---|
| `baixo` | $< 0{,}5$ | Ampla margem de segurança |
| `medio` | $0{,}5 \leq \text{ISL} < 0{,}8$ | Atenção recomendada |
| `alto` | $\geq 0{,}8$ | Próximo ao limite de aderência |

## Dados

Dataset público no HuggingFace: [`jwsouza13/routes_ML_inmetro`](https://huggingface.co/datasets/jwsouza13/routes_ML_inmetro)

| Conjunto | Veículo | Trecho |
|---|---|---|
| ELETRONUCLEAR | Spin / Van | Rio de Janeiro |
| RJ-DF | Nivus | Rio de Janeiro → Brasília |
| SERRA | Jetta | Trecho serrano |

## Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `data/eletro_rjdf_serra_clean.parquet` | Dataset limpo |
| `results/tab_result.tex` | Distribuição dos critérios de condução por classe |
| `results/ml_resultados.tex` | Métricas dos modelos de condução (manobra) |
| `results/isl_resultados.tex` | Métricas dos modelos ISL (classificação) |
| `results/regressao_curve_accel_y_max.tex` | Métricas P1 — pico aceleração lateral (`--isl`) |
| `results/regressao_curve_abs_accel_max.tex` | Métricas P1 — pico aceleração total (`--isl`) |
| `results/matriz_confusao_<modelo>.pdf` | Matrizes de confusão (--plot) |
| `results/isl_cm_<modelo>.pdf` | Matrizes de confusão ISL (`--isl --plot`) |
| `results/scatter_<target>_<modelo>.pdf` | Scatter real vs. predito para regressão (`--isl --plot`) |
| `results/curva_treinamento_keras.pdf` | Curva de treinamento Keras (`--keras --plot`) |
