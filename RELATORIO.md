# CurvantML — Relatório de Progresso

**Objetivo:** Prever condução de risco em trechos curvos de rodovias utilizando dados de sensores OBD (GPS, acelerômetro, velocidade, RPM), classificando cada manobra como **Segura** ou **Perigosa** nos momentos anteriores à entrada na curva.

---

## Pipeline Implementado

```
Dados brutos (.parquet)
       ↓
[1] Pré-processamento          src/preprocessing.py
       ↓
[2] Detecção de curvas         src/curve_detection.py
       ↓
[3] Análise de condução        src/driving_analysis.py
       ↓
[4] Extração de features       src/features.py
       ↓
[5] Modelos de ML              src/models.py
```

### Pré-processamento (`src/preprocessing.py`)

- Remove acelações muito altas: |a| > 5 m/s²
- Remove velocidades impossíveis: > 150 km/h (bugs OBD)
- Se durante 30s o OBD não registrou nenhum ponto (logger pausou, perda de sinal, reconexão), a trajetória é cortada ali e se cria sub-trajetos com `time_sec` reiniciado

### Detecção de Curvas (`src/curve_detection.py`)

Curvatura de Frenet via B-spline cúbica com suavização gaussiana adaptativa. Classificação DNIT por raio de curvatura (D = 1145.92 / R):

| Classe | Raio | Grau |
|---|---|---|
| `muito_fechada` | R ≤ 50 m | D > 22.9° |
| `fechada` | R ≤ 100 m | D > 11.5° |
| `media` | R ≤ 200 m | D > 5.7° |
| `aberta` | R ≤ 500 m | D > 2.3° |
| `suave` | R > 500 m | — |

### 2.3 Análise de Condução Perigosa (`src/driving_analysis.py`)

Janelas deslizantes de 10 s classificadas como **Segura** ou **Perigosa** por dois critérios (Li et al., 2016):

| Critério | Definição |
|---|---|
| **Aceleração anormal** | Variação acumulada de velocidade Σ\|Δv\| > 20 km/h na janela |
| **Zigue-zague** | ≥ 3 mudanças de bearing > 15° (wrap-corrected) com aceleração centrípeta v²/R > 0.3 m/s² |

> **Nota:** O critério *direção perigosa* (baseado em bearing GPS) foi desabilitado (`angulo_max_direcao: 0.0`). Diagnóstico: cria dependência circular com a classificação DNIT (dispara apenas dentro de curvas detectadas), tornando ~98% das curvas Perigosas por construção. O critério foi projetado para giroscópio de alta frequência; com GPS a 1 Hz, o ruído de posição domina o sinal de heading.

### 2.4 Extração de Features (`src/features.py`)

Para cada curva detectada, extrai features da janela de **10 s anteriores à entrada na curva** — informações disponíveis em tempo real:

**Estatísticas clássicas** (5 × 4 variáveis = 20 features):

| Variável | Estatísticas |
|---|---|
| `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y` | mean, std, median, max, min |

**Features de tendência** (2 × 4 = 8 features):

| Feature | Descrição |
|---|---|
| `{var}_slope` | Coeficiente angular da regressão linear sobre o tempo — indica se o motorista está acelerando (+) ou freando (−) ao se aproximar da curva |
| `{var}_cv` | Coeficiente de variação (std/mean) — quão errático estava o comportamento |

**Contexto do trajeto** (3 features):

| Feature | Descrição |
|---|---|
| `n_curvas_antes` | Posição ordinal desta curva no trajeto |
| `n_perigosas_antes` | Quantidade de curvas perigosas já realizadas antes desta |
| `prop_perigosas_antes` | Proporção de curvas perigosas até o momento |

> Estas features são válidas em tempo real: ao chegar na N-ésima curva, o histórico das N-1 anteriores já é conhecido.

**Target (`manobra`):** proporção de pontos Perigosa dentro da curva ≥ 0.5 → evita sensibilidade à frequência de amostragem (o limiar `sum ≥ 2 pontos` era trivialmente atingido a 1 Hz).

**Total de amostras:** 1.091 manobras — **545 Perigosa (50%) / 546 Segura (50%)**

---

## 3. Modelos de Classificação (`src/models.py`)

### Pipeline de treinamento (sem data leakage)

```
Split estratificado (70/30)
       ↓
ImbPipeline: SMOTE → StandardScaler (→ PCA opcional) → Classificador
       ↓
StratifiedKFold(5) sobre X_train bruto
(SMOTE re-executado a cada fold — amostras sintéticas não cruzam para validação)
       ↓
Avaliação final no conjunto de teste
```

**Redução de dimensionalidade (PCA):** implementado no pipeline via `pca_n_components` em `config.yaml` — aceita número inteiro de componentes ou float 0–1 (variância explicada mínima). Fit exclusivamente sobre `X_train`; `X_test` apenas transformado. Atualmente desabilitado (`pca_n_components: null`) dado o volume moderado de features (34) em relação ao dataset (1.091 amostras).

### Resultados

| Classificador | CV Acc | CV F1 | Acc (teste) | F1 (teste) |
|---|---|---|---|---|
| Regressão Logística | 0.646 | 0.645 | 0.613 | 0.611 |
| SVM (linear) | 0.650 | 0.648 | 0.631 | 0.629 |
| Árvore de Decisão | 0.592 | 0.592 | 0.558 | 0.556 |
| **Floresta Aleatória** | **0.688** | **0.687** | **0.686** | **0.685** |
| Rede Neural (MLP) | 0.603 | 0.602 | 0.610 | 0.608 |

- Baseline aleatório: 50%
- Melhor modelo: **Floresta Aleatória — 68.6% acurácia** sem overfitting (CV ≈ teste)
- Também disponíveis: MLP Keras, GRU, LSTM

---

## 4. Visualização Interativa (`app/main.py`)

App Streamlit para inspeção qualitativa de cada manobra:

- **Mapa Folium:** trajeto completo + janela pré-curva (azul) + pontos da curva coloridos por classificação (🔴/🟢)
- **6 painéis Altair:** velocidade, Σ\|Δv\| com limiar, accel. lateral, accel. centrípeta, variação de bearing, RPM — todos com linha vertical em t = 0 (entrada na curva)
- **7 métricas** por curva: vel. média/máx, accel. lateral, variação acumulada, ctp máx, raio/DNIT, duração
- Filtro por rota, por classe (Segura/Perigosa) e por curva individual

```bash
streamlit run app/main.py
```

---

## 5. Estrutura do Projeto

```
curvant-ml/
├── src/               # domínio ML (pipeline, modelos, features)
├── graphics/          # visualização e plots exploratórios
├── utils/             # helpers genéricos (data, config)
├── scripts/           # entry points CLI (run.py, preprocess_data.py)
├── app/               # app Streamlit (main.py)
├── notebooks/         # Code_OBD.ipynb
├── data/
│   ├── raw/           # CSVs originais (INMETRO)
│   └── *.parquet      # dados processados
├── results/           # saídas: tabelas .tex, matrizes de confusão .pdf
└── config.yaml        # todos os hiperparâmetros e limiares
```

---

## 6. Limitações e Próximos Passos

| Limitação | Discussão |
|---|---|
| GPS a 1 Hz | Bearing calculado por diferença de posição tem ruído elevado — inviabiliza critério de direção (Li et al.). Solução: giroscópio dedicado ou fusão com IMU. |
| Teto de acurácia (~69%) | Features pré-curva têm correlação fraca com comportamento dentro da curva. Geometria da curva (raio, DNIT) seria o preditor mais forte, mas requer mapa de rodovias. |
| Tamanho do dataset | ~1.091 amostras balanceadas — pequeno para modelos deep learning. GRU/LSTM têm pouca vantagem sobre RF nesse volume. |

**Próximos passos sugeridos:**
- Avaliar target alternativo `manobra_accel_perigo` (critério mais limpo, sem GPS bearing)
- Aumentar janela pré-curva de 10 s para 20–30 s
- Incorporar dados de giroscópio quando disponíveis
