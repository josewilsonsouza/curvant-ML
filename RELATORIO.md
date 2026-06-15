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
[3] Caracterização de risco    src/characterization.py
       ↓
[4] Extração de features       src/features.py
       ↓
[5] Modelos de ML              src/models.py
```

### Pré-processamento (`src/preprocessing.py`)

- Remove acelerações muito altas: |a| > 5 m/s²
- Remove velocidades impossíveis: > 150 km/h (bugs OBD)
- Se durante 30 s o OBD não registrar nenhum ponto (logger pausou, perda de sinal, reconexão), a trajetória é cortada e surgem sub-trajetos com `time_sec` reiniciado

### Detecção de Curvas (`src/curve_detection.py`)

Curvatura de Frenet via B-spline cúbica com suavização gaussiana adaptativa. Classificação DNIT por raio de curvatura (D = 1145.92 / R):

| Classe | Raio | Grau |
|---|---|---|
| `muito_fechada` | R ≤ 50 m | D > 22.9° |
| `fechada` | R ≤ 100 m | D > 11.5° |
| `media` | R ≤ 200 m | D > 5.7° |
| `aberta` | R ≤ 500 m | D > 2.3° |
| `suave` | R > 500 m | — |

### 2.3 Caracterização de risco por curva (`src/characterization.py`)

O pipeline atual aplica `caracterizar_conducao()` a cada segmento contíguo de `curva=True`.
Para cada curva, a avaliação considera a janela de abordagem definida por `janela_aproximacao` segundos antes do início do segmento, mais todos os pontos do próprio segmento.

Critérios independentes:

| Critério | Definição |
|---|---|
| **Aceleração anormal** | `manobra_accel` = `max_t sqrt(accel_x² + accel_y²) > 0.7 · μ · g` |
| **Aceleração lateral** | `manobra_lateral` = `max_t |accel_y| > 2.0` e DNIT ≥ `aberta` (R ≤ 500 m) |
| **Zigue-zague** | `manobra_ziguezague` = ≥ 3 mudanças de bearing > 15° com alternância de sinal e `ctp_accel > 0.3` |

O rótulo combinado é `manobra_combinado = manobra_accel ∨ manobra_lateral ∨ manobra_ziguezague`, e o alias de texto é `'Perigosa'` / `'Segura'`.

> Observação: o módulo `src/driving_analysis.py` permanece no repositório como legado, mas o fluxo principal de `scripts/run.py` utiliza `src/characterization.py`.

Se a coluna `curva` estiver ausente, o código ainda pode cair em um fluxo legado de janelas fixas, mas o fluxo atual parte sempre de curvas detectadas.

### 2.4 Extração de Features (`src/features.py`)

Para cada curva detectada, o pipeline extrai features da janela pré-curva imediatamente anterior à entrada.
O tamanho da janela é configurável:
- se `features.janela_distancia` não for `null`, usa uma distância fixa antes da curva (atualmente 50 m no `config.yaml`)
- caso contrário, usa distância dinâmica baseada na desaceleração de conforto `v²/(2a)` com clip em `[janela_distancia_min, janela_distancia_max]`

Features extraídas por curva:

- F1: estatísticas de `vehicle_speed`, `engine_rpm`, `accel_x`, `accel_y` — mean, std, median, max, min, slope, cv, plus late-window mean/slope
- F2: `v_entry`, `v_speed_drop`, jerk máximo/σ, `v_entry_sq_over_raio_est`, contadores de eventos de risco na janela (`n_perigo_accel_janela`, `n_perigo_lateral_janela`)
- F3: geometria estimada da abordagem — `janela_raio_min`, `janela_raio_mean`, `janela_raio_last`
- F4: geometria real da curva seguinte — `f4_raio_min`, `f4_raio_mean`, `f4_dnit_num`, quando disponível para rotas conhecidas
- F5: contexto histórico — `n_curvas_antes`, `prop_perigosas_antes`, `prev_raio_min`, `prev_raio_mean`, `prev_dnit_num`

Targets de curvatura:

- `manobra_accel_curva`, `manobra_lateral_curva`, `manobra_ziguezague_curva`
- `manobra_combinado_curva` = 1 se qualquer critério disparar dentro da curva
- `manobra` é um alias de `manobra_combinado_curva`

> Nota: o target é binário por curva (`any`), não uma proporção de pontos dentro da curva.

---

## 3. Modelos de Classificação (`src/models.py`)

### Pipeline de treinamento (sem data leakage)

```
Split por rota base (treino/teste)
       ↓
ImbPipeline: SMOTE → StandardScaler (→ PCA opcional) → Classificador
       ↓
GroupKFold(5) sobre X_train com grupos por rota base
       ↓
Avaliação final no conjunto de teste reservado
```

O split inicial é feito por rota base (`_split_por_rota`), garantindo que subtrajectos derivados do mesmo arquivo não apareçam simultaneamente no treino e no teste.
No treinamento clássico, o `GroupKFold` é aplicado sobre `X_train` com os grupos correspondendo ao ID de rota original.
O `SMOTE` é executado apenas no fold de treino dentro de cada validação cruzada, evitando vazamento de amostras sintéticas.

**Redução de dimensionalidade (PCA):** configurável via `pca_n_components` em `config.yaml`. Está desabilitada por padrão (`null`) no repositório atual.

### Modelos disponíveis

- Regressão Logística
- SVM linear
- Árvore de Decisão
- Floresta Aleatória
- XGBoost
- MLP sklearn

O código também suporta tuning Optuna para XGBoost e RandomForest, além de `--mlp` e `--pytorch` para treinamentos específicos.

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
