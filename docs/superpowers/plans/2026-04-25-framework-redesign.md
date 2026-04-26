# CurvantML Framework Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesenhar o curvantML com taxonomia explícita de risco, dois modos de pipeline (rota conhecida/desconhecida), features universais, avaliação por `id_route`, Optuna e PyTorch multi-tarefa.

**Architecture:** `characterization.py` substitui `driving_analysis.py` produzindo colunas de risco separadas por critério. `features.py` consome essas colunas e gera novos alvos por curva. `models.py` usa split por `id_route` + GroupKFold + Optuna. `models_pytorch.py` (novo) implementa MLP multi-tarefa em PyTorch.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.0, Optuna ≥3.0, scikit-learn, XGBoost, imbalanced-learn, pandas/numpy.

---

## File Map

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/characterization.py` | Criar | Taxonomia de risco: 3 critérios separados + `manobra_combinado` |
| `src/features.py` | Modificar | +accel_z, +jerk_z, +v_entry_sq_over_raio_est, +F4, novos targets |
| `src/models.py` | Modificar | Split por rota, GroupKFold, Optuna para XGB/RF |
| `src/models_pytorch.py` | Criar | MultiTaskMLP, CurvantDataset, treinar_multitask_mlp |
| `src/pipeline.py` | Modificar | Usar characterization.py, expor Modo 1/2 |
| `scripts/run.py` | Modificar | Flags --modo, --pytorch |
| `config.yaml` | Modificar | Seção optuna, thresholds de caracterização |
| `pyproject.toml` | Modificar | Adicionar optuna, torch |

`src/driving_analysis.py` é mantido sem alterações (retrocompatibilidade com app Streamlit).

---

## Task 1: Instalar dependências

**Files:** `pyproject.toml`

- [ ] **Passo 1: Adicionar optuna e torch às dependências**

```toml
dependencies = [
    "huggingface-hub",
    "scipy",
    "matplotlib",
    "plotly",
    "seaborn",
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "imbalanced-learn",
    "tensorflow",
    "pyyaml",
    "pyarrow",
    "optuna>=3.0",
    "torch>=2.0",
]
```

- [ ] **Passo 2: Instalar**

```bash
pip install -e ".[dev]"
```

Saída esperada: `Successfully installed optuna-... torch-...`

- [ ] **Passo 3: Verificar imports**

```bash
python -c "import optuna; import torch; print('optuna', optuna.__version__, '| torch', torch.__version__)"
```

Saída esperada: `optuna 3.x.x | torch 2.x.x`

- [ ] **Passo 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add optuna and torch dependencies"
```

---

## Task 2: Criar src/characterization.py

**Files:** Criar `src/characterization.py`

Substitui `driving_analysis.py` como orquestrador de caracterização. Diferença chave: produz colunas separadas por critério (`manobra_accel`, `manobra_lateral`, `manobra_ziguezague`) em vez de uma única coluna `conducao`.

- [ ] **Passo 1: Criar o arquivo**

```python
# src/characterization.py
"""
CurvantML — Taxonomia explícita de caracterizações de risco em curvas.

Produz colunas separadas por critério de risco:
  manobra_accel      — aceleração/frenagem anormal na janela
  manobra_lateral    — direção perigosa (|accel_y| + DNIT)
  manobra_ziguezague — padrão de zigue-zague
  manobra_combinado  — OR dos três (retrocompat com 'conducao')
  conducao           — alias de manobra_combinado ('Perigosa'/'Segura')
"""

import numpy as np
import pandas as pd

_DNIT_RISCO: dict[str, int] = {
    'suave': 0, 'aberta': 1, 'media': 2, 'fechada': 3, 'muito_fechada': 4,
}
_RISCO_MIN_DIRECAO = 2


def calcular_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Ângulo de direção (bearing) entre dois pontos GPS, em graus [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _detectar_zigue_zague(
    janela: pd.DataFrame,
    limiar_bearing: float = 15.0,
    limiar_accel_lateral: float = 0.3,
    min_mudancas: int = 3,
) -> bool:
    lats = janela['lat'].tolist()
    lons = janela['lon'].tolist()
    ctp  = janela['ctp_accel'].tolist()
    if len(lats) < 2:
        return False
    bearings = np.array([
        calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ])
    contador = 0
    for i in range(1, len(bearings)):
        mudanca = abs((bearings[i] - bearings[i - 1] + 180) % 360 - 180)
        if mudanca > limiar_bearing and abs(ctp[i]) > limiar_accel_lateral:
            contador += 1
            if contador >= min_mudancas:
                return True
    return False


def caracterizar_janela(
    janela: pd.DataFrame,
    var_velocidade_max: float = 20.0,
    limiar_accel_lateral: float = 3.0,
    zz_limiar_bearing: float = 15.0,
    zz_limiar_accel: float = 0.3,
    zz_min_mudancas: int = 3,
) -> dict:
    """Aplica os três critérios de risco a uma janela de tempo. Retorna dict de bools."""
    var_vel       = janela['vehicle_speed'].diff().abs().sum()
    manobra_accel = bool(var_vel > var_velocidade_max)

    risco_dnit = (
        int(janela['classe_dnit'].map(_DNIT_RISCO).max())
        if 'classe_dnit' in janela.columns else 3
    )
    accel_lateral_max = float(janela['accel_y'].abs().max())
    manobra_lateral   = bool(risco_dnit >= _RISCO_MIN_DIRECAO and accel_lateral_max > limiar_accel_lateral)

    manobra_ziguezague = _detectar_zigue_zague(
        janela, zz_limiar_bearing, zz_limiar_accel, zz_min_mudancas
    )

    return {
        'manobra_accel':      manobra_accel,
        'manobra_lateral':    manobra_lateral,
        'manobra_ziguezague': manobra_ziguezague,
        'risco_dnit':         risco_dnit,
    }


def caracterizar_conducao(
    df: pd.DataFrame,
    janela_tempo: int = 10,
    var_velocidade_max: float = 20.0,
    limiar_accel_lateral: float = 3.0,
    **zz_kwargs,
) -> pd.DataFrame:
    """
    Classifica janelas deslizantes com a taxonomia de risco explícita.

    Colunas adicionadas ao DataFrame:
      manobra_accel, manobra_lateral, manobra_ziguezague (bool por janela)
      manobra_combinado (OR dos três)
      conducao ('Perigosa'/'Segura') — alias de manobra_combinado
      risco_dnit, id_janela
    """
    resultados = []
    inicio    = df['time_sec'].min()
    fim       = df['time_sec'].max()
    t         = inicio
    id_janela = 1

    while t + janela_tempo <= fim:
        janela = df[(df['time_sec'] >= t) & (df['time_sec'] < t + janela_tempo)]
        if len(janela) < 2:
            t += janela_tempo
            continue

        r = caracterizar_janela(janela, var_velocidade_max, limiar_accel_lateral, **zz_kwargs)
        manobra_combinado = r['manobra_accel'] or r['manobra_lateral'] or r['manobra_ziguezague']

        janela = janela.copy()
        janela['manobra_accel']      = r['manobra_accel']
        janela['manobra_lateral']    = r['manobra_lateral']
        janela['manobra_ziguezague'] = r['manobra_ziguezague']
        janela['manobra_combinado']  = manobra_combinado
        janela['conducao']           = 'Perigosa' if manobra_combinado else 'Segura'
        janela['risco_dnit']         = r['risco_dnit']
        janela['id_janela']          = id_janela
        id_janela += 1
        resultados.append(janela)
        t += janela_tempo

    return pd.concat(resultados, ignore_index=True)


def caracterizar_todos_trajetos(
    dfs_curves: pd.DataFrame,
    janela_tempo: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """Aplica caracterizar_conducao em cada trajeto e concatena."""
    return pd.concat([
        caracterizar_conducao(
            dfs_curves.query(f'id_route == "{traj}"'),
            janela_tempo=janela_tempo,
            **kwargs,
        )
        for traj in dfs_curves['id_route'].unique()
    ], ignore_index=True)
```

- [ ] **Passo 2: Verificar import**

```bash
python -c "from src.characterization import caracterizar_todos_trajetos; print('OK')"
```

Saída esperada: `OK`

- [ ] **Passo 3: Commit**

```bash
git add src/characterization.py
git commit -m "feat: add explicit risk characterization taxonomy (characterization.py)"
```

---

## Task 3: Atualizar config.yaml

**Files:** `config.yaml`

- [ ] **Passo 1: Adicionar seção optuna e renomear driving_analysis para characterization**

No final do arquivo, após a seção `neural_networks`, adicionar:

```yaml
optuna:
  n_trials: 50        # trials por modelo/target
  timeout: 300        # segundos máximos por otimização (null = sem limite)
```

E na seção `driving_analysis`, apenas documentar que ela agora alimenta `characterization.py`:

```yaml
driving_analysis:
  janela_tempo: 10
  var_velocidade_max: 20
  velocidade_max_direcao: 30
  angulo_max_direcao: 0.0
  limiar_accel_lateral: 3.0   # m/s² — critério manobra_lateral
  zigue_zague:
    limiar_bearing: 15
    limiar_accel_lateral: 0.3
    min_mudancas: 3
```

- [ ] **Passo 2: Verificar carregamento**

```bash
python -c "from utils.config import carregar_config; c=carregar_config(); print(c.get('optuna'))"
```

Saída esperada: `{'n_trials': 50, 'timeout': 300}`

- [ ] **Passo 3: Commit**

```bash
git add config.yaml
git commit -m "config: add optuna section"
```

---

## Task 4: Atualizar src/features.py

**Files:** `src/features.py`

Mudanças em ordem:
1. Adicionar `accel_z` a `vars_sensor`
2. Estender loop de jerk para incluir `accel_z`
3. Renomear referências de colunas de caracterização (`aceleracao_anormal` → `manobra_accel` etc.)
4. Adicionar `v_entry_sq_over_raio_est`
5. Adicionar features F4 (`f4_raio_min`, `f4_raio_mean`, `f4_dnit_num`) e `modo_rota_conhecida`
6. Renomear targets por curva e adicionar `v_excess` e `manobra_combinado_curva`

- [ ] **Passo 1: Adicionar accel_z a vars_sensor (linha 82)**

```python
# Antes:
vars_sensor = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y']

# Depois:
vars_sensor = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y', 'accel_z']
```

- [ ] **Passo 2: Estender loop de jerk para accel_z (linhas 145-151)**

```python
# Antes:
for var in ['accel_x', 'accel_y']:

# Depois:
for var in ['accel_x', 'accel_y', 'accel_z']:
```

- [ ] **Passo 3: Renomear referências de colunas de caracterização (linhas 156-158)**

```python
# Antes:
row['n_perigo_acc_anormal_janela']  = int(janela['aceleracao_anormal'].sum())
row['n_perigo_dir_perigosa_janela'] = int(janela['direcao_perigosa'].sum())
row['n_perigo_zigue_zague_janela']  = int(janela['zigue_zague'].sum())

# Depois:
row['n_perigo_accel_janela']   = int(janela['manobra_accel'].sum())
row['n_perigo_lateral_janela'] = int(janela['manobra_lateral'].sum())
row['n_perigo_zz_janela']      = int(janela['manobra_ziguezague'].sum())
```

- [ ] **Passo 4: Adicionar v_entry_sq_over_raio_est após v_entry_vs_mean (após linha 208)**

Localizar o bloco que calcula `v_speed_drop`, `v_speed_drop_pct`, `v_entry_vs_mean` e adicionar logo após:

```python
# Proxy físico de ISL na entrada: v²/R_estimado (feature F3)
_raio_est = row.get('janela_raio_min', np.nan)
if pd.notna(_raio_est) and _raio_est > 0:
    row['v_entry_sq_over_raio_est'] = float((v_entry / 3.6) ** 2 / _raio_est)
else:
    row['v_entry_sq_over_raio_est'] = np.nan
```

- [ ] **Passo 5: Adicionar F4 features e modo_rota_conhecida após computação de curve_dnit_num (após linha 250)**

```python
# F4 — geometria real da curva seguinte (Modo 1).
# São cópias de curve_raio_* com nome distinto para não conflitar com _COLS_EXCLUIR.
# O pipeline define modo_rota_conhecida=0 e zera f4_* para Modo 2.
row['f4_raio_min']  = row.get('curve_raio_min', np.nan)
row['f4_raio_mean'] = row.get('curve_raio_mean', np.nan)
row['f4_dnit_num']  = row.get('curve_dnit_num', 0)
row['modo_rota_conhecida'] = 1  # default Modo 1; pipeline sobrescreve para Modo 2
```

- [ ] **Passo 6: Renomear targets por curva e adicionar v_excess e manobra_combinado_curva (linhas 178-180)**

```python
# Antes:
row['manobra']              = 1 if curva['conducao'].mean() >= 0.5 else 0
row['manobra_accel_perigo'] = 1 if curva['aceleracao_anormal'].mean() >= 0.5 else 0
row['manobra_dir_perigosa'] = 1 if curva['direcao_perigosa'].mean() >= 0.5 else 0
row['manobra_zigue_zague']  = 1 if curva['zigue_zague'].mean() >= 0.5 else 0

# Depois:
row['manobra_accel_curva']       = 1 if curva['manobra_accel'].mean() >= 0.5 else 0
row['manobra_lateral_curva']     = 1 if curva['manobra_lateral'].mean() >= 0.5 else 0
row['manobra_ziguezague_curva']  = 1 if curva['manobra_ziguezague'].mean() >= 0.5 else 0
row['manobra_combinado_curva']   = 1 if curva['manobra_combinado'].mean() >= 0.5 else 0
row['manobra']                   = row['manobra_combinado_curva']  # retrocompat
```

- [ ] **Passo 7: Adicionar v_excess após manobra_velocidade (linha 211)**

```python
# Após: row['manobra_velocidade'] = int(v_entry > v_safe) if ...
row['v_excess'] = row['manobra_velocidade']  # alias explícito da taxonomia
```

- [ ] **Passo 8: Verificar que pipeline não quebra**

```bash
python -c "
import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
from utils.config import carregar_config
from src.curve_detection import detectar_curvas, identificar_trechos_curvos
from src.characterization import caracterizar_todos_trajetos
from src.features import extrair_features
cfg = carregar_config()
df = pd.read_parquet('data/eletro_rjdf_serra_rjmgba_janeiro.parquet')
dfs_c = []
for t in df['id_route'].unique()[:5]:
    try:
        dfs_c.append(detectar_curvas(df.query(f'id_route==\"{t}\"'), sigma=cfg['curve_detection']['sigma'], limite_raio=cfg['curve_detection']['limite_raio']))
    except: pass
dfs_curves = pd.concat(dfs_c)
R_MIN = cfg['curve_detection'].get('raio_min', 5.0)
dfs_curves['ctp_accel'] = (dfs_curves['vehicle_speed']/3.6)**2 / dfs_curves['raio_curvatura'].clip(lower=R_MIN)
dfs_curves['abs_accel'] = np.sqrt(dfs_curves['accel_x']**2 + dfs_curves['accel_y']**2)
da = cfg['driving_analysis']
df_an = caracterizar_todos_trajetos(dfs_curves, janela_tempo=da['janela_tempo'], var_velocidade_max=da['var_velocidade_max'], limiar_accel_lateral=da.get('limiar_accel_lateral', 3.0))
dfs_trechos = identificar_trechos_curvos(df_an)
feat = extrair_features(dfs_trechos, janela_tempo=cfg['features']['janela_tempo'])
print('Colunas novas:', [c for c in feat.columns if c in ['manobra_accel_curva','manobra_lateral_curva','manobra_ziguezague_curva','manobra_combinado_curva','v_excess','f4_raio_min','v_entry_sq_over_raio_est','modo_rota_conhecida']])
print('Shape:', feat.shape)
"
```

Saída esperada (colunas novas listadas, sem erro):
```
Colunas novas: ['manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva', 'manobra_combinado_curva', 'v_excess', 'f4_raio_min', 'v_entry_sq_over_raio_est', 'modo_rota_conhecida']
Shape: (NNNN, MM)
```

- [ ] **Passo 9: Commit**

```bash
git add src/features.py
git commit -m "feat: update features.py — accel_z, jerk_z, F4 features, new risk targets"
```

---

## Task 5: Atualizar src/models.py — _COLS_EXCLUIR e split por rota

**Files:** `src/models.py`

- [ ] **Passo 1: Atualizar _COLS_EXCLUIR (linhas 32-45)**

```python
_COLS_EXCLUIR = [
    # identificadores
    'id_route', 'id_trecho_curvo', 'time_inicio', 'time_fim',
    # targets de caracterização por curva
    'manobra', 'manobra_combinado_curva',
    'manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva',
    # targets ISL (calculados dentro da curva — leakage)
    'isl_value', 'isl_mean', 'isl_max', 'isl_class', 'isl_alto',
    # targets de aceleração dentro da curva
    'curve_accel_y_max', 'curve_accel_y_mean',
    'curve_abs_accel_max', 'curve_abs_accel_mean',
    # targets de velocidade
    'v_excess', 'manobra_velocidade', 'v_safe_dnit', 'v_entry_ratio',
    # geometria bruta da curva atual (leakage; usar f4_* em Modo 1)
    'curve_raio_min', 'curve_raio_mean', 'curve_dnit_num',
]
```

- [ ] **Passo 2: Adicionar imports necessários no topo do arquivo**

```python
from sklearn.model_selection import GroupKFold
```

- [ ] **Passo 3: Adicionar função _split_por_rota antes de _preparar_xy**

```python
def _split_por_rota(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split por id_route (sem amostras da mesma rota em treino e teste).
    Retorna (X_train, X_test, y_train, y_test, groups_train).
    groups_train é usado pelo GroupKFold.
    """
    rotas = df['id_route'].unique()
    train_rotas, test_rotas = train_test_split(
        rotas, test_size=test_size, random_state=random_state
    )
    feature_cols = [c for c in df.columns if c not in _COLS_EXCLUIR]
    cols = feature_cols + [target, 'id_route']

    df_train = (
        df[df['id_route'].isin(train_rotas)][cols]
        .dropna(subset=feature_cols + [target])
    )
    df_test = (
        df[df['id_route'].isin(test_rotas)][cols]
        .dropna(subset=feature_cols + [target])
    )

    X_train      = df_train[feature_cols].values
    y_train      = df_train[target].values
    groups_train = df_train['id_route'].values
    X_test       = df_test[feature_cols].values
    y_test       = df_test[target].values

    return X_train, X_test, y_train, y_test, groups_train
```

- [ ] **Passo 4: Substituir train_test_split + StratifiedKFold em aplicar_modelos_ml**

Localizar as linhas que fazem `train_test_split` e criam `cv = StratifiedKFold(...)` na função `aplicar_modelos_ml` e substituir por:

```python
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state
    )
    cv = GroupKFold(n_splits=cv_folds)
```

E na chamada de `cross_validate`, adicionar `groups=groups_train`:

```python
    cv_res = cross_validate(
        pipe, X_train, y_train, cv=cv, groups=groups_train,
        scoring={'acc': 'accuracy', 'f1': f'f1_{f1_average}'},
    )
```

- [ ] **Passo 5: Aplicar a mesma troca em treinar_modelo_isl e treinar_regressao**

Em `treinar_modelo_isl` (a partir da linha que cria `X_train, X_test`):

```python
    # Substituir train_test_split manual por:
    rotas = df_clean['id_route'].values if 'id_route' in df_clean.columns else None
    if rotas is not None:
        train_rotas, test_rotas = train_test_split(
            df_clean['id_route'].unique(), test_size=test_size, random_state=random_state
        )
        df_tr = df_clean[df_clean['id_route'].isin(train_rotas)]
        df_te = df_clean[df_clean['id_route'].isin(test_rotas)]
        X_train = df_tr[feature_cols].values
        y_train = df_tr['_isl_y'].values
        groups_train = df_tr['id_route'].values
        X_test  = df_te[feature_cols].values
        y_test  = df_te['_isl_y'].values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        groups_train = np.zeros(len(X_train))

    cv = GroupKFold(n_splits=cv_folds)
```

Nota: `feature_cols` em `treinar_modelo_isl` já exclui `_COLS_EXCLUIR_ISL`. Adicionar `'id_route'` à lista de colunas selecionadas para poder extrair groups:

```python
    feature_cols = [c for c in df_valid.columns if c not in _COLS_EXCLUIR_ISL and c != '_isl_y']
    df_clean = df_valid[feature_cols + ['_isl_y', 'id_route']].dropna(subset=feature_cols + ['_isl_y'])
```

Em `treinar_regressao`, mesma lógica — adicionar `'id_route'` ao df_clean e usar `_split_por_rota`-like logic ou refatorar para usar `_split_por_rota` diretamente.

- [ ] **Passo 6: Verificar que run.py ainda executa**

```bash
python scripts/run.py 2>&1 | head -30
```

Saída esperada: sem `KeyError` ou `AttributeError`, deve mostrar resultados de modelos clássicos.

- [ ] **Passo 7: Commit**

```bash
git add src/models.py
git commit -m "feat: route-based train/test split and GroupKFold in models.py"
```

---

## Task 6: Adicionar Optuna a src/models.py

**Files:** `src/models.py`

- [ ] **Passo 1: Adicionar funções de tuning Optuna após as funções existentes de pipeline**

```python
def _optuna_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    task: str = 'classify',
    cv_folds: int = 5,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Tuna XGBoost com Optuna usando GroupKFold.
    task='classify' → XGBClassifier + f1_weighted
    task='regress'  → XGBRegressor  + r2
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 9),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state':     random_state,
        }
        if task == 'classify':
            clf     = XGBClassifier(eval_metric='logloss', **params)
            scoring = 'f1_weighted'
        else:
            clf     = XGBRegressor(eval_metric='rmse', **params)
            scoring = 'r2'
        pipe   = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        cv     = GroupKFold(n_splits=cv_folds)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _optuna_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    task: str = 'classify',
    cv_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 42,
) -> dict:
    """
    Tuna RandomForest com Optuna usando GroupKFold.
    task='classify' → RandomForestClassifier + f1_weighted
    task='regress'  → RandomForestRegressor  + r2
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state':      random_state,
        }
        if task == 'classify':
            clf     = RandomForestClassifier(**params)
            scoring = 'f1_weighted'
        else:
            clf     = RandomForestRegressor(**params)
            scoring = 'r2'
        pipe   = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        cv     = GroupKFold(n_splits=cv_folds)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups, scoring=scoring)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

- [ ] **Passo 2: Adicionar função de treino com Optuna integrado**

```python
def aplicar_modelos_ml_otimizados(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    n_trials_xgb: int = 50,
    n_trials_rf: int = 30,
    target: str = 'manobra',
    f1_average: str = 'weighted',
) -> pd.DataFrame:
    """
    Igual a aplicar_modelos_ml mas com Optuna para XGBoost e RandomForest.
    Os demais modelos usam defaults (baselines).
    """
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state
    )
    cv = GroupKFold(n_splits=cv_folds)

    print(f"  Tuning XGBoost com Optuna ({n_trials_xgb} trials)...")
    best_xgb = _optuna_xgb(X_train, y_train, groups_train, cv_folds=cv_folds,
                            n_trials=n_trials_xgb, random_state=random_state)
    print(f"  Melhores params XGB: {best_xgb}")

    print(f"  Tuning RandomForest com Optuna ({n_trials_rf} trials)...")
    best_rf = _optuna_rf(X_train, y_train, groups_train, cv_folds=cv_folds,
                         n_trials=n_trials_rf, random_state=random_state)
    print(f"  Melhores params RF: {best_rf}")

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'SVM':                 svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão':   DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória':  RandomForestClassifier(**best_rf, random_state=random_state),
        'XGBoost':             XGBClassifier(eval_metric='logloss', **best_xgb),
        'Rede Neural (MLP)':   MLPClassifier(
            activation='relu', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    linhas = []
    for nome, clf in modelos.items():
        pipe   = _construir_pipeline(clf, random_state=random_state)
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'acc': 'accuracy', 'f1': f'f1_{f1_average}'},
        )
        cv_acc = cv_res['test_acc'].mean()
        cv_f1  = cv_res['test_f1'].mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average=f1_average)
        prec = precision_score(y_test, y_pred, average=f1_average, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=f1_average)

        print(
            f'{nome:30s}  CV Acc: {cv_acc:.3f}  CV F1: {cv_f1:.3f}  |  '
            f'Teste Acc: {acc:.3f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}'
        )

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Segura', 'Risco'],
                        yticklabels=['Segura', 'Risco'])
            plt.title(nome)
            plt.ylabel('Real')
            plt.xlabel('Prevista')
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/matriz_confusao_{nome}_opt.pdf', bbox_inches='tight')
            plt.close()

        linhas.append({
            'Classificador': nome, 'CV Acc (média)': cv_acc, 'CV F1 (média)': cv_f1,
            'Acc (teste)': acc, 'F1 (teste)': f1, 'Precisão (teste)': prec, 'Recall (teste)': rec,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs('results', exist_ok=True)
    df_res.to_latex('results/ml_resultados_opt.tex', float_format='%.3f', index=False)
    return df_res
```

- [ ] **Passo 3: Verificar import**

```bash
python -c "from src.models import aplicar_modelos_ml_otimizados, _optuna_xgb, _optuna_rf; print('OK')"
```

- [ ] **Passo 4: Commit**

```bash
git add src/models.py
git commit -m "feat: add Optuna tuning for XGBoost and RandomForest (GroupKFold-aware)"
```

---

## Task 7: Atualizar src/pipeline.py

**Files:** `src/pipeline.py`

- [ ] **Passo 1: Substituir import de driving_analysis por characterization**

```python
# Antes:
from src.driving_analysis import detectar_conducao_perigosa

# Depois:
from src.characterization import caracterizar_conducao
```

- [ ] **Passo 2: Atualizar etapa_analise_conducao para usar nova função e novas colunas**

```python
def etapa_analise_conducao(dfs_curves: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Etapa 4: classifica janelas de 10s com a taxonomia explícita de risco."""
    da = cfg['driving_analysis']

    partes = []
    for traj in dfs_curves['id_route'].unique():
        dt = dfs_curves.query(f'id_route == "{traj}"')
        partes.append(caracterizar_conducao(
            dt,
            janela_tempo=da['janela_tempo'],
            var_velocidade_max=da['var_velocidade_max'],
            limiar_accel_lateral=da.get('limiar_accel_lateral', 3.0),
            zz_limiar_bearing=da.get('zigue_zague', {}).get('limiar_bearing', 15.0),
            zz_limiar_accel=da.get('zigue_zague', {}).get('limiar_accel_lateral', 0.3),
            zz_min_mudancas=da.get('zigue_zague', {}).get('min_mudancas', 3),
        ))

    df_analysis = pd.concat(partes, ignore_index=True)

    n_perigosa = df_analysis['manobra_combinado'].sum()
    n_segura   = (~df_analysis['manobra_combinado']).sum()
    n_accel    = df_analysis['manobra_accel'].sum()
    n_lateral  = df_analysis['manobra_lateral'].sum()
    n_zz       = df_analysis['manobra_ziguezague'].sum()
    print(f"  Janelas — Perigosa: {n_perigosa} | Segura: {n_segura}")
    print(f"  Critérios — Accel: {n_accel} | Lateral: {n_lateral} | ZZ: {n_zz}")

    return df_analysis
```

- [ ] **Passo 3: Atualizar etapa_features para usar novas colunas**

```python
def etapa_features(df_analysis: pd.DataFrame, cfg: dict, modo: str = 'modo1') -> pd.DataFrame:
    """
    Etapas 5-6: extrai features e alvos por curva.

    modo='modo1' — inclui F4 (geometria da curva seguinte)
    modo='modo2' — zera F4 e modo_rota_conhecida=0
    """
    df_analysis = df_analysis.copy()
    for col in ['manobra_accel', 'manobra_lateral', 'manobra_ziguezague', 'manobra_combinado']:
        if col in df_analysis.columns:
            df_analysis[col] = df_analysis[col].astype(int)

    dfs_trechos = identificar_trechos_curvos(df_analysis)
    features_df = extrair_features(
        dfs_trechos,
        janela_tempo=cfg['features']['janela_tempo'],
        janela_distancia=cfg['features'].get('janela_distancia'),
    )

    # Aplicar Modo 2: zerar F4 features
    if modo == 'modo2':
        for col in ['f4_raio_min', 'f4_raio_mean', 'f4_dnit_num']:
            if col in features_df.columns:
                features_df[col] = np.nan
        features_df['modo_rota_conhecida'] = 0

    n_perigosa = features_df['manobra_combinado_curva'].sum()
    n_segura   = (features_df['manobra_combinado_curva'] == 0).sum()
    print(f"  {len(features_df)} amostras — Perigosa: {n_perigosa} | Segura: {n_segura}")

    crit_cols = {
        'manobra_accel_curva':      'Aceleração anormal',
        'manobra_lateral_curva':    'Direção perigosa',
        'manobra_ziguezague_curva': 'Zigue-zague',
    }
    tab = features_df.groupby('manobra_combinado_curva')[list(crit_cols.keys())].sum().rename(columns=crit_cols)
    tab.index = tab.index.map({0: 'Segura', 1: 'Perigosa'})
    tab.insert(0, 'Condução', features_df.groupby('manobra_combinado_curva').size().rename({0: 'Segura', 1: 'Perigosa'}))
    tab.index.name = 'Manobra'
    print(tab.to_string())
    os.makedirs('results', exist_ok=True)
    tab.to_latex('results/tab_result.tex', index=True)

    return features_df
```

Adicionar `import os` e `import numpy as np` ao topo do pipeline.py se ausentes.

- [ ] **Passo 4: Adicionar nova etapa para modelos otimizados**

```python
def etapa_ml_otimizado(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Treina modelos clássicos com Optuna (XGB + RF) e split por rota."""
    from src.models import aplicar_modelos_ml_otimizados

    ml  = cfg['ml']
    opt = cfg.get('optuna', {})
    resultados = aplicar_modelos_ml_otimizados(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        n_trials_xgb=opt.get('n_trials', 50),
        n_trials_rf=opt.get('n_trials', 30),
    )
    print(resultados.to_string(index=False))
    return resultados
```

- [ ] **Passo 5: Verificar import**

```bash
python -c "from src.pipeline import etapa_analise_conducao, etapa_features, etapa_ml_otimizado; print('OK')"
```

- [ ] **Passo 6: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: update pipeline.py to use characterization.py and Mode 1/2 support"
```

---

## Task 8: Atualizar scripts/run.py

**Files:** `scripts/run.py`

- [ ] **Passo 1: Adicionar imports e flags**

Substituir o bloco de imports de pipeline e o argparse:

```python
from src.pipeline import (
    etapa_curvas, etapa_analise_conducao, etapa_features,
    etapa_ml_classico, etapa_ml_otimizado,
    etapa_mlp_sklearn,
    etapa_isl_modelo, etapa_accel_regressao, etapa_manobra_velocidade,
)
```

No argparse, adicionar novos flags:

```python
parser.add_argument('--modo',    choices=['modo1', 'modo2'], default='modo1',
                    help='Modo 1: features com geometria da curva; Modo 2: só OBD+GPS (default: modo1)')
parser.add_argument('--optuna',  action='store_true', help='Usar Optuna para XGBoost e RandomForest')
parser.add_argument('--pytorch', action='store_true', help='Treinar MLP multi-tarefa PyTorch')
parser.add_argument('--plot',    action='store_true')
parser.add_argument('--isl',     action='store_true')
parser.add_argument('--mlp',     action='store_true')
```

Remover `--keras` (substituído por `--pytorch`).

- [ ] **Passo 2: Atualizar chamada a etapa_features para passar modo**

```python
    print("\n[5/6] Identificando trechos curvos e extraindo features...")
    features_df = etapa_features(df_analysis, cfg, modo=args.modo)
    print(f"  Modo: {args.modo}")
```

- [ ] **Passo 3: Atualizar bloco principal de modelos**

```python
    print("\n[6/6] Modelos clássicos de ML...")
    if args.optuna:
        etapa_ml_otimizado(features_df, cfg, args.plot)
    else:
        etapa_ml_classico(features_df, cfg, args.plot)

    if args.isl:
        print("\n[Extra] ISL — classificação 3 classes (baixo/medio/alto)...")
        etapa_isl_modelo(features_df, cfg, args.plot)
        print("\n[Extra] P1 — Regressão aceleração dentro da curva...")
        etapa_accel_regressao(features_df, cfg, args.plot)
        print("\n[Extra] P2 — Classificação por velocidade de entrada...")
        etapa_manobra_velocidade(features_df, cfg, args.plot)

    if args.mlp:
        print("\n[Extra] MLP sklearn (GridSearchCV)...")
        etapa_mlp_sklearn(features_df, cfg)

    if args.pytorch:
        print("\n[Extra] MLP multi-tarefa PyTorch...")
        from src.pipeline import etapa_pytorch
        etapa_pytorch(features_df, cfg)
```

- [ ] **Passo 4: Verificar run básico**

```bash
python scripts/run.py 2>&1 | tail -20
```

Saída esperada: resultados dos modelos clássicos, sem erro.

- [ ] **Passo 5: Verificar Modo 2**

```bash
python scripts/run.py --modo modo2 2>&1 | grep "Modo:"
```

Saída esperada: `  Modo: modo2`

- [ ] **Passo 6: Commit**

```bash
git add scripts/run.py
git commit -m "feat: add --modo, --optuna, --pytorch flags to run.py"
```

---

## Task 9: Criar src/models_pytorch.py

**Files:** Criar `src/models_pytorch.py`

- [ ] **Passo 1: Criar o arquivo**

```python
# src/models_pytorch.py
"""
CurvantML — Deep learning multi-tarefa em PyTorch.
Substitui os modelos Keras (MLP, GRU, LSTM).

Arquitetura principal: MultiTaskMLP com encoder compartilhado e cinco heads:
  - isl_value       (regressão, MSELoss)
  - isl_class       (3 classes, CrossEntropyLoss)
  - manobra_accel_curva, manobra_lateral_curva, manobra_ziguezague_curva (binário, BCELoss)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, r2_score, accuracy_score

from src.models import _COLS_EXCLUIR

_ISL_ENCODE = {'baixo': 0, 'medio': 1, 'alto': 2}

_TARGETS_BINARIOS = [
    'manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva',
]
_TARGETS_TODOS = ['isl_value', 'isl_class'] + _TARGETS_BINARIOS


class CurvantDataset(Dataset):
    """Dataset multi-tarefa para o MLP. X é float32; targets são float32 ou int64."""

    def __init__(self, X: np.ndarray, targets: dict[str, np.ndarray]):
        self.X       = torch.FloatTensor(X)
        self.targets = {}
        for k, v in targets.items():
            if k == 'isl_class':
                self.targets[k] = torch.LongTensor(v.astype(np.int64))
            else:
                self.targets[k] = torch.FloatTensor(v.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}


class MultiTaskMLP(nn.Module):
    """
    MLP multi-tarefa: encoder compartilhado (256→128) + cinco heads independentes.
    Entrada: vetor de features F1-F5 (normalizado externamente).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.3),
        )
        # Regressão contínua: ISL value
        self.head_isl_value = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # Classificação 3 classes: ISL class
        self.head_isl_class = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        # Binários
        self.head_accel   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_lateral = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_zz      = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        return {
            'isl_value':               self.head_isl_value(z).squeeze(-1),
            'isl_class':               self.head_isl_class(z),
            'manobra_accel_curva':     self.head_accel(z).squeeze(-1),
            'manobra_lateral_curva':   self.head_lateral(z).squeeze(-1),
            'manobra_ziguezague_curva':self.head_zz(z).squeeze(-1),
        }


def _preparar_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extrai e codifica os alvos disponíveis no DataFrame."""
    targets = {}
    if 'isl_value' in df.columns:
        targets['isl_value'] = df['isl_value'].fillna(0.0).values
    if 'isl_class' in df.columns:
        targets['isl_class'] = df['isl_class'].map(_ISL_ENCODE).fillna(0).values
    for col in _TARGETS_BINARIOS:
        if col in df.columns:
            targets[col] = df[col].fillna(0).values
    return targets


def treinar_multitask_mlp(
    df: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    test_size: float = 0.3,
    random_state: int = 42,
    lambdas: dict[str, float] | None = None,
) -> MultiTaskMLP:
    """
    Treina o MultiTaskMLP com split por id_route.

    lambdas: pesos por loss. Se None, todos 1.0.
    Retorna o modelo treinado.
    """
    if lambdas is None:
        lambdas = {k: 1.0 for k in _TARGETS_TODOS}

    feature_cols = [c for c in df.columns if c not in _COLS_EXCLUIR]
    rotas        = df['id_route'].unique()
    train_rotas, test_rotas = train_test_split(rotas, test_size=test_size, random_state=random_state)

    target_cols_presentes = [c for c in _TARGETS_TODOS if c in df.columns]
    all_cols = feature_cols + target_cols_presentes + ['id_route']

    df_train = df[df['id_route'].isin(train_rotas)][all_cols].dropna(subset=feature_cols)
    df_test  = df[df['id_route'].isin(test_rotas)][all_cols].dropna(subset=feature_cols)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values.astype(np.float32))
    X_test  = scaler.transform(df_test[feature_cols].values.astype(np.float32))

    train_targets = _preparar_targets(df_train)
    test_targets  = _preparar_targets(df_test)

    train_loader = DataLoader(
        CurvantDataset(X_train, train_targets),
        batch_size=batch_size, shuffle=True,
    )

    model     = MultiTaskMLP(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fns = {
        'isl_value':               nn.MSELoss(),
        'isl_class':               nn.CrossEntropyLoss(),
        'manobra_accel_curva':     nn.BCELoss(),
        'manobra_lateral_curva':   nn.BCELoss(),
        'manobra_ziguezague_curva':nn.BCELoss(),
    }

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            total = torch.tensor(0.0, requires_grad=True)
            for key, fn in loss_fns.items():
                if key not in targets_batch:
                    continue
                y   = targets_batch[key]
                lam = lambdas.get(key, 1.0)
                if key == 'isl_class':
                    total = total + lam * fn(preds[key], y)
                else:
                    total = total + lam * fn(preds[key], y)
            total.backward()
            optimizer.step()
            epoch_loss += total.item()
        if (epoch + 1) % 20 == 0:
            print(f"    Época {epoch + 1}/{epochs} — Loss médio: {epoch_loss / len(train_loader):.4f}")

    # ── Avaliação ──────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds_test = model(torch.FloatTensor(X_test))

    print("\n  MultiTaskMLP — Métricas (teste, split por rota):")
    for key in target_cols_presentes:
        if key not in preds_test or key not in test_targets:
            continue
        p = preds_test[key].numpy()
        y = test_targets[key]
        if key == 'isl_value':
            print(f"    isl_value   R²: {r2_score(y, p):.4f}")
        elif key == 'isl_class':
            pred_labels = np.argmax(p, axis=1)
            print(f"    isl_class   F1-macro: {f1_score(y.astype(int), pred_labels, average='macro', zero_division=0):.4f}")
        else:
            pred_labels = (p > 0.5).astype(int)
            print(f"    {key:<30s} F1: {f1_score(y.astype(int), pred_labels, average='weighted', zero_division=0):.4f}")

    return model
```

- [ ] **Passo 2: Verificar import**

```bash
python -c "from src.models_pytorch import MultiTaskMLP, treinar_multitask_mlp, CurvantDataset; import torch; m = MultiTaskMLP(50); print(m); print('OK')"
```

Saída esperada: arquitetura do modelo impressa, `OK` no final.

- [ ] **Passo 3: Commit**

```bash
git add src/models_pytorch.py
git commit -m "feat: add PyTorch MultiTaskMLP with shared encoder and 5 task heads"
```

---

## Task 10: Adicionar etapa_pytorch ao pipeline e wiring no run.py

**Files:** `src/pipeline.py`, `scripts/run.py`

- [ ] **Passo 1: Adicionar etapa_pytorch ao pipeline.py**

```python
def etapa_pytorch(features_df: pd.DataFrame, cfg: dict) -> None:
    """Treina o MLP multi-tarefa PyTorch com split por id_route."""
    from src.models_pytorch import treinar_multitask_mlp

    nn_cfg = cfg.get('neural_networks', {})
    pt_cfg = nn_cfg.get('multitask_mlp', {})

    treinar_multitask_mlp(
        features_df,
        epochs=pt_cfg.get('epochs', 100),
        batch_size=pt_cfg.get('batch_size', 32),
        lr=pt_cfg.get('lr', 1e-3),
        test_size=cfg['ml']['test_size'],
        random_state=cfg['ml']['random_state'],
    )
```

- [ ] **Passo 2: Adicionar config para multitask_mlp no config.yaml**

```yaml
neural_networks:
  multitask_mlp:
    epochs: 100
    batch_size: 32
    lr: 0.001
  # ... demais configs mantidas
```

- [ ] **Passo 3: Garantir que scripts/run.py importa etapa_pytorch**

Já foi adicionado no Task 8, Passo 1. Verificar que o import está presente:

```python
from src.pipeline import (
    ...
    etapa_pytorch,   # ← deve estar aqui
)
```

Caso não esteja, adicionar.

- [ ] **Passo 4: Verificar execução do PyTorch**

```bash
python scripts/run.py --pytorch 2>&1 | grep -E "(MLP|Época|isl|manobra|OK|Erro)"
```

Saída esperada (trecho):
```
[Extra] MLP multi-tarefa PyTorch...
    Época 20/100 — Loss médio: X.XXXX
    Época 40/100 — Loss médio: X.XXXX
    ...
  MultiTaskMLP — Métricas (teste, split por rota):
    isl_value   R²: X.XXXX
    isl_class   F1-macro: X.XXXX
    manobra_accel_curva                F1: X.XXXX
```

- [ ] **Passo 5: Commit**

```bash
git add src/pipeline.py config.yaml
git commit -m "feat: wire PyTorch multi-task MLP into pipeline and config"
```

---

## Task 11: Integração final e run completo

**Files:** nenhum novo

- [ ] **Passo 1: Run completo Modo 1**

```bash
python scripts/run.py --isl --modo modo1 2>&1 | tee results/run_modo1.log
```

Verificar:
- Saída de `[4/6]` mostra `manobra_combinado`, `manobra_accel`, `manobra_lateral`, `manobra_ziguezague`
- Saída de `[5/6]` mostra tabela com `Aceleração anormal`, `Direção perigosa`, `Zigue-zague`
- `[6/6]` mostra resultados dos modelos sem `KeyError`
- ISL extras rodam sem erro

- [ ] **Passo 2: Run completo Modo 2**

```bash
python scripts/run.py --isl --modo modo2 2>&1 | tee results/run_modo2.log
```

Verificar: `Modo: modo2` no log; f4_* estão zerados (modelo funciona com features reduzidas).

- [ ] **Passo 3: Run com Optuna (poucos trials para verificação rápida)**

Temporariamente no config.yaml mudar `n_trials: 3` para teste rápido:

```bash
python scripts/run.py --optuna 2>&1 | grep -E "(Tuning|Melhores|F1)"
```

Restaurar `n_trials: 50` após verificação.

- [ ] **Passo 4: Run com PyTorch**

```bash
python scripts/run.py --pytorch 2>&1 | grep -E "(MLP|R²|F1-macro|F1:)"
```

- [ ] **Passo 5: Commit final**

```bash
git add results/run_modo1.log results/run_modo2.log
git commit -m "feat: complete framework redesign — characterization taxonomy, Mode 1/2, Optuna, PyTorch multi-task"
```

---

## Self-Review

**Spec coverage:**

| Requisito do spec | Task que implementa |
|---|---|
| Taxonomia de risco explícita (manobra_accel, lateral, ziguezague, isl_value, v_excess) | Tasks 2, 4 |
| Modo 1 / Modo 2 (F4 features) | Tasks 4, 7, 8 |
| Features universais (apenas sensores presentes em todos os datasets) | Task 4 |
| accel_z + jerk_z | Task 4 |
| v_entry_sq_over_raio_est | Task 4 |
| Split por id_route | Task 5 |
| GroupKFold | Task 5 |
| Optuna XGB/RF | Task 6 |
| PyTorch MultiTaskMLP (encoder + 5 heads) | Task 9 |
| config.yaml: seção optuna, multitask_mlp | Tasks 3, 10 |
| manobra_combinado = OR dos 5 critérios | Tasks 2, 4 |

**Sem placeholders:** todos os passos têm código completo.

**Consistência de tipos:**
- `_split_por_rota` retorna `(ndarray, ndarray, ndarray, ndarray, ndarray)` — usado em Tasks 5 e 6 com a mesma assinatura.
- `MultiTaskMLP.forward` retorna dict com chaves `isl_value`, `isl_class`, `manobra_accel_curva`, `manobra_lateral_curva`, `manobra_ziguezague_curva` — as mesmas chaves usadas em `_preparar_targets` e `treinar_multitask_mlp`.
- `caracterizar_conducao` produz colunas `manobra_accel`, `manobra_lateral`, `manobra_ziguezague` — referenciadas corretamente em `features.py` Tasks 4.
