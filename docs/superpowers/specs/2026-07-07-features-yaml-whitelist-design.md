# Governança de features: `features.yaml` com whitelist por flag

Data: 2026-07-07

## Contexto e problema

Hoje o controle de features está espalhado em **três listas, dois arquivos e duas filosofias opostas**:

1. `NAO_FEATURES` (dict em `curvant/driving/features.py`) — o que nunca é feature (anti-leakage), agrupado por motivo (`id`/`target`/`in_curve`/`boundary`).
2. `config.features.desativar` (config.yaml) — lista **opt-out** (blacklist) aplicada a todas as flags tabulares.
3. `config.temporais.scalares_extras` + `config.temporais.sensors` (config.yaml) — whitelist **opt-in** do `--velocity`.

As flags tabulares (`--risk`, `--importance`) usam `colunas_features(df)` = *"tudo menos `NAO_FEATURES` menos `desativar`"* (blacklist). O `--velocity` usa uma whitelist explícita. Essa assimetria — e o fato de uma feature nova entrar automaticamente numa rota e não na outra — é a fonte da confusão.

## Objetivo

Uma **fonte única, honesta e uniforme** para a seleção de features: cada flag lista explicitamente as features que usa (whitelist), num arquivo dedicado. O que está na lista é o que o modelo vê — nada implícito.

## Decisões (validadas com o usuário)

- **Arquivo dedicado `features.yaml`** na raiz, ao lado do `config.yaml`. `config.yaml` continua com parâmetros de modelo/pipeline; `features.yaml` cuida só de quais colunas cada flag consome.
- **Whitelist por flag** — acaba o `desativar` (blacklist). Cada flag lista o que usa, como o `velocity` já faz.
- **`nao_features` removido de vez** — sem lista de proibidos. A whitelist é a verdade.
- **`--importance` herda a lista do `--risk`** — não tem lista própria (é a análise de importância do modelo de risco).
- **Rede de segurança**: nas funções de treino, `assert target not in feature_cols` — pega o caso de listar por engano o alvo entre as features.
- **Formato YAML** (consistente com o config.yaml e o loader existente).
- **Comportamento preservado**: as listas iniciais reproduzem exatamente as features usadas hoje.

## Estrutura do `features.yaml`

```yaml
# Governança de features do CurvantML.
# Fonte única de QUAIS colunas cada flag usa. Whitelist explícita: o que está
# listado é o que o modelo vê. config.yaml cuida de modelo/pipeline; aqui, das colunas.

flags:
  # ---- Flag tabular: matriz de features (uma lista chapada) ----
  risk:
    features:
      - vehicle_speed_mean
      - vehicle_speed_std
      - vehicle_speed_median
      - vehicle_speed_max
      - vehicle_speed_min
      - vehicle_speed_slope
      - vehicle_speed_cv
      - vehicle_speed_mean_tarde
      - vehicle_speed_slope_tarde
      - accel_x_mean
      - accel_x_std
      - accel_x_median
      - accel_x_max
      - accel_x_min
      - accel_x_slope
      - accel_x_mean_tarde
      - accel_x_slope_tarde
      - accel_y_mean
      - accel_y_std
      - accel_y_median
      - accel_y_max
      - accel_y_min
      - accel_y_slope
      - accel_y_mean_tarde
      - accel_y_slope_tarde
      - accelerator_pedal_pos_d_mean
      - accelerator_pedal_pos_d_std
      - accelerator_pedal_pos_d_median
      - accelerator_pedal_pos_d_max
      - accelerator_pedal_pos_d_min
      - accelerator_pedal_pos_d_slope
      - accelerator_pedal_pos_d_cv
      - accelerator_pedal_pos_d_mean_tarde
      - accelerator_pedal_pos_d_slope_tarde
      - jerk_x_max
      - jerk_x_std
      - jerk_y_max
      - jerk_y_std
      - distance_car_curve
      - n_perigo_accel_janela
      - janela_abs_accel_max
      - n_perigo_lateral_janela
      - janela_bearing_std
      - janela_n_mudancas_dir
      - janela_bearing_range
      - janela_raio_min
      - janela_raio_mean
      - janela_raio_last
      - throttle_off_distance
      - f4_raio_min
      - f4_raio_mean
      - f4_dnit_num
      - n_curvas_antes
      - prev_isl_max
      - mean_isl_antes
      - prev_raio_min
      - prev_raio_mean
      - prev_dnit_num
      - mc_p_baixo
      - mc_p_medio
      - mc_p_alto

  # ---- Flag de análise: herda a lista do risk (sem lista própria) ----
  importance:
    herda: risk

  # ---- Flag temporal: dois tipos de entrada (não achata numa lista só) ----
  velocity:
    sensors:                      # canais de série temporal (50 timesteps)
      - vehicle_speed
      - accel_x
      - accel_y
      - engine_rpm
      - accelerator_pedal_pos_d
    scalares_extras:              # escalares fixos por curva (canais constantes)
      - janela_raio_min
      - janela_raio_mean
      - janela_raio_last
      - jerk_y_max
      - n_perigo_accel_janela
      - n_perigo_lateral_janela
      - f4_raio_min
      - f4_raio_mean
      - f4_dnit_num
      - mc_p_baixo
      - mc_p_medio
      - mc_p_alto
      - mean_isl_antes
      - prev_isl_max
      - n_curvas_antes
```

Notas:
- A lista do `risk` é exatamente o `colunas_features()` de hoje **menos** `accel_x_cv` e `accel_y_cv` (as duas hoje em `desativar`), reproduzindo o comportamento atual. São 61 features (63 produzidas − 2).
- O `velocity` mantém `sensors` + `scalares_extras` porque são coisas semanticamente distintas (série temporal vs escalar por curva). Igual a hoje.
- `importance: { herda: risk }` evita duplicar 61 itens; o código resolve a herança.

## O que sai do `config.yaml`

- `features.desativar` — **removido** (fim da blacklist).
- `temporais.sensors` e `temporais.scalares_extras` — **movidos** para `flags.velocity` no `features.yaml`.
- **Toda a seção `features:`** (janela, `lead_gap`, `vars_sensor`) — **movida** para `features.yaml > extracao`.
  (Decisão posterior: como o `features.yaml` é a casa das features, os parâmetros de extração
  também pertencem a ele. O `config.yaml` deixa de ter seção `features:`.)

Permanece no `config.yaml`:
- `temporais.*` restante (`model`, `target`, `task`, hiperparâmetros, `n_timesteps`, etc.) — configuração do modelo.

## Mudanças no código

### 1. Loader — `curvant/utils/config.py`
Adicionar `carregar_features_config()` espelhando `carregar_config()`:
```python
def carregar_features_config() -> dict:
    """Lê features.yaml da raiz do projeto (governança de features por flag)."""
    with open('features.yaml', encoding='utf-8') as f:
        return yaml.safe_load(f)
```
Se `features.yaml` não existir, erro claro com orientação (mantém a fonte única honesta; não cai num default oculto).

### 2. `curvant/driving/features.py`
- Remover o dict `NAO_FEATURES` e a lógica de blacklist.
- Remover `configurar_features_desativadas()` e o global `_FEATURES_DESATIVADAS`.
- `colunas_features(df)` deixa de derivar features por subtração. Substituir por uma resolução baseada em whitelist:
  - `features_de_flag(feat_cfg, flag, df=None) -> list[str]` — resolve a lista da flag (seguindo `herda`), e (se `df` fornecido) intersecta com as colunas presentes, avisando sobre listadas ausentes.
  - Manter uma função utilitária que, dada a lista e o `df`, devolve as colunas válidas.
- A extração de features (`extrair_features`) **não muda** — ela produz todas as colunas; a seleção é feita na hora de treinar.

### 3. `curvant/cli.py`
- No startup: `feat_cfg = carregar_features_config()`.
- Passar a lista da flag para cada etapa:
  - `--risk` → `etapa_ml_classico/otimizado/criterios_separados(features_df, cfg, plot, feature_list=features_de_flag(feat_cfg, 'risk', features_df))`.
  - `--importance` → resolve `importance` (herda `risk`).
  - `--velocity` → passa `sensors` e `scalares_extras` de `feat_cfg['flags']['velocity']` para `etapa_regressao_ts`.

### 4. `curvant/models/tabular.py` e `curvant/pipeline.py`
- Funções de treino tabular (`aplicar_modelos_ml`, `aplicar_modelos_ml_otimizados`, splits, `etapa_criterios_separados`, `etapa_importancia_features`) recebem `feature_cols`/`feature_list` explícito em vez de chamar `colunas_features(df)` internamente.
- Os helpers de split (`_split_por_rota`, `_split_rota_generico`) já aceitam `feature_cols`; ajustar os call sites que hoje derivam de `colunas_features(df)` para receber a whitelist da flag.

### 5. `curvant/models/temporais.py`
- `treinar_regressao_ts` lê `sensors`/`scalares_extras` do `features.yaml` (recebidos por parâmetro) em vez de `cfg['temporais']`.
- A validação atual `cols_validas = colunas_features(features_df)` some (não há mais blacklist); no lugar, valida que cada `scalar_extra` existe no `features_df` (aviso se faltar).

### 6. Rede de segurança (anti-leakage)
Em cada função de treino, após montar `feature_cols` e conhecer o `target`:
```python
if target in feature_cols:
    raise ValueError(
        f"Leakage: o alvo '{target}' está na lista de features de '{flag}' "
        f"(features.yaml). Remova-o da whitelist."
    )
```
Cobre exatamente o modo de falha da whitelist manual.

## Preservação de comportamento

- `--risk` e `--importance`: mesma matriz de features de hoje (61 colunas; as 63 produzidas menos `accel_x_cv`/`accel_y_cv`).
- `--velocity`: mesmos `sensors` e `scalares_extras`.
- Verificar rodando cada flag antes/depois e comparando as colunas efetivamente usadas (log "Canais/Features").

## Documentação a atualizar

- `CLAUDE.md`: seção "Lista do que não é feature" e "Seleção de features por flag" reescritas para descrever `features.yaml` (whitelist por flag, sem `desativar`/`nao_features`). Tabela de config e árvore de arquivos.
- `README.md`: menção à seleção de features / `desativar`, se houver.
- `docs/TARGETS_E_FEATURES.md`: nota sobre onde a seleção por flag vive agora; atualizar contagem/lista se citar `NAO_FEATURES`.

## Fora de escopo

- Unificar a estrutura do `velocity` (sensors+scalares) com a do tabular (lista chapada) — são entradas genuinamente diferentes; mantidas separadas de propósito.
- Grupos de features como unidade de liga/desliga — não pedido; YAGNI.
- Qualquer mudança na *extração* de features (o que é computado).
