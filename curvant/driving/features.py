import numpy as np
import pandas as pd

from curvant.constants import G as _G, MU as _MU_PADRAO, ISL_ALTO
from curvant.driving.isl import classificar_isl
from curvant.driving.risk_measures import calcular_bearing as _calcular_bearing_gps

# Encoding numérico da classe DNIT para feature prev_dnit_num
_DNIT_NUM = {'suave': 0, 'aberta': 1, 'media': 2, 'fechada': 3, 'muito_fechada': 4}

# Seleção de features por flag (whitelist).
#
# A verdade sobre quais colunas cada flag usa mora em features.yaml. Antes de
# treinar, o CLI resolve a lista da flag (resolver_features_flag) e a instala na
# global _FEATURES_ATIVAS (configurar_features_ativas). colunas_features() então
# devolve exatamente essas colunas, na ordem listada, ignorando as ausentes.

_FEATURES_ATIVAS: list[str] | None = None


def resolver_features_flag(feat_cfg: dict, flag: str) -> list[str]:
    """Lista de features de uma flag no features.yaml, seguindo 'herda'."""
    flags = feat_cfg.get('flags', {})
    if flag not in flags:
        raise KeyError(f"Flag '{flag}' não encontrada em features.yaml (flags: {list(flags)}).")
    conf = flags[flag] or {}
    if 'herda' in conf:
        return resolver_features_flag(feat_cfg, conf['herda'])
    lista = conf.get('features')
    if lista is None:
        raise KeyError(f"Flag '{flag}' não define 'features' nem 'herda' em features.yaml.")
    return list(lista)


def configurar_features_ativas(lista: list[str] | None) -> None:
    """Instala a whitelist de features da flag em execução (usada por colunas_features)."""
    global _FEATURES_ATIVAS
    _FEATURES_ATIVAS = list(lista) if lista is not None else None


def colunas_features(df: pd.DataFrame) -> list[str]:
    """Features de df segundo a whitelist ativa (features.yaml), na ordem listada."""
    if _FEATURES_ATIVAS is None:
        raise RuntimeError(
            "Whitelist de features não configurada. Chame configurar_features_ativas() "
            "com a lista da flag (features.yaml) antes de treinar."
        )
    presentes = [c for c in _FEATURES_ATIVAS if c in df.columns]
    ausentes  = [c for c in _FEATURES_ATIVAS if c not in df.columns]
    if ausentes:
        print(f"  [features] AVISO: {len(ausentes)} feature(s) da whitelist ausente(s) no dataframe: {ausentes}")
    return presentes


def checar_leakage(feature_cols: list[str], target: str) -> None:
    """Falha se o alvo estiver entre as features (whitelist mal configurada = leakage)."""
    if target in feature_cols:
        raise ValueError(
            f"Leakage: o alvo '{target}' está na whitelist de features. Remova-o de features.yaml."
        )


# Resolução de parâmetros

def _resolver_vars_sensor(vars_sensor: list | None) -> list:
    """vars_sensor explícito -> features.yaml (extracao.vars_sensor) -> default."""
    if vars_sensor:
        return vars_sensor
    try:
        from curvant.utils.config import carregar_features_config

        fc = carregar_features_config()
        cfg_vars = fc.get('extracao', {}).get('vars_sensor') if isinstance(fc, dict) else None
        if cfg_vars and isinstance(cfg_vars, list) and len(cfg_vars) > 0:
            return cfg_vars
    except Exception:
        pass
    return ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y']


# Janela pré-curva

def _montar_janela_precurva(
    df: pd.DataFrame,
    id_route: str,
    dist_entrada: float,
    lead_gap: float,
    janela_distancia: float | None,
    janela_acel_confort: float,
    janela_distancia_min: float,
    janela_distancia_max: float,
) -> pd.DataFrame:
    """
    Seleciona os pontos da janela pré-curva — o trecho antes do ponto de decisão,
    que fica recuado da entrada por lead_gap (predição antecipada).

    Tamanho da janela: distância fixa (janela_distancia) ou dinâmica baseada na
    distância de frenagem confortável d = v²/(2·a_confort). Remove pontos que
    pertençam a uma curva anterior. Retorna o DataFrame da janela (pode ser vazio).
    """
    dist_decisao = dist_entrada - lead_gap

    if janela_distancia is not None:
        d_janela = float(janela_distancia)
    else:
        pontos_antes = df[
            (df['id_route'] == id_route)
            & (df['distancia_acumulada'] < dist_decisao)
        ].tail(5)
        v_ms = (
            float(pontos_antes['vehicle_speed'].mean()) / 3.6
            if not pontos_antes.empty else 60.0 / 3.6
        )
        d_janela = float(np.clip(
            v_ms ** 2 / (2.0 * janela_acel_confort),
            janela_distancia_min,
            janela_distancia_max,
        ))

    janela = df[
        (df['id_route'] == id_route)
        & (df['distancia_acumulada'] < dist_decisao)
        & (df['distancia_acumulada'] >= dist_decisao - d_janela)
    ].copy()

    # remove pontos pertencentes a uma curva anterior dentro da janela
    if 'curva' in janela.columns:
        janela = janela[~janela['curva']].copy()

    return janela


# Grupos de features (janela pré-curva)

def _features_sensores(janela: pd.DataFrame, vars_sensor: list) -> dict:
    """
    Grupo 1 — estatísticas dos sensores OBD na janela pré-curva.
    Para cada sensor: mean/std/median/max/min/slope/cv + mean_tarde/slope_tarde
    (a metade final da janela, mais perto da curva).
    """
    out: dict = {}
    t = janela['time_sec'].values - janela['time_sec'].values[0]

    mid = max(len(janela) // 2, 1)
    janela_tarde = janela.iloc[mid:]
    t_tarde = (
        janela_tarde['time_sec'].values - janela_tarde['time_sec'].values[0]
        if len(janela_tarde) > 0 else np.array([0.0])
    )

    for var in vars_sensor:
        vals = janela[var].values
        out[f'{var}_mean']   = float(np.mean(vals))
        out[f'{var}_std']    = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f'{var}_median'] = float(np.median(vals))
        out[f'{var}_max']    = float(np.max(vals))
        out[f'{var}_min']    = float(np.min(vals))
        out[f'{var}_slope']  = float(np.polyfit(t, vals, 1)[0]) if len(t) >= 2 else 0.0
        out[f'{var}_cv']     = (
            float(out[f'{var}_std'] / out[f'{var}_mean'])
            if out[f'{var}_mean'] != 0 else 0.0
        )
        if len(janela_tarde) >= 2:
            vals_t = janela_tarde[var].values
            out[f'{var}_mean_tarde']  = float(np.mean(vals_t))
            out[f'{var}_slope_tarde'] = float(np.polyfit(t_tarde, vals_t, 1)[0])
        else:
            out[f'{var}_mean_tarde']  = out[f'{var}_mean']
            out[f'{var}_slope_tarde'] = out[f'{var}_slope']

    return out


def _features_bearing_janela(janela: pd.DataFrame) -> dict:
    """
    Grupo 2 (bearing) — oscilação de direção na janela pré-curva.
    Precursores diretos do critério de zigue-zague: variação de bearing e
    contagem de alternâncias de direção na aproximação.
    """
    out = {
        'janela_bearing_std':      0.0,
        'janela_n_mudancas_dir':   0,
        'janela_bearing_range':    0.0,
    }

    cols_ok = {'lat', 'lon', 'vehicle_speed'}.issubset(janela.columns)
    if not cols_ok or len(janela) < 3:
        return out

    janela_mov = janela[janela['vehicle_speed'] >= 5.0].reset_index(drop=True)
    if len(janela_mov) < 3:
        return out

    lats = janela_mov['lat'].tolist()
    lons = janela_mov['lon'].tolist()
    bearings = np.array([
        _calcular_bearing_gps(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ])

    deltas = np.array([
        (bearings[i] - bearings[i - 1] + 180) % 360 - 180
        for i in range(1, len(bearings))
    ])

    out['janela_bearing_std']   = float(np.std(deltas)) if len(deltas) > 1 else 0.0
    out['janela_bearing_range'] = float(np.ptp(deltas))  # range sobre deltas circulares, não bearings brutos

    # mesma lógica de alternância do critério de zigue-zague (limiar 15°)
    contador, ultimo_sinal = 0, 0
    for d in deltas:
        if abs(d) > 15.0:
            sinal = int(np.sign(d))
            if sinal != ultimo_sinal:
                contador    += 1
                ultimo_sinal = sinal
    out['janela_n_mudancas_dir'] = contador

    return out


def _features_dinamica(janela: pd.DataFrame) -> dict:
    """
    Grupo 2 — dinâmica derivada da aproximação:
    jerk (variação brusca da aceleração), comprimento da janela, contagem de
    eventos onde a aceleração ultrapassa um limite e oscilação de direção (bearing).
    """
    out: dict = {}

    for var in ['accel_x', 'accel_y']:
        vals = janela[var].values
        dt_arr = np.diff(janela['time_sec'].values)
        dt_arr = np.where(dt_arr > 0, dt_arr, 1e-3)
        jerk = np.diff(vals) / dt_arr
        out[f'jerk_{var[-1]}_max'] = float(np.abs(jerk).max()) if len(jerk) > 0 else 0.0
        out[f'jerk_{var[-1]}_std'] = float(np.std(jerk))       if len(jerk) > 1 else 0.0

    # distance_car_curve = comprimento da janela pré-curva (distância coberta pela
    # aproximação). NÃO é a distância até a curva — a janela termina lead_gap metros
    # antes da entrada (ver TARGETS_E_FEATURES.md).
    out['distance_car_curve'] = float(
        janela['distancia_acumulada'].max() - janela['distancia_acumulada'].min()
    )

    _kamm_lim = 0.7 * _MU_PADRAO * _G
    _accel_total = np.sqrt(janela['accel_x'].values**2 + janela['accel_y'].values**2)
    out['n_perigo_accel_janela']   = int((_accel_total > _kamm_lim).sum())
    out['janela_abs_accel_max']    = float(_accel_total.max())
    out['n_perigo_lateral_janela'] = int((janela['accel_y'].abs() > 2.0).sum())

    out.update(_features_bearing_janela(janela))

    return out


def _features_geometria_janela(janela: pd.DataFrame) -> dict:
    """
    Grupo 3 — raio de curvatura na janela pré-curva.
    Atenção: a B-spline é ajustada ao trajeto inteiro, então a curvatura aqui é
    influenciada pelos pontos da curva à frente (spline global + filtro Gaussiano).
    """
    nan3 = {'janela_raio_min': np.nan, 'janela_raio_mean': np.nan, 'janela_raio_last': np.nan}
    if 'raio_curvatura' not in janela.columns:
        return nan3

    raios = janela['raio_curvatura']
    raios = raios[np.isfinite(raios) & (raios > 0)].clip(upper=2000.0)
    if len(raios) == 0:
        return nan3

    return {
        'janela_raio_min':  float(raios.min()),
        'janela_raio_mean': float(raios.mean()),
        'janela_raio_last': float(raios.iloc[-1]),  # último ponto = mais perto da curva
    }


# Alvos (medidos dentro da curva)

def _alvos_manobra(curva: pd.DataFrame) -> dict:
    """Targets de classificação de manobra — disparou cada critério na curva?"""
    out = {
        'manobra_accel_curva':      1 if curva['manobra_accel'].any() else 0,
        'manobra_lateral_curva':    1 if curva['manobra_lateral'].any() else 0,
        'manobra_ziguezague_curva': 1 if curva['manobra_ziguezague'].any() else 0,
        'manobra_combinado_curva':  1 if curva['manobra_combinado'].any() else 0,
    }
    out['manobra'] = out['manobra_combinado_curva']  # retrocompat
    return out


def _alvos_isl(pts_curva: pd.DataFrame) -> dict:
    """
    Targets de ISL e velocidade crítica.
    - ISL cinemático: v²/(R·g·μ) = ctp_accel/(g·μ) — depende do raio GPS (B-spline).
    - ISL via sensor: |accel_y|/(g·μ) — não depende do raio, mais robusto a GPS ruidoso.
    - v_critica: velocidade real no ponto de pico do ISL (prever isto equivale a prever ISL).
    """
    out: dict = {}

    if 'ctp_accel' in pts_curva.columns:
        isl_vals = pts_curva['ctp_accel'].abs() / (_G * _MU_PADRAO)
        isl_max  = float(isl_vals.max())
        out['isl_mean']  = float(isl_vals.mean())
        out['isl_max']   = isl_max
        out['isl_class'] = classificar_isl(isl_max)
        out['isl_alto']  = 1 if isl_max >= ISL_ALTO else 0
        idx_max          = isl_vals.idxmax()
        out['v_critica'] = float(pts_curva.loc[idx_max, 'vehicle_speed'])  # km/h
    else:
        out['isl_mean'] = out['isl_max'] = out['isl_class'] = out['isl_alto'] = np.nan
        out['v_critica'] = np.nan

    if 'accel_y' in pts_curva.columns:
        isl_s = pts_curva['accel_y'].abs() / (_G * _MU_PADRAO)
        out['isl_sensor_max']   = float(isl_s.max())
        out['isl_sensor_mean']  = float(isl_s.mean())
        out['isl_sensor_class'] = classificar_isl(float(isl_s.max()))
    else:
        out['isl_sensor_max'] = out['isl_sensor_mean'] = out['isl_sensor_class'] = np.nan

    return out



def _alvos_pedal(pts_curva: pd.DataFrame) -> dict:
    """Targets de comportamento do acelerador dentro da curva."""
    col = 'accelerator_pedal_pos_d'
    vazio = {'accel_pedal_min_curva': np.nan, 'accel_pedal_mean_curva': np.nan}
    if col not in pts_curva.columns:
        return vazio
    vals = pts_curva[col].dropna()
    if vals.empty:
        return vazio
    return {
        'accel_pedal_min_curva':  float(vals.min()),
        'accel_pedal_mean_curva': float(vals.mean()),
    }


def _features_pedal_janela(janela: pd.DataFrame, dist_entrada: float) -> dict:
    """
    Feature comportamental: distância (m) antes da entrada em que o motorista
    fechou o acelerador pela última vez (pedal < 10%).
    Zero = nunca fechou na janela (entrou acelerando).
    Maior valor = fechou o acelerador mais cedo = abordagem mais controlada.
    """
    col = 'accelerator_pedal_pos_d'
    if col not in janela.columns or janela[col].isna().all():
        return {'throttle_off_distance': np.nan}
    fechado = janela[janela[col] < 10.0]
    if fechado.empty:
        return {'throttle_off_distance': 0.0}
    dist_ultimo_fechado = float(fechado['distancia_acumulada'].max())
    return {'throttle_off_distance': max(0.0, dist_entrada - dist_ultimo_fechado)}


def _geometria_curva(pts_curva: pd.DataFrame) -> dict:
    """
    Grupo 4 — geometria real da curva à frente.
    curve_* são auxiliares (excluídas das features); f4_* são as cópias usadas
    como feature (a rota é conhecida, então a geometria à frente é legítima).
    """
    out: dict = {}

    if 'raio_curvatura' in pts_curva.columns:
        out['curve_raio_min']  = float(pts_curva['raio_curvatura'].min())
        out['curve_raio_mean'] = float(pts_curva['raio_curvatura'].mean())
    else:
        out['curve_raio_min'] = out['curve_raio_mean'] = np.nan

    if 'classe_dnit' in pts_curva.columns:
        dnit_mode = pts_curva['classe_dnit'].mode()
        out['curve_dnit_num'] = _DNIT_NUM.get(
            dnit_mode.iloc[0] if not dnit_mode.empty else 'suave', 0
        )
    else:
        out['curve_dnit_num'] = 0

    out['f4_raio_min']  = out['curve_raio_min']
    out['f4_raio_mean'] = out['curve_raio_mean']
    out['f4_dnit_num']  = out['curve_dnit_num']

    return out


def _adicionar_contexto(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Grupo 5 — contexto das curvas anteriores, por rota (shift garante ausência de
    leakage: usa só as curvas já percorridas, nunca a atual).
    """
    partes = []
    raio_min_global  = df_out['curve_raio_min'].median()
    raio_mean_global = df_out['curve_raio_mean'].median()

    for _, grupo in df_out.groupby('id_route', sort=False):
        grupo = grupo.copy()

        grupo['n_curvas_antes'] = np.arange(len(grupo))
        # ISL acumulado das curvas anteriores — medido pelos sensores (v²/R·g·μ),
        # disponível em tempo real sem depender do rótulo Segura/Risco do modelo.
        isl_shifted = grupo['isl_max'].shift(1)
        grupo['prev_isl_max']   = isl_shifted.fillna(0.0).round(3)
        grupo['mean_isl_antes'] = isl_shifted.expanding().mean().fillna(0.0).round(3)

        grupo['prev_raio_min']  = grupo['curve_raio_min'].shift(1).fillna(raio_min_global)
        grupo['prev_raio_mean'] = grupo['curve_raio_mean'].shift(1).fillna(raio_mean_global)
        grupo['prev_dnit_num']  = grupo['curve_dnit_num'].shift(1).fillna(0).astype(int)

        partes.append(grupo)

    return pd.concat(partes, ignore_index=True)


# Orquestração

def extrair_features(
    data: pd.DataFrame,
    janela_distancia: float | None = None,
    janela_acel_confort: float = 2.5,
    janela_distancia_min: float = 50.0,
    janela_distancia_max: float = 400.0,
    lead_gap: float = 0.0,
    vars_sensor: list | None = None,
) -> pd.DataFrame:
    """
    Extrai, por curva, as features da janela pré-curva e os targets medidos dentro
    da curva. Monta a janela e delega cada grupo a uma função dedicada:

      _features_sensores        Grupo 1 — estatísticas dos sensores OBD
      _features_dinamica        Grupo 2 — jerk, tamanho da janela, contagem de eventos
      _features_geometria_janela Grupo 3 — raio na janela pré-curva
      _geometria_curva          Grupo 4 — geometria real da curva à frente (f4_*)
      _adicionar_contexto       Grupo 5 — contexto das curvas anteriores (post-hoc)

      _alvos_manobra / _alvos_isl — targets dentro da curva

    Janela pré-curva: distância fixa (janela_distancia) ou dinâmica
    d = v²/(2·a_confort) ∈ [janela_distancia_min, janela_distancia_max]. A janela
    termina lead_gap metros antes da entrada (predição antecipada).
    """
    df = data.copy()
    df['conducao'] = df['conducao'].map({'Perigosa': 1, 'Segura': 0})
    df = df.sort_values(by=['id_route', 'time_sec'])
    vars_sensor = _resolver_vars_sensor(vars_sensor)

    dados_janela = []
    for (id_route_atual, trecho_curvo), curva in df.groupby(['id_route', 'trecho_curvo']):
        if trecho_curvo == 0:
            continue
        if len(curva) <= 2:
            continue

        dist_entrada = curva['distancia_acumulada'].min()
        janela = _montar_janela_precurva(
            df, id_route_atual, dist_entrada, lead_gap, janela_distancia,
            janela_acel_confort, janela_distancia_min, janela_distancia_max,
        )
        if len(janela) < 3:
            continue

        row: dict = {
            'time_inicio':     janela['time_sec'].min(),
            'time_fim':        janela['time_sec'].max(),
            'id_route':        id_route_atual,
            'id_trecho_curvo': trecho_curvo,
        }
        row.update(_features_sensores(janela, vars_sensor))
        row.update(_features_dinamica(janela))
        row.update(_features_geometria_janela(janela))
        row.update(_features_pedal_janela(janela, dist_entrada))
        row.update(_alvos_manobra(curva))

        pts_curva = curva[curva['curva'] == True] if 'curva' in curva.columns else curva
        if pts_curva.empty:
            pts_curva = curva
        row.update(_alvos_isl(pts_curva))
        row.update(_alvos_pedal(pts_curva))
        row.update(_geometria_curva(pts_curva))

        dados_janela.append(row)

    df_out = (
        pd.DataFrame(dados_janela)
        .sort_values(['id_route', 'time_inicio'])
        .reset_index(drop=True)
    )

    return _adicionar_contexto(df_out)
