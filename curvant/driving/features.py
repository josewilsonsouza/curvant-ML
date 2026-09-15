import numpy as np
import pandas as pd

from curvant.constants import G as _G, MU as _MU_PADRAO, RAIO_MIN_ISL
from curvant.driving.isl import classificar_isl

def calcular_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Ângulo de direção (bearing) entre dois pontos GPS, em graus [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


_FEATURES_ATIVAS: list[str] | None = None

def resolver_features_flag(feat_cfg: dict, flag: str) -> list[str]:
    """Lista de features de uma flag no features.yaml."""
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
    """Instala a lista de features da flag em execução"""
    global _FEATURES_ATIVAS
    _FEATURES_ATIVAS = list(lista) if lista is not None else None


def colunas_features(df: pd.DataFrame) -> list[str]:
    """Features de df segundo a lista da flag ativa"""
    if _FEATURES_ATIVAS is None:
        raise RuntimeError(
            "Lista de features da flag não configurada."
        )
    presentes = [c for c in _FEATURES_ATIVAS if c in df.columns]
    ausentes  = [c for c in _FEATURES_ATIVAS if c not in df.columns]
    if ausentes:
        print(f"  [features] AVISO: {len(ausentes)} feature(s) da flag ausente(s) no dataframe: {ausentes}")
    return presentes


def checar_leakage(feature_cols: list[str], target: str) -> None:
    """Falha se o target estiver entre as features."""
    if target in feature_cols:
        raise ValueError(
            f"Leakage: o target '{target}' está na lista de features da flag. Remova-o de features.yaml."
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
    return ['vehicle_speed']


# Janela pré-curva

def _montar_janela_precurva(
    df: pd.DataFrame,
    id_route: str,
    dist_entrada: float,
    lead_gap: float,
    precurva_distancia: float | None,
    precurva_desacel_confort: float,
    precurva_distancia_min: float,
    precurva_distancia_max: float,
) -> pd.DataFrame:
    """
    Seleciona os pontos da janela pré-curva.
    Tamanho da janela precurva: distância fixa (precurva_distancia) ou dinâmica baseada na
    distância de frenagem confortável d = v^2/(2*a_confort). Remove pontos que
    pertençam a uma curva anterior. Retorna o DataFrame da janela (pode ser vazio).
    """
    dist_decisao = dist_entrada - lead_gap

    if precurva_distancia is not None:
        d_janela = float(precurva_distancia)
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
            v_ms ** 2 / (2.0 * precurva_desacel_confort),
            precurva_distancia_min,
            precurva_distancia_max,
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


# Features da janela de aproximação

def _features_sensores(janela: pd.DataFrame, vars_sensor: list) -> dict:
    """Estatísticas de cada sensor na janela de aproximação.

    As versões "tarde" repetem a conta só na metade final da janela, que é o trecho
    mais perto da entrada da curva.
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
    """Oscilação de direção na aproximação: o quanto o rumo variou e quantas vezes
    o motorista trocou de lado."""
    out = {'precurva_n_mudancas_dir': 0, 'precurva_bearing_range': 0.0}

    cols_ok = {'lat', 'lon', 'vehicle_speed'}.issubset(janela.columns)
    if not cols_ok or len(janela) < 3:
        return out

    janela_mov = janela[janela['vehicle_speed'] >= 5.0].reset_index(drop=True)
    if len(janela_mov) < 3:
        return out

    lats = janela_mov['lat'].tolist()
    lons = janela_mov['lon'].tolist()
    bearings = np.array([
        calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ])

    deltas = np.array([
        (bearings[i] - bearings[i - 1] + 180) % 360 - 180
        for i in range(1, len(bearings))
    ])

    # amplitude sobre as variações, não sobre os bearings brutos, que dão a volta em 360
    out['precurva_bearing_range'] = float(np.ptp(deltas))

    contador, ultimo_sinal = 0, 0
    for d in deltas:
        if abs(d) > 15.0:
            sinal = int(np.sign(d))
            if sinal != ultimo_sinal:
                contador    += 1
                ultimo_sinal = sinal
    out['precurva_n_mudancas_dir'] = contador

    return out


def _features_geometria_janela(janela: pd.DataFrame) -> dict:
    """Quão sinuoso já era o trecho antes da curva.

    A B-spline é ajustada ao trajeto inteiro, então a curvatura medida aqui sofre
    alguma influência dos pontos da curva que vem à frente.
    """
    vazio = {'precurva_raio_min': np.nan, 'precurva_raio_mean': np.nan, 'precurva_raio_last': np.nan}
    if 'raio_curvatura' not in janela.columns:
        return vazio

    raios = janela['raio_curvatura']
    raios = raios[np.isfinite(raios) & (raios > 0)].clip(upper=2000.0)
    if len(raios) == 0:
        return vazio

    return {
        'precurva_raio_min':  float(raios.min()),
        'precurva_raio_mean': float(raios.mean()),
        'precurva_raio_last': float(raios.iloc[-1]),   # o ponto mais perto da curva
    }

# Alvos (medidos dentro da curva)

def _alvos_correcao(curva: pd.DataFrame) -> dict:
    """Target de correção tardia: o motorista freou forte dentro da curva?

    A flag de indefinição marca as curvas na faixa morta do limiar, onde o rótulo
    binário é decidido pelo ruído da medida. Serve para descartá-las do treino,
    nunca como feature.
    """
    return {
        'correcao_tardia_curva':     1 if curva['correcao_tardia'].any() else 0,
        'correcao_indefinida_curva': 1 if 'correcao_indefinida' in curva.columns
                                       and curva['correcao_indefinida'].any() else 0,
    }

def _alvos_isl(pts_curva: pd.DataFrame) -> dict:
    """Quanto a curva exigiu da aderência, e a que velocidade.

    O ISL é v²/(R·g·μ) ponto a ponto. A curva é resumida pelo percentil 95, não pelo
    máximo: o máximo se deixa levar por um único raio espúrio da B-spline, que faz
    v²/R explodir. Pela mesma razão o raio recebe um piso próprio antes da conta.

    A v_critica é a velocidade real no ponto de maior ISL, então prever uma equivale
    a prever a outra.
    """
    out: dict = {}

    if {'raio_curvatura', 'vehicle_speed'} <= set(pts_curva.columns):
        v_ms     = pts_curva['vehicle_speed'].values / 3.6
        raio     = np.clip(pts_curva['raio_curvatura'].abs().values, RAIO_MIN_ISL, None)
        isl_vals = (v_ms ** 2) / (raio * _G * _MU_PADRAO)
        isl_p95  = float(np.percentile(isl_vals, 95))
        out['isl_max']   = float(isl_vals.max())
        out['isl_p95']   = isl_p95
        out['isl_class'] = classificar_isl(isl_p95)
        idx_max          = int(np.argmax(isl_vals))
        out['v_critica'] = float(pts_curva['vehicle_speed'].values[idx_max])
    else:
        out['isl_max'] = out['isl_p95'] = out['isl_class'] = np.nan
        out['v_critica'] = np.nan

    return out

def _geometria_curva(pts_curva: pd.DataFrame) -> dict:
    """Geometria da curva que vem pela frente.

    Entra como feature porque a rota é conhecida de antemão. O raio, porém, não vem
    de mapa: sai da mesma B-spline que gera o rótulo de ISL, então feature e alvo
    compartilham o ruído do GPS. Ver docs/TARGETS_E_FEATURES.md.
    """
    if 'raio_curvatura' not in pts_curva.columns:
        return {'curva_raio_min': np.nan, 'curva_raio_mean': np.nan}

    raios = pts_curva['raio_curvatura']
    return {'curva_raio_min': float(raios.min()), 'curva_raio_mean': float(raios.mean())}


def _adicionar_contexto(df_out: pd.DataFrame) -> pd.DataFrame:
    """O que o motorista já enfrentou nesta rota até aqui.

    O shift garante que só entram curvas já percorridas, nunca a atual.
    """
    partes = []
    raio_global = df_out['curva_raio_min'].median()

    for _, grupo in df_out.groupby('id_route', sort=False):
        grupo = grupo.copy()
        grupo['n_curvas_antes'] = np.arange(len(grupo))

        isl_anterior = grupo['isl_max'].shift(1)
        grupo['prev_isl_max']   = isl_anterior.fillna(0.0).round(3)
        grupo['mean_isl_antes'] = isl_anterior.expanding().mean().fillna(0.0).round(3)
        grupo['prev_raio_min']  = grupo['curva_raio_min'].shift(1).fillna(raio_global)

        partes.append(grupo)

    return pd.concat(partes, ignore_index=True)


# Orquestração

def extrair_features(
    data: pd.DataFrame,
    precurva_distancia: float | None = None,
    precurva_desacel_confort: float = 2.5,
    precurva_distancia_min: float = 50.0,
    precurva_distancia_max: float = 400.0,
    lead_gap: float = 0.0,
    vars_sensor: list | None = None,
) -> pd.DataFrame:
    """Monta, para cada curva, as features da aproximação e os alvos medidos dentro dela.

    A janela de aproximação tem tamanho fixo (precurva_distancia) ou proporcional à
    distância de frenagem confortável, e termina lead_gap metros antes da entrada,
    que é o que torna a predição antecipada.

    O que sai daqui são todas as colunas possíveis. Quem escolhe as que o modelo vê
    é a whitelist do features.yaml.
    """
    df = data.copy()
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
            df, id_route_atual, dist_entrada, lead_gap, precurva_distancia,
            precurva_desacel_confort, precurva_distancia_min, precurva_distancia_max,
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
        row.update(_features_bearing_janela(janela))
        row.update(_features_geometria_janela(janela))
        row.update(_alvos_correcao(curva))

        pts_curva = curva[curva['curva'] == True] if 'curva' in curva.columns else curva
        if pts_curva.empty:
            pts_curva = curva
        row.update(_alvos_isl(pts_curva))
        row.update(_geometria_curva(pts_curva))

        dados_janela.append(row)

    df_out = (
        pd.DataFrame(dados_janela)
        .sort_values(['id_route', 'time_inicio'])
        .reset_index(drop=True)
    )

    return _adicionar_contexto(df_out)