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

    if not resultados:
        return pd.DataFrame()
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
