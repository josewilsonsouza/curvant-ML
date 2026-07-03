import numpy as np
import pandas as pd

from curvant.constants import G as _G, MU as _MU

_DNIT_RISCO: dict[str, int] = {
    'suave': 0, 'aberta': 1, 'media': 2, 'fechada': 3, 'muito_fechada': 4,
}
_RISCO_MIN_DIRECAO = 1  # aberta (R ≤ 500 m) ou mais fechada; era 2 (media, R ≤ 200 m), mas
                        # o raio B-spline tem ruído suficiente para classificar curvas de 100–150 m
                        # como 'aberta', bloqueando o gate mesmo com accel_y elevado


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
    vel_min_kmh: float = 5.0,
) -> bool:
    # Em velocidades baixas, o espaçamento GPS (~1–3 m) é da mesma ordem que o erro de
    # posição (~3–5 m), gerando mudanças de bearing fictícias mesmo em linha reta.
    if 'vehicle_speed' in janela.columns:
        janela = janela[janela['vehicle_speed'] >= vel_min_kmh]

    lats = janela['lat'].tolist()
    lons = janela['lon'].tolist()
    ctp  = janela['ctp_accel'].tolist()
    if len(lats) < 2:
        return False

    bearings = np.array([
        calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ])

    contador     = 0
    ultimo_sinal = 0  # sinal da última mudança qualificada; 0 = nenhuma ainda

    for i in range(1, len(bearings)):
        # preserva o sinal: positivo = virou à direita, negativo = à esquerda
        mudanca = (bearings[i] - bearings[i - 1] + 180) % 360 - 180
        if abs(mudanca) > limiar_bearing and abs(ctp[i]) > limiar_accel_lateral:
            sinal = int(np.sign(mudanca))
            if sinal != ultimo_sinal:   # alternância real de direção
                contador     += 1
                ultimo_sinal  = sinal
                if contador >= min_mudancas:
                    return True
            # mesma direção que a anterior: curva contínua, não zigue-zague

    return False


def caracterizar_janela(
    janela: pd.DataFrame,
    kamm_alpha: float = 0.7,
    limiar_accel_lateral: float = 2.0,
    zz_limiar_bearing: float = 15.0,
    zz_limiar_accel: float = 0.3,
    zz_min_mudancas: int = 3,
) -> dict:
    """Aplica os três critérios de risco a uma janela de tempo. Retorna dict de bools.

    Critério 1 — Círculo de Kamm:
        max_t sqrt(accel_x² + accel_y²) > alpha * mu * g
        Detecta qualquer instante em que o vetor de aceleração total ultrapassa
        uma fração alpha do limite de aderência disponível.
    """
    kamm_limite   = kamm_alpha * _MU * _G
    accel_total   = np.sqrt(janela['accel_x'].values**2 + janela['accel_y'].values**2)
    manobra_accel = bool(accel_total.max() > kamm_limite)

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
    kamm_alpha: float = 0.7,
    limiar_accel_lateral: float = 2.0,
    **zz_kwargs,
) -> pd.DataFrame:
    """
    Caracteriza risco por segmento de curva (trecho contíguo curva=True).

    Para cada segmento, avalia os três critérios sobre os pontos do segmento.
    Os rótulos são atribuídos apenas aos pontos do segmento; pontos fora de
    curvas recebem False/Segura.
    """
    _cols = ['manobra_accel', 'manobra_lateral', 'manobra_ziguezague', 'manobra_combinado']

    df_out = df.sort_values('time_sec').copy()
    for col in _cols:
        df_out[col] = False
    df_out['conducao']   = 'Segura'
    df_out['risco_dnit'] = 0
    df_out['id_janela']  = 0

    df_out['_bloco'] = (df_out['curva'] != df_out['curva'].shift()).cumsum()

    id_janela = 1
    for _, bloco in df_out.groupby('_bloco', sort=False):
        if not bool(bloco['curva'].iloc[0]):
            continue

        if len(bloco) < 2:
            continue

        r    = caracterizar_janela(bloco, kamm_alpha, limiar_accel_lateral, **zz_kwargs)
        comb = r['manobra_accel'] or r['manobra_lateral'] or r['manobra_ziguezague']

        idx = bloco.index
        df_out.loc[idx, 'manobra_accel']      = r['manobra_accel']
        df_out.loc[idx, 'manobra_lateral']    = r['manobra_lateral']
        df_out.loc[idx, 'manobra_ziguezague'] = r['manobra_ziguezague']
        df_out.loc[idx, 'manobra_combinado']  = comb
        df_out.loc[idx, 'conducao']           = 'Perigosa' if comb else 'Segura'
        df_out.loc[idx, 'risco_dnit']         = r['risco_dnit']
        df_out.loc[idx, 'id_janela']          = id_janela
        id_janela += 1

    return df_out.drop(columns=['_bloco'])


def caracterizar_todos_trajetos(
    dfs_curves: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Aplica caracterizar_conducao em cada trajeto e concatena."""
    partes = [
        caracterizar_conducao(grupo, **kwargs)
        for _, grupo in dfs_curves.groupby('id_route', sort=False)
    ]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
