import numpy as np
import pandas as pd

from curvant.constants import (
    G as _G, MU as _MU, ISL_BAIXO as _ISL_BAIXO_MAX, ISL_ALTO as _ISL_MEDIO_MAX,
)

def simular_risco_monte_carlo(
    velocidades: np.ndarray,
    tempos: np.ndarray,
    raio_curva: float,
    n_sim: int = 1000,
    k_ultimos: int = 10,
    sigma_min: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Simula N cenários de velocidade na entrada da curva e retorna
    a distribuição de probabilidade sobre as classes de ISL.

    Parâmetros
    ----------
    velocidades : ndarray (km/h) - perfil temporal da janela pré-curva
    tempos      : ndarray (s)    - timestamps correspondentes
    raio_curva  : float (m)      - raio mínimo da curva (f4_raio_min)
    n_sim       : int            - número de simulações
    k_ultimos   : int            - pontos finais usados para tendência
    sigma_min   : float (km/h)   - desvio mínimo (incerteza de sensor)
    rng         : Generator      - gerador numpy para reprodutibilidade

    Retorna
    -------
    dict{'baixo', 'medio', 'alto'} com probabilidades somando 1.0,
    ou {'baixo': 0.0, 'medio': 0.0, 'alto': 0.0} se raio inválido.
    """
    if rng is None:
        rng = np.random.default_rng()

    zero = {'baixo': 0.0, 'medio': 0.0, 'alto': 0.0}

    if raio_curva is None or np.isnan(raio_curva) or raio_curva <= 0:
        return zero
    if len(velocidades) < 2:
        return zero

    k   = min(k_ultimos, len(velocidades))
    v_k = velocidades[-k:].astype(float)
    t_k = tempos[-k:].astype(float)
    t_n = t_k - t_k[0]   # normaliza origem

    # Tendência linear nos últimos k pontos
    if k >= 2:
        a, b    = np.polyfit(t_n, v_k, 1)
        v_trend = a * t_n + b
        v_entry = float(np.clip(a * t_n[-1] + b, 0.0, 200.0))
        residuos = v_k - v_trend
        sigma    = float(np.std(residuos))
    else:
        v_entry = float(v_k[-1])
        sigma   = 0.0

    sigma = max(sigma, sigma_min)

    # Simula N velocidades de entrada
    v_sim = rng.normal(loc=v_entry, scale=sigma, size=n_sim)
    v_sim = np.clip(v_sim, 0.0, 200.0)

    # ISL por cenário: (v m/s)² / (R × g × μ)
    isl = (v_sim / 3.6) ** 2 / (raio_curva * _G * _MU)

    p_baixo = float((isl <  _ISL_BAIXO_MAX).sum()) / n_sim
    p_medio = float(((isl >= _ISL_BAIXO_MAX) & (isl < _ISL_MEDIO_MAX)).sum()) / n_sim
    p_alto  = float((isl >= _ISL_MEDIO_MAX).sum()) / n_sim

    return {
        'baixo': round(p_baixo, 4),
        'medio': round(p_medio, 4),
        'alto':  round(p_alto,  4),
    }


def aplicar_mc_features(
    features_df: pd.DataFrame,
    df_analysis: pd.DataFrame,
    n_sim: int = 1000,
    k_ultimos: int = 10,
    sigma_min: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Adiciona colunas mc_p_baixo, mc_p_medio, mc_p_alto a features_df
    reutilizando a janela pré-curva de df_analysis (time_inicio / time_fim).

    Se o raio da curva não estiver disponível, os três campos ficam 0.0.
    """
    rng = np.random.default_rng(random_state)

    # Indexa df_analysis por rota para lookup rápido
    grupos = {
        rota: sub[['vehicle_speed', 'time_sec']].reset_index(drop=True)
        for rota, sub in df_analysis.groupby('id_route')
    }

    mc_baixo, mc_medio, mc_alto = [], [], []

    for _, row in features_df.iterrows():
        t0  = row.get('time_inicio')
        t1  = row.get('time_fim')
        r   = row.get('f4_raio_min', 0.0)
        sub = grupos.get(str(row['id_route']))

        if sub is None or pd.isna(t0) or pd.isna(t1):
            mc_baixo.append(0.0); mc_medio.append(0.0); mc_alto.append(0.0)
            continue

        mask   = (sub['time_sec'] >= t0) & (sub['time_sec'] <= t1)
        janela = sub.loc[mask]

        if len(janela) < 2 or pd.isna(r) or float(r) <= 0:
            mc_baixo.append(0.0); mc_medio.append(0.0); mc_alto.append(0.0)
            continue

        res = simular_risco_monte_carlo(
            velocidades=janela['vehicle_speed'].values,
            tempos=janela['time_sec'].values,
            raio_curva=float(r),
            n_sim=n_sim,
            k_ultimos=k_ultimos,
            sigma_min=sigma_min,
            rng=rng,
        )
        mc_baixo.append(res['baixo'])
        mc_medio.append(res['medio'])
        mc_alto.append(res['alto'])

    out = features_df.copy()
    out['mc_p_baixo'] = mc_baixo
    out['mc_p_medio'] = mc_medio
    out['mc_p_alto']  = mc_alto
    return out
