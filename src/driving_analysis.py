import numpy as np
import pandas as pd


def calcular_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula o ângulo de direção (bearing) entre dois pontos geográficos, em graus [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def detectar_aceleracao_anormal(var_velocidade: float, limite: float = 15.0) -> bool:
    """True se a variação acumulada de velocidade na janela superar o limite (km/h)."""
    return var_velocidade > limite


def detectar_direcao_perigosa(
    velocidade: float,
    theta: float,
    velocidade_min: float = 30.0,
    angulo_max: float = 0.7,
) -> bool:
    """True se o veículo estiver acima de velocidade_min e com variação angular abaixo de angulo_max rad."""
    return velocidade > velocidade_min and np.abs(theta) < angulo_max


def detectar_zigue_zague(
    janela: pd.DataFrame,
    limiar_bearing: float = 15.0,
    limiar_accel_lateral: float = 0.3,
    min_mudancas: int = 3,
) -> bool:
    """
    Detecta padrão de zigue-zague via mudanças bruscas de bearing + aceleração centrípeta.
    """
    lats = janela['lat'].tolist()
    lons = janela['lon'].tolist()
    ctp_accel = janela['ctp_accel'].tolist()

    if len(lats) < 2:
        return False

    bearing_angles = np.array([
        calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ])

    contador = 0
    for i in range(1, len(bearing_angles)):
        mudanca = abs(bearing_angles[i] - bearing_angles[i - 1])
        if mudanca > limiar_bearing and abs(ctp_accel[i]) > limiar_accel_lateral:
            contador += 1
            if contador >= min_mudancas:
                return True
    return False


def detectar_conducao_perigosa(
    df: pd.DataFrame,
    janela_tempo: int = 10,
    var_velocidade_max: float = 15.0,
    velocidade_max_direcao: float = 30.0,
    angulo_max_direcao: float = 0.7,
) -> pd.DataFrame:
    """
    Classifica janelas de tempo como 'Perigosa' ou 'Segura' conforme Li et al. (2016).

    Critérios:
      1. Aceleração/desaceleração anormal  (var. velocidade > var_velocidade_max km/h)
      2. Direção perigosa                  (curva > angulo_max_direcao rad acima de velocidade_max_direcao km/h)
      3. Zigue-zague                       (≥3 mudanças bruscas de bearing + aceleração lateral)
    """
    resultados = []
    inicio = df['time_sec'].min()
    fim = df['time_sec'].max()
    t = inicio
    id_janela = 1

    while t + janela_tempo <= fim:
        janela = df[(df['time_sec'] >= t) & (df['time_sec'] < t + janela_tempo)]

        if len(janela) < 2:
            t += janela_tempo
            continue

        # 1. Aceleração anormal
        var_vel = janela['vehicle_speed'].diff().abs().sum()
        aceleracao_anormal = detectar_aceleracao_anormal(var_vel, var_velocidade_max)

        # 2. Direção perigosa
        lats = janela['lat'].tolist()
        lons = janela['lon'].tolist()
        thetas = [
            calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
            for i in range(1, len(janela))
        ]
        theta_direcao = float(np.abs(np.sum(np.diff(thetas)))) if thetas else 0.0
        direcao_perigosa = detectar_direcao_perigosa(
            janela['vehicle_speed'].mean(), theta_direcao,
            velocidade_max_direcao, angulo_max_direcao,
        )

        # 3. Zigue-zague
        zigue_zague = detectar_zigue_zague(janela)

        conducao = 'Perigosa' if (aceleracao_anormal or direcao_perigosa or zigue_zague) else 'Segura'

        janela = janela.copy()
        janela['conducao'] = conducao
        janela['aceleracao_anormal'] = aceleracao_anormal
        janela['direcao_perigosa'] = direcao_perigosa
        janela['zigue_zague'] = zigue_zague
        janela['theta_direcao'] = theta_direcao
        janela['id_janela'] = id_janela
        id_janela += 1

        resultados.append(janela)
        t += janela_tempo

    return pd.concat(resultados, ignore_index=True)


def analisar_todos_trajetos(
    dfs_curves: pd.DataFrame,
    janela_tempo: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """Aplica detectar_conducao_perigosa em cada trajeto e concatena os resultados."""
    return pd.concat([
        detectar_conducao_perigosa(
            dfs_curves.query(f'id_route == "{traj}"'),
            janela_tempo=janela_tempo,
            **kwargs,
        )
        for traj in dfs_curves['id_route'].unique()
    ], ignore_index=True)
