import numpy as np
import pandas as pd

# Mapeamento classe DNIT → nível de risco inteiro (para comparações)
_DNIT_RISCO: dict[str, int] = {
    'suave':         0,  # R > 500 m — permissível em velocidade
    'aberta':        1,  # 200 m < R ≤ 500 m — risco baixo
    'media':         2,  # 100 m < R ≤ 200 m — risco moderado
    'fechada':       3,  # 50 m < R ≤ 100 m — risco alto
    'muito_fechada': 4,  # R ≤ 50 m — risco muito alto
}

# Nível mínimo de risco DNIT para aplicar o critério de direção perigosa.
# Abaixo desse nível (curvas suave/aberta), alta velocidade é permissível.
_RISCO_MIN_DIRECAO = 2  # 'media' em diante


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
    theta_graus: float,
    velocidade_min: float = 30.0,
    angulo_max_graus: float = 40.0,
) -> bool:
    """
    True se o veículo estiver acima de velocidade_min km/h e com variação líquida
    de bearing abaixo de angulo_max_graus graus.

    Ambos os ângulos estão em GRAUS (bearing é calculado em graus [0, 360)).
    Equivalência Li et al. (2016): angulo_max = 0.7 rad ≈ 40.1°.
    """
    return velocidade > velocidade_min and np.abs(theta_graus) < angulo_max_graus


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
        mudanca = abs((bearing_angles[i] - bearing_angles[i - 1] + 180) % 360 - 180)
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
    angulo_max_direcao: float = 40.0,
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

        # Classe DNIT mais restritiva da janela (menor raio = maior risco)
        if 'classe_dnit' in janela.columns:
            risco_dnit = int(janela['classe_dnit'].map(_DNIT_RISCO).max())
        else:
            risco_dnit = 3  # fallback conservador se coluna ausente

        # 2. Direção perigosa — suprimida em curvas suave/aberta (risco DNIT < 2)
        lats = janela['lat'].tolist()
        lons = janela['lon'].tolist()
        # Variação de heading robusta a ruído GPS:
        # Comparamos o bearing da primeira metade da janela com o bearing da janela toda.
        # Usar bearings ponto-a-ponto (< 1m de distância) produz ruído de ±40–90° que
        # domina o sinal real — daí a mediana de theta ≈ 5° mesmo em curvas reais.
        n = len(lats)
        if n >= 4:
            mid = n // 2
            b_inicio = calcular_bearing(lats[0], lons[0], lats[mid], lons[mid])
            b_total  = calcular_bearing(lats[0], lons[0], lats[-1], lons[-1])
            theta_direcao = float(abs((b_total - b_inicio + 180) % 360 - 180))
        elif n >= 2:
            theta_direcao = float(abs((
                calcular_bearing(lats[0], lons[0], lats[-1], lons[-1]) + 180
            ) % 360 - 180))
        else:
            theta_direcao = 0.0
        direcao_perigosa = (
            risco_dnit >= _RISCO_MIN_DIRECAO
            and detectar_direcao_perigosa(
                janela['vehicle_speed'].mean(), theta_direcao,
                velocidade_max_direcao, angulo_max_direcao,
            )
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
        janela['risco_dnit'] = risco_dnit
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
