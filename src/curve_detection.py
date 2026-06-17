import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, splprep, splev
from scipy.ndimage import gaussian_filter1d, binary_opening


# ── Derivadas da trajetória via spline ────────────────────────────────────────

def _derivadas_spline(
    x: np.ndarray,
    y: np.ndarray,
    sigma_gps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Primeira e segunda derivadas de (x(u), y(u)) por spline cúbica.

    sigma_gps == 0 → spline INTERPOLADORA (passa por todos os pontos; comportamento
                     legado). Sofre overshoot entre pontos espaçados → raios espúrios.
    sigma_gps  > 0 → spline SUAVIZADORA (splprep, s = n · sigma_gps²), parametrizada
                     por comprimento de arco. Não força a interpolação, eliminando o
                     overshoot. sigma_gps ≈ ruído de posição do GPS (m), ~2 m aqui.

    A curvatura de Frenet é invariante à reparametrização, então o arco é válido.
    Retorna (dx, dy, ddx, ddy). Em falha do splprep, cai no interpolador.
    """
    n = len(x)
    if sigma_gps and sigma_gps > 0 and n >= 4:
        arc = np.r_[0.0, np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))]
        if arc[-1] > 0:
            u = arc / arc[-1]
            # splprep exige u estritamente crescente; pontos coincidentes (x,y iguais,
            # passo de arco zero) violam isso. Ajusta a spline só nos pontos com arco
            # crescente e a avalia em TODOS os pontos (coincidentes herdam a derivada).
            keep = np.r_[True, np.diff(arc) > 0]
            if int(keep.sum()) >= 4:
                s = int(keep.sum()) * float(sigma_gps) ** 2
                try:
                    tck, _ = splprep([x[keep], y[keep]], u=u[keep], s=s, k=3)
                    d1 = np.array(splev(u, tck, der=1))
                    d2 = np.array(splev(u, tck, der=2))
                    return d1[0], d1[1], d2[0], d2[1]
                except Exception:
                    pass  # degenerado → usa interpolador abaixo

    t  = np.arange(n)
    cs = make_interp_spline(t, np.c_[x, y], k=3)
    d1 = cs.derivative(1)(t)
    d2 = cs.derivative(2)(t)
    return d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1]


# ── Sigma adaptativo ──────────────────────────────────────────────────────────

def _sigma_adaptativo(
    x: np.ndarray,
    y: np.ndarray,
    target_metros: float = 20.0,
    sigma_min: float = 1.0,
    sigma_max: float = 8.0,
) -> float:
    """
    Calcula o sigma de suavização gaussiana com base na densidade de pontos GPS.

    Lógica: para suavizar uma janela de target_metros, precisamos de
        sigma ≈ target_metros / espaçamento_mediano  (em número de pontos)

    GPS denso  (d ≈ 2 m/ponto)  → sigma ≈ 10  → mais suavização
    GPS esparso (d ≈ 20 m/ponto) → sigma ≈ 1   → menos suavização

    Parâmetros
    ----------
    target_metros : extensão espacial (em metros) que deve ser suavizada
    sigma_min     : piso — evita suavização insuficiente
    sigma_max     : teto — evita apagar curvas reais em GPS muito denso
    """
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    dist_validas = dist[dist > 0.1]    # descarta pontos quase coincidentes
    if len(dist_validas) == 0:
        return float(sigma_min)
    mediana = float(np.median(dist_validas))
    sigma = target_metros / mediana
    return float(np.clip(sigma, sigma_min, sigma_max))


# Classificação DNIT de curvas horizontais
# Grau de curva: D = 1145.92 / R  (graus por corda de 20 m)
# Referência: Manual de Projeto Geométrico de Rodovias Rurais, DNIT (2010)
_DNIT_LIMIARES: list[tuple[float, str]] = [
    (50,  'muito_fechada'),  # D > 22.9° — risco muito alto
    (100, 'fechada'),        # 11.5° < D ≤ 22.9° — risco alto
    (200, 'media'),          # 5.7°  < D ≤ 11.5° — risco moderado
    (500, 'aberta'),         # 2.3°  < D ≤  5.7° — risco baixo
]


def classificar_curva_dnit(raio: float) -> str:
    """
    Classifica a intensidade da curva segundo o grau de curvatura do DNIT.

    D = 1145.92 / R  (graus por corda de 20 m)

    Retorna uma das classes: 'muito_fechada', 'fechada', 'media', 'aberta', 'suave'.
    """
    for limite, classe in _DNIT_LIMIARES:
        if raio <= limite:
            return classe
    return 'suave'  # R > 500 m (D < 2.3°) ou reta


def detectar_curvas(df,
                    sigma=2,
                    limite_raio=100,
                    min_pontos=3,
                    sigma_gps=0.0,
                    name_traj=None):
    '''
    Detecta curvas em um trajeto usando curvatura de Frenet via B-spline cúbica.

    Parâmetros
    ----------
    sigma       : suavização gaussiana sobre a curvatura.
                  Pode ser um número fixo (ex.: 2) ou a string 'auto'.
                  Com 'auto', o sigma é calculado por _sigma_adaptativo():
                  sigma ≈ 20m / espaçamento_mediano_entre_pontos, clipado em [1, 8].
                  GPS denso (2 m/pt) → sigma ≈ 10 (mais suave);
                  GPS esparso (20 m/pt) → sigma ≈ 1 (menos suave).
    limite_raio : raio máximo (metros) para classificar um ponto como curva
    min_pontos  : mínimo de pontos consecutivos para considerar curva válida
                  (elimina picos isolados de ruído GPS)
    '''
    if name_traj is not None:
        df = df.query(f'id_route == "{name_traj}"').copy()

    df = df.copy()

    if 'x' not in df.columns or 'y' not in df.columns:
        raise KeyError(
            f"Trajeto {df['id_route'].iloc[0]!r} não possui as colunas 'x'/'y' "
            "(projeção cartesiana de lat/lon). Gere-as no pré-processamento "
            "antes da detecção de curvas."
        )

    df_unicos = df.drop_duplicates(subset=['lat', 'lon', 'vehicle_speed'], keep='last').reset_index(drop=True)

    if len(df_unicos) < 4:
        raise ValueError(
            f"Trajeto {df['id_route'].iloc[0]!r} tem apenas {len(df_unicos)} ponto(s) "
            "únicos após drop_duplicates — mínimo de 4 para spline cúbica."
        )

    x = df_unicos['x'].values
    y = df_unicos['y'].values

    if sigma == 'auto':
        sigma = _sigma_adaptativo(x, y)

    # Derivadas da trajetória via spline (interpoladora se sigma_gps=0,
    # suavizadora por comprimento de arco se sigma_gps>0 — evita overshoot)
    dxdt, dydt, ddx, ddy = _derivadas_spline(x, y, sigma_gps=sigma_gps)

    # Curvatura de Frenet: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    denom = np.power(dxdt**2 + dydt**2, 3 / 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        curvatura = np.where(denom > 0, np.abs(dxdt * ddy - dydt * ddx) / denom, 0.0)

    curvatura_suave = gaussian_filter1d(curvatura, sigma=sigma)
    sinal_curvatura = np.sign(gaussian_filter1d(dxdt * ddy - dydt * ddx, sigma=sigma))

    with np.errstate(divide='ignore', invalid='ignore'):
        raio_curvatura = np.where(curvatura_suave != 0, 1 / curvatura_suave, np.inf)

    # Critério absoluto: raio < limite_raio
    # binary_opening remove segmentos com menos de min_pontos consecutivos (ruído GPS)
    pontos_curva = raio_curvatura < limite_raio
    pontos_curva = binary_opening(pontos_curva, iterations=min_pontos)

    distancia_acumulada = np.zeros(len(x))
    distancia_acumulada[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    # Aplica DNIT apenas nos pontos que a detecção confirmou como curva real
    # (binary_opening já filtrou picos isolados de GPS).
    # Pontos fora de curva recebem 'suave' independente do raio calculado,
    # que é não-confiável em trechos retos por ruído GPS na B-spline.
    classe_dnit_raw = np.vectorize(classificar_curva_dnit)(raio_curvatura)
    classe_dnit = np.where(pontos_curva, classe_dnit_raw, 'suave')

    df_unicos = df_unicos.assign(
        raio_curvatura=raio_curvatura,
        curvatura=curvatura_suave,
        curva=pontos_curva,
        sinal_curvatura=sinal_curvatura,
        distancia_acumulada=distancia_acumulada,
        classe_dnit=classe_dnit,
    )

    return df_unicos

def identificar_trechos_curvos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica trechos contínuos de curva com base na coluna 'curva'.
    Adiciona a coluna 'trecho_curvo' com ID numérico por trecho, reiniciando em 1
    para cada rota (trecho_curvo=0 significa ponto fora de curva).
    """
    partes = []

    for traj in df['id_route'].unique():
        trecho_id = 1  # reinicia por rota — evita IDs globais que crescem indefinidamente
        df_traj = df.query(f'id_route == "{traj}"').copy().reset_index(drop=True)
        df_traj['trecho_curvo'] = 0
        em_trecho = False

        for i in range(len(df_traj)):
            if df_traj.at[i, 'curva']:
                if not em_trecho:
                    em_trecho = True
                df_traj.at[i, 'trecho_curvo'] = trecho_id
            else:
                if em_trecho:
                    trecho_id += 1
                    em_trecho = False

        partes.append(df_traj)

    return pd.concat(partes, ignore_index=True)