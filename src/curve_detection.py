import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d


def detectar_curvas(
    df: pd.DataFrame,
    sigma: float = 2,
    limiar: float = 30,
    limite_raio: float = 50,
    name_traj: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta curvas em um trajeto e calcula o raio de curvatura via curvatura de Frenet.

    Parâmetros
    ----------
    sigma       : suavização gaussiana sobre a curvatura
    limiar      : percentil de corte para classificar ponto como curva
    limite_raio : usado apenas por plot_trajeto_com_curvatura
    name_traj   : filtra por id_route se fornecido

    Retorna (df_final, df_unicos)
    """
    if name_traj is not None:
        df = df.query(f'id_route == "{name_traj}"').copy()

    df = df.copy()
    df_unicos = df.drop_duplicates(subset=['lat', 'lon'], keep='last').reset_index(drop=True)

    x, y = df_unicos['x'].values, df_unicos['y'].values
    t = np.arange(len(x))

    cs = make_interp_spline(t, np.c_[x, y], k=3)
    t_fino = np.linspace(0, len(x) - 1, len(x))
    x_interp, y_interp = cs(t_fino).T

    dxdt = np.gradient(x_interp, t_fino)
    dydt = np.gradient(y_interp, t_fino)
    ddx = np.gradient(dxdt, t_fino)
    ddy = np.gradient(dydt, t_fino)

    curvatura = np.abs(dxdt * ddy - dydt * ddx) / np.power(dxdt**2 + dydt**2, 3 / 2)
    curvatura_suave = gaussian_filter1d(curvatura, sigma=sigma)
    sinal_curvatura = np.sign(gaussian_filter1d(dxdt * ddy - dydt * ddx, sigma=2))

    pontos_curva = curvatura_suave > np.percentile(curvatura_suave, limiar)

    with np.errstate(divide='ignore', invalid='ignore'):
        raio_curvatura = np.where(curvatura_suave != 0, 1 / curvatura_suave, np.inf)

    distancia_acumulada = np.zeros(len(x_interp))
    distancia_acumulada[1:] = np.cumsum(
        np.sqrt(np.diff(x_interp) ** 2 + np.diff(y_interp) ** 2)
    )

    n_cols = ['raio_curvatura', 'curvatura', 'curva', 'sinal_curvatura', 'distancia_acumulada']
    df_unicos = df_unicos.assign(
        raio_curvatura=raio_curvatura,
        curvatura=curvatura_suave,
        curva=pontos_curva,
        sinal_curvatura=sinal_curvatura,
        distancia_acumulada=distancia_acumulada,
    )

    df_final = pd.merge(df, df_unicos[['lat', 'lon'] + n_cols], on=['lat', 'lon'], how='inner')
    pd.set_option('future.no_silent_downcasting', True)
    df_final[n_cols] = df_final[n_cols].ffill()

    return df_final, df_unicos


def detectar_curvas_todos(
    dfs: list[pd.DataFrame],
    sigma: float = 2,
    limiar: float = 30,
) -> pd.DataFrame:
    """Aplica detectar_curvas em todos os trajetos e retorna df_unicos concatenado."""
    return pd.concat(
        [detectar_curvas(df, sigma=sigma, limiar=limiar)[1] for df in dfs],
        ignore_index=True,
    )


def identificar_trechos_curvos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica trechos contínuos de curva com base na coluna 'curva'.
    Adiciona a coluna 'trecho_curvo' com ID numérico por trecho.
    """
    partes = []
    trecho_id = 1

    for traj in df['id_route'].unique():
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


def contar_curvas(dfs_curves: pd.DataFrame) -> dict[str, int]:
    """Retorna dicionário {id_route: número de curvas detectadas}."""
    contagem = {}
    for traj in dfs_curves['id_route'].unique():
        df = dfs_curves.query(f'id_route == "{traj}"')
        curva = df['curva'].tolist()
        n, em_sequencia = 0, False
        for val in curva:
            if val and not em_sequencia:
                n += 1
                em_sequencia = True
            elif not val:
                em_sequencia = False
        contagem[traj] = n
    return contagem
