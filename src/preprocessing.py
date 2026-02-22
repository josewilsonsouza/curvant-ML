import math

import numpy as np
import pandas as pd


def latlon_to_cartesian(latitudes, longitudes) -> list[tuple[float, float]]:
    """Converte coordenadas lat/lon para coordenadas cartesianas em metros."""
    latitudes_rad = [math.radians(lat) for lat in latitudes]
    longitudes_rad = [math.radians(lon) for lon in longitudes]
    lat0, lon0 = latitudes_rad[0], longitudes_rad[0]
    R = 6_371_000  # raio da Terra em metros

    coords = []
    for lat, lon in zip(latitudes_rad, longitudes_rad):
        if lat != lat0 or lon != lon0:
            x = R * (lon - lon0) * math.cos((lat + lat0) / 2)
            y = R * (lat - lat0)
        else:
            x, y = 0.0, 0.0
        coords.append((x, y))
    return coords


def preprocess_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa um único DataFrame de trajeto:
      - Calcula time_sec: usa coluna 'time' para rjdf/serra; converte 'timestamp' para eletronuclear
      - Interpolação temporal de vehicle_speed, lat, lon para dados eletronuclear
      - Calcula dt (delta de tempo), zerando quando o veículo estava parado no passo anterior
      - Filtra baixíssimas velocidades + acelerações nulas
      - Remove duplicatas de coordenadas/velocidade/combustível
      - Adiciona coordenadas cartesianas x, y
      - Remove colunas inteiramente zeradas ou nulas
    """
    df = df.copy()
    df.dropna(subset=['lat', 'lon'], inplace = True) # deletando valores invalidos de posição

    loc_rjdf_serra = (
        'loc_coleta' in df.columns
        and len({'rjdf', 'serra'} & set(df['loc_coleta'].unique())) > 0
    )

    if loc_rjdf_serra:
        df['time_sec'] = df['time']
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        df = df.set_index('timestamp').sort_index()
        df[['vehicle_speed', 'lat', 'lon']] = (
            df[['vehicle_speed', 'lat', 'lon']].interpolate(method='time')
        )

    df['dt'] = df['time_sec'].diff().fillna(0)
    df['prev_speed'] = df['vehicle_speed'].shift(1).fillna(0)
    df.loc[(df['prev_speed'] == 0) & (df['fuel_rate'] == 0), 'dt'] = 0
    df.drop(columns='prev_speed', inplace=True)

    df = df[~(
        (df['vehicle_speed'] < 1)
        & (df['accel_x'].abs() < 0.005)
        & (df['accel_y'].abs() < 0.005)
    )]
    df = df.drop_duplicates(subset=['lat', 'lon', 'vehicle_speed', 'fuel_rate'], keep='last')
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        return df  # trajeto sem dados úteis após filtragem

    coords = latlon_to_cartesian(df['lat'], df['lon'])
    x, y = zip(*coords)
    df = df.assign(x=x, y=y)

    # protege x e y para não serem removidos pela limpeza de colunas zeradas
    cols_to_drop = [
        col for col in df.columns
        if col not in ('x', 'y')
        and (np.all(df[col] == 0) or df[col].notnull().sum() == 0)
    ]
    return df.drop(columns=cols_to_drop)


def preprocess_all(
    dfs: list[pd.DataFrame],
    excluded_routes: list[str] | None = None,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """
    Aplica preprocess_trajectory em todos os DataFrames.

    Parâmetros
    ----------
    excluded_routes : id_routes a descartar antes de processar (ex: trajetos com dados ruins)
    """
    if excluded_routes:
        dfs = [df for df in dfs if df['id_route'].iloc[0] not in excluded_routes]

    result = []
    for i, df in enumerate(dfs, start=1):
        route = df['id_route'].iloc[0]
        rows_prev, cols_prev = df.shape
        try:
            df_proc = preprocess_trajectory(df)
        except Exception as e:
            print(f'[ERRO] Trajeto {i} ({route}): {type(e).__name__}: {e} — ignorado')
            continue

        if df_proc.empty:
            print(f'[AVISO] Trajeto {i} ({route}): vazio após filtragem — ignorado')
            continue

        rows_now, cols_now = df_proc.shape
        if verbose:
            print(
                f'(Trajeto {i}) Linhas removidas: {rows_prev - rows_now}, '
                f'tinha {rows_prev}, agora tem {rows_now}. '
                f'Colunas antes: {cols_prev}, depois: {cols_now}  [{route}]'
            )
        result.append(df_proc)
    return result
