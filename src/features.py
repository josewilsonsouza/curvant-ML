import pandas as pd

def calcular_estatisticas_por_trajeto(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatísticas médias de variáveis-chave agrupadas por trajeto."""
    vars_interesse = {
        'vehicle_speed': 'v',
        'theta_direcao': 'theta',
        'accel_x': 'a_x',
        'accel_y': 'a_y',
        'accel_z': 'a_z',
    }
    rows = []
    for id_route_atual, trajeto in df.groupby('id_route'):
        row = {
            'id_trajeto': id_route_atual,
            'numb_curve': len(trajeto['trecho_curvo'].unique()) - 1,
        }
        for var, alias in vars_interesse.items():
            if var in trajeto.columns:
                row[alias] = round(trajeto[var].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def extrair_features(data: pd.DataFrame, janela_tempo: int = 10) -> pd.DataFrame:
    """
    Extrai features estatísticas da janela de tempo imediatamente anterior a cada trecho de curva.

    Target: 'manobra' (1 = Perigosa, 0 = Segura) — definido como ≥2 pontos perigosos na curva.

    Parâmetros
    ----------
    data         : DataFrame com colunas id_route, trecho_curvo, conducao, aceleracao_anormal,
                   direcao_perigosa, zigue_zague, distancia_acumulada
    janela_tempo : segundos antes do início da curva a considerar
    """
    dados_janela = []
    df = data.copy()
    df['conducao'] = df['conducao'].map({'Perigosa': 1, 'Segura': 0})
    df = df.sort_values(by=['id_route', 'time_sec'])

    vars_sensor = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y']

    for (id_route_atual, trecho_curvo), curva in df.groupby(['id_route', 'trecho_curvo']):
        if len(curva) <= 2:
            continue

        inicio_curva = curva['time_sec'].min()
        janela = df[
            (df['id_route'] == id_route_atual)
            & (df['time_sec'] < inicio_curva)
            & (df['time_sec'] >= inicio_curva - janela_tempo)
        ].copy()

        if janela.empty:
            print(f'Janela vazia para {id_route_atual} e trecho {trecho_curvo}')
            continue

        row: dict = {
            'time_inicio': janela['time_sec'].min(),
            'time_fim': janela['time_sec'].max(),
            'id_route': id_route_atual,
            'id_trecho_curvo': trecho_curvo,
        }

        for var in vars_sensor:
            row[f'{var}_mean'] = janela[var].mean()
            row[f'{var}_std'] = janela[var].std()
            row[f'{var}_median'] = janela[var].median()
            row[f'{var}_max'] = janela[var].max()
            row[f'{var}_min'] = janela[var].min()

        row['distance_car_curve'] = (
            janela['distancia_acumulada'].max() - janela['distancia_acumulada'].min()
        )
        row['n_perigo_acc_anormal_janela'] = int(janela['aceleracao_anormal'].sum())
        row['n_perigo_dir_perigosa_janela'] = int(janela['direcao_perigosa'].sum())
        row['n_perigo_zigue_zague_janela'] = int(janela['zigue_zague'].sum())

        row['manobra'] = 1 if curva['conducao'].sum() >= 2 else 0
        row['manobra_accel_perigo'] = 1 if curva['aceleracao_anormal'].sum() >= 2 else 0
        row['manobra_dir_perigosa'] = 1 if curva['direcao_perigosa'].sum() >= 2 else 0
        row['manobra_zigue_zague'] = 1 if curva['zigue_zague'].sum() >= 2 else 0

        dados_janela.append(row)

    return pd.DataFrame(dados_janela).reset_index(drop=True)
