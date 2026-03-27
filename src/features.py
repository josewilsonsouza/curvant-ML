import numpy as np
import pandas as pd

from src.isl import classificar_isl

_G = 9.81
_MU_PADRAO = 0.6


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
    Extrai features da janela de tempo imediatamente anterior a cada trecho de curva.

    Target: 'manobra' (1 = Perigosa, 0 = Segura) — maioria dos pontos da curva é Perigosa.

    Features extraídas
    ------------------
    - Estatísticas clássicas (mean/std/median/max/min) para cada variável sensor
    - Tendência (slope via regressão linear, cv = std/mean) de cada variável sensor
    - Contexto do trajeto: posição ordinal e histórico de curvas perigosas anteriores

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

        # Tempo normalizado para a regressão linear (começa em 0)
        t = janela['time_sec'].values - janela['time_sec'].values[0]

        for var in vars_sensor:
            vals = janela[var].values
            row[f'{var}_mean']   = float(np.mean(vals))
            row[f'{var}_std']    = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{var}_median'] = float(np.median(vals))
            row[f'{var}_max']    = float(np.max(vals))
            row[f'{var}_min']    = float(np.min(vals))
            # Tendência: coeficiente angular da regressão linear sobre o tempo
            # Positivo = crescente (acelerando/aumentando RPM); negativo = decrescente (freando)
            row[f'{var}_slope'] = float(np.polyfit(t, vals, 1)[0]) if len(t) >= 2 else 0.0
            # Dispersão relativa: quão errático estava o comportamento
            row[f'{var}_cv'] = float(row[f'{var}_std'] / row[f'{var}_mean']) if row[f'{var}_mean'] != 0 else 0.0

        row['distance_car_curve'] = (
            janela['distancia_acumulada'].max() - janela['distancia_acumulada'].min()
        )
        row['n_perigo_acc_anormal_janela'] = int(janela['aceleracao_anormal'].sum())
        row['n_perigo_dir_perigosa_janela'] = int(janela['direcao_perigosa'].sum())
        row['n_perigo_zigue_zague_janela'] = int(janela['zigue_zague'].sum())

        # Threshold mean >= 0.5: a maioria dos pontos da curva deve ser Perigosa.
        # "sum >= 2 pontos" era trivialmente atingido com amostragem 1 Hz (1 janela = ~10 pts).
        row['manobra']              = 1 if curva['conducao'].mean() >= 0.5 else 0
        row['manobra_accel_perigo'] = 1 if curva['aceleracao_anormal'].mean() >= 0.5 else 0
        row['manobra_dir_perigosa'] = 1 if curva['direcao_perigosa'].mean() >= 0.5 else 0
        row['manobra_zigue_zague']  = 1 if curva['zigue_zague'].mean() >= 0.5 else 0

        # ISL — Índice de Segurança Lateral
        # Calculado apenas nos pontos dentro da curva onde ctp_accel está disponível.
        # isl_alto é o target binário para o modelo preditivo (1 = alto risco lateral).
        if 'ctp_accel' in curva.columns and 'curva' in curva.columns:
            pts_curva = curva[curva['curva'] == True]
        else:
            pts_curva = curva
        if not pts_curva.empty and 'ctp_accel' in pts_curva.columns:
            isl_vals = pts_curva['ctp_accel'].abs() / (_G * _MU_PADRAO)
            isl_max = float(isl_vals.max())
            row['isl_mean']  = float(isl_vals.mean())
            row['isl_max']   = isl_max
            row['isl_class'] = classificar_isl(isl_max)
            row['isl_alto']  = 1 if isl_max >= 0.8 else 0
        else:
            row['isl_mean']  = np.nan
            row['isl_max']   = np.nan
            row['isl_class'] = np.nan
            row['isl_alto']  = np.nan

        dados_janela.append(row)

    df_out = (
        pd.DataFrame(dados_janela)
        .sort_values(['id_route', 'time_inicio'])
        .reset_index(drop=True)
    )

    # Contexto do trajeto: posição ordinal e histórico de perigo acumulado no mesmo trajeto.
    # Válido em tempo real — ao chegar na N-ésima curva o motorista já vivenciou as N-1 anteriores.
    partes = []
    for _, grupo in df_out.groupby('id_route', sort=False):
        grupo = grupo.copy()
        grupo['n_curvas_antes']       = np.arange(len(grupo))
        grupo['n_perigosas_antes']    = grupo['manobra'].shift(1).fillna(0).cumsum().astype(int)
        grupo['prop_perigosas_antes'] = (
            grupo['n_perigosas_antes'] / grupo['n_curvas_antes'].replace(0, np.nan)
        ).fillna(0.0).round(3)
        partes.append(grupo)

    return pd.concat(partes, ignore_index=True)
