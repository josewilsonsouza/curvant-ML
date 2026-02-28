"""
Pré-processamento de dados OBD para limpeza de ruídos identificados na análise exploratória.

Etapas aplicadas (em ordem):
  1. Clipa spikes do acelerômetro (|a| > limite fisicamente plausível)
  2. Remove pontos com velocidade impossível (OBD bug)
  3. Thinning de pontos parados consecutivos (GPS oscila quando parado → contamina spline)
  4. Divide trajetos nos gaps temporais grandes (dados de momentos distintos concatenados)
"""

import numpy as np
import pandas as pd


def clipar_acelerometro(df: pd.DataFrame, limite: float = 5.0) -> pd.DataFrame:
    """
    Clipa spikes do acelerômetro para ±limite m/s².
    Valores > 5 m/s² (~0.5g) são fisicamente improváveis em condução normal.
    """
    df = df.copy()
    for col in ['accel_x', 'accel_y', 'accel_z']:
        if col in df.columns:
            df[col] = df[col].clip(-limite, limite)
    return df


def filtrar_velocidade(df: pd.DataFrame, vel_max: float = 150.0) -> pd.DataFrame:
    """Remove pontos com velocidade acima de vel_max km/h (leituras OBD corrompidas)."""
    return df[df['vehicle_speed'] <= vel_max].copy()


def refinar_parados(df: pd.DataFrame, vel_min: float = 2.0, max_manter: int = 3) -> pd.DataFrame:
    """
    Em cada sequência consecutiva de pontos parados (vel < vel_min), mantém apenas
    os primeiros max_manter pontos e descarta o restante.

    Justificativa: quando o carro está parado, o GPS oscila em torno de um ponto fixo.
    Isso cria curvaturas artificiais na B-spline que inflam o raio_curvatura calculado.
    Manter alguns pontos preserva o contexto temporal; remover o excesso elimina o ruído.
    """
    partes = []
    for _, grp in df.groupby('id_route', sort=False):
        grp = grp.reset_index(drop=True)
        manter = []
        contador_parado = 0

        for i, vel in enumerate(grp['vehicle_speed']):
            if vel < vel_min:
                contador_parado += 1
                if contador_parado <= max_manter:
                    manter.append(i)
            else:
                contador_parado = 0
                manter.append(i)

        partes.append(grp.iloc[manter])

    return pd.concat(partes, ignore_index=True)


def splittar_por_gaps(
    df: pd.DataFrame,
    max_gap: float = 30.0,
    min_pontos: int = 10,
) -> pd.DataFrame:
    """
    Divide trajetos em sub-trajetos onde o intervalo temporal entre pontos consecutivos
    excede max_gap segundos. Cria IDs no formato '<id_route>_p<N>'.

    Sub-trajetos com menos de min_pontos pontos são descartados (insuficientes para
    cálculo de spline cúbica e janelas de análise).

    Reseta time_sec para começar em 0 em cada sub-trajeto.
    """
    partes = []
    for id_route, grp in df.groupby('id_route', sort=False):
        grp = grp.sort_values('time_sec').reset_index(drop=True)
        dt = grp['time_sec'].diff().fillna(0)

        # Índices onde começa um novo segmento (após gap)
        cortes = [0] + dt[dt > max_gap].index.tolist() + [len(grp)]

        n_segmentos = len(cortes) - 1
        parte = 1
        for i in range(n_segmentos):
            sub = grp.iloc[cortes[i]:cortes[i + 1]].copy()
            if len(sub) < min_pontos:
                continue

            # Renomeia apenas se há mais de um segmento válido
            if n_segmentos > 1:
                sub['id_route'] = f'{id_route}_p{parte}'

            sub['time_sec'] = sub['time_sec'] - sub['time_sec'].iloc[0]
            parte += 1
            partes.append(sub)

    return pd.concat(partes, ignore_index=True)


def preprocessar(
    df: pd.DataFrame,
    accel_limite: float = 5.0,
    vel_max: float = 150.0,
    vel_min_parado: float = 2.0,
    max_parados_consecutivos: int = 3,
    max_gap: float = 30.0,
    min_pontos_segmento: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica o pipeline completo de limpeza de ruído.

    Retorna o DataFrame limpo e um dict com estatísticas do processo.
    """
    stats: dict = {
        'n_original': len(df),
        'trajs_original': df['id_route'].nunique(),
    }

    # 1. Acelerômetro
    df = clipar_acelerometro(df, limite=accel_limite)

    # 2. Velocidade impossível
    n_antes = len(df)
    df = filtrar_velocidade(df, vel_max=vel_max)
    stats['removidos_vel'] = n_antes - len(df)

    # 3. Pontos parados excessivos
    n_antes = len(df)
    df = refinar_parados(df, vel_min=vel_min_parado, max_manter=max_parados_consecutivos)
    stats['removidos_parados'] = n_antes - len(df)

    # 4. Split nos gaps temporais
    n_trajs_antes = df['id_route'].nunique()
    df = splittar_por_gaps(df, max_gap=max_gap, min_pontos=min_pontos_segmento)
    stats['trajs_apos_split'] = df['id_route'].nunique()
    stats['novos_segmentos'] = stats['trajs_apos_split'] - n_trajs_antes
    stats['n_final'] = len(df)

    return df, stats
