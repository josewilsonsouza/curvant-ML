"""
Pré-processamento de dados OBD para limpeza de ruídos identificados na análise exploratória.

Etapas aplicadas (em ordem):
  1. Clipa spikes do acelerômetro (|a| > limite fisicamente plausível)
  2. Remove pontos com velocidade impossível (OBD bug)
  3. Thinning de pontos parados consecutivos (GPS oscila quando parado → contamina spline)
  4. Divide trajetos nos gaps temporais grandes (dados de momentos distintos concatenados)
"""

from curvant.utils.config import carregar_config
import numpy as np
import pandas as pd
import argparse
import os

_DEFAULT_INPUT  = 'data/eletro_rjdf_serra.parquet'
_DEFAULT_OUTPUT = 'data/eletro_rjdf_serra_clean.parquet'


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

def _derive_output(in_path: str) -> str:
    """Gera nome do output adicionando '_clean' antes de .parquet."""
    base, ext = os.path.splitext(in_path)
    return f"{base}_clean{ext}"

def main(args: argparse.Namespace) -> None:
    """Orquestra o pré-processamento de dados."""
    cfg = carregar_config()
    pp = cfg.get('preprocessing', {})

    # Resolve argumentos: CLI > config.yaml > defaults
    _arg = lambda v, key, default: v if v is not None else pp.get(key, default)
    accel_limite             = _arg(args.accel_limite,             'accel_limite',             5.0)
    vel_max                  = _arg(args.vel_max,                  'vel_max',                  150.0)
    vel_min_parado           = _arg(args.vel_min_parado,           'vel_min_parado',           2.0)
    max_parados_consecutivos = _arg(args.max_parados_consecutivos, 'max_parados_consecutivos', 3)
    max_gap                  = _arg(args.max_gap,                  'max_gap',                  30.0)
    min_pontos_segmento      = _arg(args.min_pontos_segmento,      'min_pontos_segmento',      10)

    in_path  = args.input  or _DEFAULT_INPUT
    out_path = args.output or _derive_output(in_path)

    print(f"Carregando {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} pontos | {df['id_route'].nunique()} trajetos")
    if 'loc_coleta' in df.columns:
        print(f"  loc_coleta: {sorted(df['loc_coleta'].unique())}")

    print("\nAplicando limpeza de ruídos...")
    print(f"  accel_limite:             ±{accel_limite} m/s²")
    print(f"  vel_max:                  {vel_max} km/h")
    print(f"  vel_min_parado:           {vel_min_parado} km/h (max {max_parados_consecutivos} pts consecutivos)")
    print(f"  max_gap:                  {max_gap}s")
    print(f"  min_pontos_segmento:      {min_pontos_segmento}")

    df_clean, stats = preprocessar(
        df,
        accel_limite=accel_limite,
        vel_max=vel_max,
        vel_min_parado=vel_min_parado,
        max_parados_consecutivos=max_parados_consecutivos,
        max_gap=max_gap,
        min_pontos_segmento=min_pontos_segmento,
    )

    print("\n=== Resultado ===")
    print(f"  Pontos originais:         {stats['n_original']:,}")
    print(f"  Removidos (vel > {vel_max}):  {stats['removidos_vel']:,}")
    print(f"  Removidos (parados exc.): {stats['removidos_parados']:,}")
    print(f"  Pontos finais:            {stats['n_final']:,}  ({stats['n_final']/stats['n_original']*100:.1f}% do original)")
    print(f"  Trajetos originais:       {stats['trajs_original']}")
    print(f"  Novos segmentos (gaps):   +{stats['novos_segmentos']}")
    print(f"  Total de trajetos:        {stats['trajs_apos_split']}")

    df_clean.to_parquet(out_path, index=False)
    print(f"\nSalvo em: {out_path}")

def criar_parser() -> argparse.ArgumentParser:
    """Cria parser de argumentos para CLI."""
    parser = argparse.ArgumentParser(description='CurvantML — pré-processamento de dados OBD')
    parser.add_argument('--input',  type=str, default=None, help=f'Arquivo parquet de entrada (padrão: {_DEFAULT_INPUT})')
    parser.add_argument('--output', type=str, default=None, help='Arquivo parquet de saída (padrão: <input>_clean.parquet)')
    parser.add_argument('--accel-limite',             type=float, default=None)
    parser.add_argument('--vel-max',                  type=float, default=None)
    parser.add_argument('--vel-min-parado',           type=float, default=None)
    parser.add_argument('--max-parados-consecutivos', type=int,   default=None)
    parser.add_argument('--max-gap',                  type=float, default=None)
    parser.add_argument('--min-pontos-segmento',      type=int,   default=None)
    return parser