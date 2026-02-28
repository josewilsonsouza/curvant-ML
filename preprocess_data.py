"""
CurvantML — Script de pré-processamento de dados

Lê 'data/eletro_rjdf_serra.parquet', aplica limpeza de ruídos e salva
'data/eletro_rjdf_serra_clean.parquet'.

Uso:
    python preprocess_data.py
    python preprocess_data.py --max-gap 60 --accel-limite 4.0
"""

import argparse

import pandas as pd
import yaml

from src.preprocessing import preprocessar


def main(args: argparse.Namespace) -> None:
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    pp = cfg.get('preprocessing', {})

    accel_limite            = args.accel_limite            or pp.get('accel_limite',            5.0)
    vel_max                 = args.vel_max                 or pp.get('vel_max',                 150.0)
    vel_min_parado          = args.vel_min_parado          or pp.get('vel_min_parado',          2.0)
    max_parados_consecutivos = args.max_parados_consecutivos or pp.get('max_parados_consecutivos', 3)
    max_gap                 = args.max_gap                 or pp.get('max_gap',                 30.0)
    min_pontos_segmento     = args.min_pontos_segmento     or pp.get('min_pontos_segmento',     10)

    print("Carregando dados...")
    df = pd.read_parquet('data/eletro_rjdf_serra.parquet')
    print(f"  {len(df):,} pontos | {df['id_route'].nunique()} trajetos")

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

    out_path = 'data/eletro_rjdf_serra_clean.parquet'
    df_clean.to_parquet(out_path, index=False)
    print(f"\nSalvo em: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CurvantML — pré-processamento de dados OBD')
    parser.add_argument('--accel-limite',             type=float, default=None, help='Clip do acelerometro em m/s² (default: 5.0)')
    parser.add_argument('--vel-max',                  type=float, default=None, help='Velocidade maxima km/h (default: 150.0)')
    parser.add_argument('--vel-min-parado',           type=float, default=None, help='Vel. minima para considerar parado km/h (default: 2.0)')
    parser.add_argument('--max-parados-consecutivos', type=int,   default=None, help='Max pontos parados consecutivos mantidos (default: 3)')
    parser.add_argument('--max-gap',                  type=float, default=None, help='Gap temporal maximo em segundos (default: 30.0)')
    parser.add_argument('--min-pontos-segmento',      type=int,   default=None, help='Minimo de pontos por segmento (default: 10)')
    main(parser.parse_args())
