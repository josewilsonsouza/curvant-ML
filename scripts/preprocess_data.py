"""
CurvantML — Script de pré-processamento de dados

Uso:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --input data/eletro_rjdf_serra_rjmgba_janeiro.parquet
    python scripts/preprocess_data.py --input data/eletro_rjdf_serra_rjmgba_janeiro.parquet --output data/full_clean.parquet
    python scripts/preprocess_data.py --max-gap 60 --accel-limite 4.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from src.preprocessing import preprocessar
from utils.config import carregar_config

_DEFAULT_INPUT  = 'data/eletro_rjdf_serra.parquet'
_DEFAULT_OUTPUT = 'data/eletro_rjdf_serra_clean.parquet'


def main(args: argparse.Namespace) -> None:
    cfg = carregar_config()
    pp = cfg.get('preprocessing', {})

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


def _derive_output(in_path: str) -> str:
    """Gera nome do output adicionando '_clean' antes de .parquet."""
    base, ext = os.path.splitext(in_path)
    return f"{base}_clean{ext}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CurvantML — pré-processamento de dados OBD')
    parser.add_argument('--input',  type=str, default=None, help=f'Arquivo parquet de entrada (padrão: {_DEFAULT_INPUT})')
    parser.add_argument('--output', type=str, default=None, help='Arquivo parquet de saída (padrão: <input>_clean.parquet)')
    parser.add_argument('--accel-limite',             type=float, default=None)
    parser.add_argument('--vel-max',                  type=float, default=None)
    parser.add_argument('--vel-min-parado',           type=float, default=None)
    parser.add_argument('--max-parados-consecutivos', type=int,   default=None)
    parser.add_argument('--max-gap',                  type=float, default=None)
    parser.add_argument('--min-pontos-segmento',      type=int,   default=None)
    main(parser.parse_args())
