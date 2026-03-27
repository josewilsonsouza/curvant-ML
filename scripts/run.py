"""
CurvantML — Pipeline de Experimentos

Uso:
    python scripts/run_experiments.py                    # apenas modelos clássicos
    python scripts/run_experiments.py --plot             # gera plots
    python scripts/run_experiments.py --mlp              # + MLP sklearn (GridSearchCV)
    python scripts/run_experiments.py --keras            # + redes neurais Keras (MLP, GRU, LSTM)
    python scripts/run_experiments.py --plot --mlp --keras
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from src.pipeline import (
    etapa_curvas, etapa_analise_conducao, etapa_features,
    etapa_ml_classico, etapa_mlp_sklearn, etapa_keras,
    etapa_isl_modelo,
)
from utils.config import carregar_config


def main(args: argparse.Namespace) -> None:
    cfg = carregar_config()

    clean_path = 'data/eletro_rjdf_serra_clean.parquet'
    raw_path   = 'data/eletro_rjdf_serra.parquet'
    data_path  = clean_path if os.path.exists(clean_path) else raw_path

    print("[1/6] Carregando dados...")
    if data_path == clean_path:
        print("  Usando dados pré-processados (clean). Para regenerar: python scripts/preprocess_data.py")
    else:
        print("  AVISO: dados limpos não encontrados. Execute 'python scripts/preprocess_data.py' primeiro.")
        print(f"  Usando: {raw_path}")

    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} registros | {df['id_route'].nunique()} trajetos")

    print("\n[2/6] Detectando curvas...")
    dfs_curves = etapa_curvas(df, cfg)

    print("\n[3/6] Calculando aceleração centrípeta e absoluta...")
    dfs_curves['ctp_accel'] = (
        (dfs_curves['vehicle_speed'] / 3.6) ** 2 / dfs_curves['raio_curvatura']
    )
    dfs_curves['abs_accel'] = np.sqrt(
        dfs_curves['accel_x'] ** 2 + dfs_curves['accel_y'] ** 2
    )

    print("\n[4/6] Analisando condução perigosa...")
    df_analysis = etapa_analise_conducao(dfs_curves, cfg, args.plot)

    print("\n[5/6] Identificando trechos curvos e extraindo features...")
    features_df = etapa_features(df_analysis, cfg)

    print("\n[6/6] Modelos clássicos de ML...")
    etapa_ml_classico(features_df, cfg, args.plot)

    if args.isl:
        print("\n[Extra] ISL — Índice de Segurança Lateral...")
        etapa_isl_modelo(features_df, cfg, args.plot)

    if args.mlp:
        print("\n[Extra] MLP sklearn (GridSearchCV)...")
        etapa_mlp_sklearn(features_df, cfg)

    if args.keras:
        print("\n[Extra] Redes neurais Keras...")
        etapa_keras(features_df, cfg, args.plot)

    print("\nConcluído.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CurvantML — pipeline completo de experimentos'
    )
    parser.add_argument('--plot',  action='store_true', help='Gerar plots durante a execução')
    parser.add_argument('--isl',   action='store_true', help='Treinar modelos preditivos de ISL')
    parser.add_argument('--mlp',   action='store_true', help='Treinar MLP sklearn com GridSearchCV')
    parser.add_argument('--keras', action='store_true', help='Treinar redes neurais Keras (MLP, GRU, LSTM)')
    main(parser.parse_args())
