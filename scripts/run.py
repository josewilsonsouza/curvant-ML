"""
CurvantML — Pipeline de Experimentos

Uso:
    python scripts/run.py                            # modelos clássicos (modo1)
    python scripts/run.py --plot                     # gera plots
    python scripts/run.py --isl                      # + ISL 3-class + regressões
    python scripts/run.py --mlp                      # + MLP sklearn (GridSearchCV)
    python scripts/run.py --optuna                   # + Optuna para XGBoost e RandomForest
    python scripts/run.py --pytorch                  # + MLP multi-tarefa PyTorch
    python scripts/run.py --modo modo2               # Modo 2 (sem geometria da curva seguinte)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from src.pipeline import (
    etapa_curvas, etapa_analise_conducao, etapa_features,
    etapa_ml_classico, etapa_ml_otimizado, etapa_mlp_sklearn,
    etapa_isl_modelo, etapa_accel_regressao,
)
from utils.config import carregar_config


def main(args: argparse.Namespace) -> None:
    cfg = carregar_config()

    # Prioridade: dataset maior (novo) > dataset antigo > raw fallback
    candidates = [
        ('data/eletro_rjdf_serra_rjmgba_janeiro_clean.parquet', True),
        ('data/eletro_rjdf_serra_clean.parquet',                True),
        ('data/eletro_rjdf_serra_rjmgba_janeiro.parquet',       False),
        ('data/eletro_rjdf_serra.parquet',                      False),
    ]
    data_path, is_clean = next(((p, c) for p, c in candidates if os.path.exists(p)), (None, False))
    if data_path is None:
        raise FileNotFoundError("Nenhum arquivo de dados encontrado em data/")

    print("[1/6] Carregando dados...")
    if is_clean:
        print(f"  Usando: {data_path}")
        print("  Para regenerar: python scripts/preprocess_data.py --input <raw.parquet>")
    else:
        print(f"  AVISO: dados limpos não encontrados. Execute 'python scripts/preprocess_data.py --input {data_path}' primeiro.")
        print(f"  Usando raw: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} registros | {df['id_route'].nunique()} trajetos")

    print("\n[2/6] Detectando curvas...")
    dfs_curves = etapa_curvas(df, cfg)

    print("\n[3/6] Calculando aceleração centrípeta e absoluta...")
    # Clipar raio mínimo para evitar ISL → ∞ por ruído B-spline / GPS
    # R < 5 m é fisicamente impossível para um veículo em movimento normal
    R_MIN = cfg.get('curve_detection', {}).get('raio_min', 5.0)
    raio_clipado = dfs_curves['raio_curvatura'].clip(lower=R_MIN)
    dfs_curves['ctp_accel'] = (dfs_curves['vehicle_speed'] / 3.6) ** 2 / raio_clipado
    dfs_curves['abs_accel'] = np.sqrt(
        dfs_curves['accel_x'] ** 2 + dfs_curves['accel_y'] ** 2
    )

    print("\n[4/6] Analisando risco na condução...")
    df_analysis = etapa_analise_conducao(dfs_curves, cfg, args.plot)

    print("\n[5/6] Identificando trechos curvos e extraindo features...")
    features_df = etapa_features(df_analysis, cfg, modo=args.modo)
    print(f"  Modo: {args.modo}")

    print("\n[6/6] Modelos clássicos de ML...")
    if args.optuna:
        etapa_ml_otimizado(features_df, cfg, args.plot)
    else:
        etapa_ml_classico(features_df, cfg, args.plot)

    if args.isl:
        print("\n[Extra] ISL — classificação 3 classes (baixo/medio/alto)...")
        etapa_isl_modelo(features_df, cfg, args.plot)
        print("\n[Extra] P1 — Regressão aceleração dentro da curva...")
        etapa_accel_regressao(features_df, cfg, args.plot)

    if args.mlp:
        print("\n[Extra] MLP sklearn (GridSearchCV)...")
        etapa_mlp_sklearn(features_df, cfg)

    if args.pytorch:
        print("\n[Extra] MLP multi-tarefa PyTorch...")
        from src.pipeline import etapa_pytorch
        etapa_pytorch(features_df, cfg)

    print("\nConcluído.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CurvantML — pipeline completo de experimentos'
    )
    parser.add_argument('--modo',    choices=['modo1', 'modo2'], default='modo1',
                        help='Modo 1: features com geometria da curva; Modo 2: só OBD+GPS (default: modo1)')
    parser.add_argument('--plot',    action='store_true', help='Gerar plots durante a execução')
    parser.add_argument('--isl',     action='store_true', help='Treinar modelos preditivos de ISL')
    parser.add_argument('--mlp',     action='store_true', help='Treinar MLP sklearn com GridSearchCV')
    parser.add_argument('--optuna',  action='store_true', help='Usar Optuna para XGBoost e RandomForest')
    parser.add_argument('--pytorch', action='store_true', help='Treinar MLP multi-tarefa PyTorch')
    main(parser.parse_args())
