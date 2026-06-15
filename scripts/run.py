"""
CurvantML — Pipeline de Experimentos

Uso:
    python scripts/run.py                            # modelos clássicos
    python scripts/run.py --classical                # modelos clássicos (explícito)
    python scripts/run.py --classical --plot         # + matrizes de confusão
    python scripts/run.py --optuna                   # Optuna (XGBoost e RandomForest)
    python scripts/run.py --isl                      # ISL 3-class + regressão P1
    python scripts/run.py --mlp                      # MLP sklearn (GridSearchCV)
    python scripts/run.py --pytorch                  # MLP multi-tarefa PyTorch
    python scripts/run.py --ts                       # regressão série temporal (GRU/LSTM/CNN1D/MLP)
    python scripts/run.py --ts --rebuild             # força reprocessamento (ignora cache)
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
    etapa_isl_modelo, etapa_accel_regressao, etapa_regressao_ts,
    etapa_importancia_features,
)
from utils.config import carregar_config

_FLAGS_MODELO = ('classical', 'optuna', 'isl', 'mlp', 'pytorch', 'ts', 'fi')


_CACHE_ANALYSIS  = 'data/.cache_df_analysis.parquet'
_CACHE_FEATURES  = 'data/.cache_features_df.parquet'


def _cache_valido(cache_path: str, data_path: str) -> bool:
    """True se o cache existe e é mais recente que o arquivo de dados."""
    if not os.path.exists(cache_path):
        return False
    return os.path.getmtime(cache_path) >= os.path.getmtime(data_path)


def main(args: argparse.Namespace) -> None:
    cfg = carregar_config()

    # Se nenhum flag de modelo foi passado, roda clássicos por padrão
    nenhum_modelo = not any(getattr(args, f) for f in _FLAGS_MODELO)
    rodar_classico = args.classical or args.optuna or nenhum_modelo

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

    cache_features = _CACHE_FEATURES
    usar_cache = (
        not args.rebuild
        and _cache_valido(_CACHE_ANALYSIS, data_path)
        and _cache_valido(cache_features, data_path)
    )

    if usar_cache:
        print(f"[cache] Carregando intermediários do cache (use --rebuild para reprocessar)...")
        df_analysis = pd.read_parquet(_CACHE_ANALYSIS)
        features_df = pd.read_parquet(cache_features)
        print(f"  df_analysis: {len(df_analysis):,} linhas | features_df: {len(features_df):,} curvas")
    else:
        if not is_clean:
            print(f"  AVISO: dados limpos não encontrados. Execute 'python scripts/preprocess_data.py --input {data_path}' primeiro.")

        print("[1/5] Carregando dados...")
        print(f"  Usando: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"  {len(df):,} registros | {df['id_route'].nunique()} trajetos")

        print("\n[2/5] Detectando curvas...")
        dfs_curves = etapa_curvas(df, cfg)

        print("\n[3/5] Calculando aceleração centrípeta e absoluta...")
        R_MIN = cfg.get('curve_detection', {}).get('raio_min', 5.0)
        raio_clipado = dfs_curves['raio_curvatura'].clip(lower=R_MIN)
        dfs_curves['ctp_accel'] = (dfs_curves['vehicle_speed'] / 3.6) ** 2 / raio_clipado
        dfs_curves['abs_accel'] = np.sqrt(
            dfs_curves['accel_x'] ** 2 + dfs_curves['accel_y'] ** 2
        )

        print("\n[4/5] Analisando risco na condução...")
        df_analysis = etapa_analise_conducao(dfs_curves, cfg, args.plot)

        print("\n[5/5] Identificando trechos curvos e extraindo features...")
        features_df = etapa_features(df_analysis, cfg)

        # Salva cache para próximas execuções
        df_analysis.to_parquet(_CACHE_ANALYSIS, index=False)
        features_df.to_parquet(cache_features, index=False)
        print(f"  [cache] Intermediários salvos em {_CACHE_ANALYSIS} e {cache_features}")

    if rodar_classico:
        print("\n[ML] Modelos clássicos...")
        if args.optuna:
            etapa_ml_otimizado(features_df, cfg, args.plot)
        else:
            etapa_ml_classico(features_df, cfg, args.plot)

    if args.isl:
        print("\n[ML] ISL — classificação 3 classes (baixo/medio/alto)...")
        etapa_isl_modelo(features_df, cfg, args.plot)
        print("\n[ML] P1 — Regressão aceleração dentro da curva...")
        etapa_accel_regressao(features_df, cfg, args.plot)

    if args.mlp:
        print("\n[ML] MLP sklearn (GridSearchCV)...")
        etapa_mlp_sklearn(features_df, cfg)

    if args.fi:
        print("\n[FI] Importância de features (XGBoost)...")
        etapa_importancia_features(features_df, cfg)

    if args.pytorch:
        print("\n[DL] MLP multi-tarefa PyTorch...")
        from src.pipeline import etapa_pytorch
        etapa_pytorch(features_df, cfg)

    if args.ts:
        _m = cfg.get('time_series_regression', {}).get('model', 'gru')
        ts_label = ', '.join(_m).upper() if isinstance(_m, list) else _m.upper()
        print(f"\n[DL] Regressão série temporal ({ts_label})...")
        etapa_regressao_ts(df_analysis, features_df, cfg, args.plot)

    print("\nConcluído.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CurvantML — pipeline completo de experimentos'
    )
    parser.add_argument('--plot',      action='store_true', help='Gerar plots durante a execução')
    parser.add_argument('--classical', action='store_true', help='Modelos clássicos (LogReg, SVM, RF, XGB)')
    parser.add_argument('--optuna',    action='store_true', help='Optuna para XGBoost e RandomForest')
    parser.add_argument('--isl',       action='store_true', help='ISL 3-class + regressão P1')
    parser.add_argument('--mlp',       action='store_true', help='MLP sklearn com GridSearchCV')
    parser.add_argument('--pytorch',   action='store_true', help='MLP multi-tarefa PyTorch (4 heads)')
    parser.add_argument('--ts',        action='store_true', help='Regressão série temporal (GRU/LSTM/CNN1D/MLP)')
    parser.add_argument('--fi',        action='store_true', help='Importância de features (XGBoost gain)')
    parser.add_argument('--rebuild',   action='store_true', help='Ignora cache e reprocessa etapas 1-5')
    main(parser.parse_args())
