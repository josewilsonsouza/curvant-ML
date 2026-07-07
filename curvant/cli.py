import argparse
import os

import numpy as np
import pandas as pd

from curvant.steps import (
    detect_curves, label_driving, extract_features, train_risk, train_velocity,
)
from curvant.utils.config import carregar_config, carregar_features_config
from curvant.driving.features import configurar_features_ativas, resolver_features_flag

_FLAGS_ACAO = ('risk', 'velocity')

_CACHE_ANALYSIS = 'data/.cache_df_analysis.parquet'
_CACHE_FEATURES = 'data/.cache_features_df.parquet'

_RAW_CANDIDATOS = [
    'data/eletro_rjdf_serra_rjmgba_janeiro.parquet',
    'data/eletro_rjdf_serra.parquet',
]


def _input_bruto() -> str:
    caminho = next((p for p in _RAW_CANDIDATOS if os.path.exists(p)), None)
    if caminho is None:
        raise FileNotFoundError(
            "Nenhum arquivo bruto encontrado em data/. Esperado um de: "
            + ', '.join(_RAW_CANDIDATOS)
        )
    return caminho


def _cache_valido(cache_path: str, data_path: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    return os.path.getmtime(cache_path) >= os.path.getmtime(data_path)

def _carregar_features(cfg: dict, feat_cfg: dict, rebuild: bool, mostrar_risco: bool = False):
    """Carrega features do cache ou reconstrói o pipeline (etapas 1-5)."""
    candidates = [
        ('data/eletro_rjdf_serra_rjmgba_janeiro_clean.parquet', True),
        ('data/eletro_rjdf_serra_clean.parquet',                True),
        ('data/eletro_rjdf_serra_rjmgba_janeiro.parquet',       False),
        ('data/eletro_rjdf_serra.parquet',                      False),
    ]
    data_path, is_clean = next(((p, c) for p, c in candidates if os.path.exists(p)), (None, False))
    if data_path is None:
        raise FileNotFoundError("Nenhum arquivo de dados encontrado em data/")

    usar_cache = (
        not rebuild
        and _cache_valido(_CACHE_ANALYSIS, data_path)
        and _cache_valido(_CACHE_FEATURES, data_path)
    )

    if usar_cache:
        print("[cache] Carregando intermediários do cache (use --rebuild para reprocessar)...")
        df_analysis = pd.read_parquet(_CACHE_ANALYSIS)
        features_df = pd.read_parquet(_CACHE_FEATURES)
        print(f"  df_analysis: {len(df_analysis):,} linhas | features_df: {len(features_df):,} curvas")
        return df_analysis, features_df

    if not is_clean:
        print(f"  AVISO: dados limpos não encontrados. Execute 'cvt --preprocessar --input {data_path}' primeiro.")

    print("[1/5] Carregando dados...")
    print(f"  Usando: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} registros | {df['id_route'].nunique()} trajetos")

    print("\n[2/5] Detectando curvas...")
    dfs_curves = detect_curves.run(df, cfg)

    print("\n[3/5] Calculando aceleração centrípeta e absoluta...")
    R_MIN = cfg.get('curve_detection', {}).get('raio_min', 5.0)
    raio_clipado = dfs_curves['raio_curvatura'].clip(lower=R_MIN)
    dfs_curves['ctp_accel'] = (dfs_curves['vehicle_speed'] / 3.6) ** 2 / raio_clipado
    dfs_curves['abs_accel'] = np.sqrt(dfs_curves['accel_x'] ** 2 + dfs_curves['accel_y'] ** 2)

    print("\n[4/5] Caracterizando a condução...")
    df_analysis = label_driving.run(dfs_curves, cfg, mostrar_risco=mostrar_risco)

    print("\n[5/5] Identificando trechos curvos e extraindo features...")
    features_df = extract_features.run(df_analysis, cfg, feat_cfg, mostrar_risco=mostrar_risco)

    df_analysis.to_parquet(_CACHE_ANALYSIS, index=False)
    features_df.to_parquet(_CACHE_FEATURES, index=False)
    print(f"  [cache] Intermediários salvos em {_CACHE_ANALYSIS} e {_CACHE_FEATURES}")
    return df_analysis, features_df

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CurvantML - escolha o que prever com uma flag de target.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--risk',       action='store_true', help='Classify curve as Safe/Risk')
    parser.add_argument('--velocity',   action='store_true', help='Predict critical speed and derive ISL')
    parser.add_argument('--no-plot',    action='store_true', help='Skip plots (faster)')
    parser.add_argument('--rebuild',    action='store_true', help='Ignore cache and reprocess steps 1-5')
    parser.add_argument('--preprocess', action='store_true', help='Clean raw data (auto-detects main file)')
    parser.add_argument('--input',  type=str, default=None, help='Raw file to clean; used with --preprocess')
    parser.add_argument('--output', type=str, default=None, help='Output clean file; used with --preprocess')
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    acoes = any(getattr(args, f) for f in _FLAGS_ACAO)
    if not acoes and not args.preprocess:
        parser.print_help()
        return

    if args.preprocess:
        from curvant.utils.preprocessing import executar_preprocessamento
        in_path = args.input or _input_bruto()
        print(f"[preprocess] Cleaning raw data: {in_path}")
        executar_preprocessamento(in_path, args.output)
        if not acoes:
            return

    plot = not args.no_plot
    cfg = carregar_config()
    feat_cfg = carregar_features_config()
    df_analysis, features_df = _carregar_features(cfg, feat_cfg, args.rebuild, mostrar_risco=args.risk)

    if args.risk:
        print("\n[risk] Safe/Risk classification...")
        configurar_features_ativas(resolver_features_flag(feat_cfg, 'risk'))
        train_risk.classicos(features_df, cfg, plot)
        print("\n[risk] All models per individual criterion...")
        train_risk.criterios_separados(features_df, cfg, plot)

    if args.velocity:
        print("\n[velocity] Critical speed prediction (v_critica)...")
        train_velocity.run(df_analysis, features_df, cfg, feat_cfg, plot)

    print("\nDone.")