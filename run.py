"""
CurvantML Experimentos

As flags escolhem O QUE prever. Cada uma escreve seus resultados em results/<alvo>/.

    python run.py --risco              # classifica a curva em Segura/Risco
    python run.py --risco --otimizar   # + ajuste de hiperparâmetros (Optuna)
    python run.py --isl                # classifica o ISL (3 classes) + baseline físico
    python run.py --velocidade         # prevê a velocidade crítica e deriva o ISL
    python run.py --aceleracao         # regressão das acelerações dentro da curva
    python run.py --multitarefa        # MLP PyTorch: ISL + manobras juntos
    python run.py --importancia        # importância das features

    python run.py --velocidade --rebuild   # ignora o cache e reprocessa
    python run.py --isl --no-plot          # sem gráficos (mais rápido)

Os gráficos são gerados por default (em results/<alvo>/); use --no-plot para pular.
Sem nenhuma flag, mostra esta ajuda.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from curvant.pipeline import (
    etapa_curvas, etapa_analise_conducao, etapa_features,
    etapa_ml_classico, etapa_ml_otimizado,
    etapa_isl_modelo, etapa_accel_regressao, etapa_regressao_ts,
    etapa_importancia_features, etapa_baseline_fisico, etapa_pytorch,
)
from curvant.utils.config import carregar_config

_FLAGS_ACAO = ('risco', 'isl', 'velocidade', 'aceleracao', 'multitarefa', 'importancia')

_CACHE_ANALYSIS = 'data/.cache_df_analysis.parquet'
_CACHE_FEATURES = 'data/.cache_features_df.parquet'

# Arquivos brutos (sem _clean) para o --preprocessar, do principal para o menor
_RAW_CANDIDATOS = [
    'data/eletro_rjdf_serra_rjmgba_janeiro.parquet',
    'data/eletro_rjdf_serra.parquet',
]


def _input_bruto() -> str:
    """Primeiro arquivo bruto existente (maior/principal primeiro)."""
    caminho = next((p for p in _RAW_CANDIDATOS if os.path.exists(p)), None)
    if caminho is None:
        raise FileNotFoundError(
            "Nenhum arquivo bruto encontrado em data/. Esperado um de: "
            + ', '.join(_RAW_CANDIDATOS)
        )
    return caminho


def _cache_valido(cache_path: str, data_path: str) -> bool:
    """True se o cache existe e é mais recente que o arquivo de dados."""
    if not os.path.exists(cache_path):
        return False
    return os.path.getmtime(cache_path) >= os.path.getmtime(data_path)


def _carregar_features(cfg: dict, rebuild: bool, mostrar_risco: bool = False):
    """Carrega features do cache ou reconstrói o pipeline (etapas 1-5).

    mostrar_risco: imprime as contagens de manobra/risco na preparação (só faz
    sentido com --risco; nas outras flags elas só poluem a saída).
    """
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
    dfs_curves['abs_accel'] = np.sqrt(dfs_curves['accel_x'] ** 2 + dfs_curves['accel_y'] ** 2)

    print("\n[4/5] Caracterizando a condução...")
    df_analysis = etapa_analise_conducao(dfs_curves, cfg, mostrar_risco=mostrar_risco)

    print("\n[5/5] Identificando trechos curvos e extraindo features...")
    features_df = etapa_features(df_analysis, cfg, mostrar_risco=mostrar_risco)

    df_analysis.to_parquet(_CACHE_ANALYSIS, index=False)
    features_df.to_parquet(_CACHE_FEATURES, index=False)
    print(f"  [cache] Intermediários salvos em {_CACHE_ANALYSIS} e {_CACHE_FEATURES}")
    return df_analysis, features_df


def main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    acoes = any(getattr(args, f) for f in _FLAGS_ACAO)
    if not acoes and not args.preprocessar:
        parser.print_help()
        return

    if args.preprocessar:
        from curvant.utils.preprocessing import executar_preprocessamento
        in_path = args.input or _input_bruto()
        print(f"[preprocessar] Limpando dados brutos: {in_path}")
        executar_preprocessamento(in_path, args.output)
        if not acoes:
            return

    plot = not args.no_plot   # plota por default; --no-plot pula os gráficos
    cfg = carregar_config()
    df_analysis, features_df = _carregar_features(cfg, args.rebuild, mostrar_risco=args.risco)

    if args.risco:
        print("\n[risco] Classificação Segura/Risco...")
        if args.otimizar:
            etapa_ml_otimizado(features_df, cfg, plot)
        else:
            etapa_ml_classico(features_df, cfg, plot)

    if args.isl:
        print("\n[isl] Baseline físico (Monte Carlo / fórmula ISL)...")
        etapa_baseline_fisico(features_df, cfg)
        print("\n[isl] Classificação do ISL (3 classes)...")
        etapa_isl_modelo(features_df, cfg, plot)

    if args.aceleracao:
        print("\n[aceleracao] Regressão das acelerações na curva...")
        etapa_accel_regressao(features_df, cfg, plot)

    if args.velocidade:
        print("\n[velocidade] Previsão da velocidade crítica (v_critica)...")
        etapa_regressao_ts(df_analysis, features_df, cfg, plot)

    if args.multitarefa:
        print("\n[multitarefa] MLP multi-tarefa PyTorch...")
        etapa_pytorch(features_df, cfg, plot)

    if args.importancia:
        print("\n[importancia] Importância das features (XGBoost)...")
        etapa_importancia_features(features_df, cfg)

    print("\nConcluído.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CurvantML - escolha o que prever com uma flag de target.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Alvos
    parser.add_argument('--risco',       action='store_true', help='Classifica a curva em Segura/Risco')
    parser.add_argument('--isl',         action='store_true', help='Classifica o ISL (3 classes) + baseline físico')
    parser.add_argument('--velocidade',  action='store_true', help='Prevê a velocidade crítica e deriva o ISL')
    parser.add_argument('--aceleracao',  action='store_true', help='Regressão das acelerações dentro da curva')
    parser.add_argument('--multitarefa', action='store_true', help='MLP PyTorch: ISL + manobras juntos')
    parser.add_argument('--importancia', action='store_true', help='Importância das features (XGBoost)')
    # Modificadores e utilidades
    parser.add_argument('--otimizar',    action='store_true', help='Aplica Optuna ao --risco')
    parser.add_argument('--no-plot',     action='store_true', help='Não gera os gráficos (mais rápido)')
    parser.add_argument('--rebuild',     action='store_true', help='Ignora o cache e reprocessa as etapas 1-5')
    # Pré-processamento
    parser.add_argument('--preprocessar', action='store_true', help='Limpa os dados brutos (auto-detecta o arquivo principal)')
    parser.add_argument('--input',  type=str, default=None, help='Arquivo bruto a limpar (padrão: auto-detecta); usado com --preprocessar')
    parser.add_argument('--output', type=str, default=None, help='Arquivo de saída limpo (padrão: <input>_clean.parquet); usado com --preprocessar')
    main(parser.parse_args(), parser)
