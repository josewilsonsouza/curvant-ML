import argparse
import os

import numpy as np
import pandas as pd

from curvant.steps import (
    detect_curves, label_driving, extract_features, train_tab, train_seq,
)
from curvant.utils.config import carregar_config, carregar_features_config
from curvant.driving.features import configurar_features_ativas, resolver_features_flag

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


_DESCRICAO_ALVO = {
    'risk':     'Segura/Risco (manobra_combinado_curva)',
    'isl':      'faixa de ISL (baixo/medio/alto)',
    'velocity': 'velocidade crítica (v_critica)',
}


def _add_alvo(sub, nome: str, ajuda: str):
    """Subcomando de representação, com o alvo como argumento posicional."""
    p = sub.add_parser(nome, help=ajuda, description=ajuda)
    p.add_argument(
        'target', choices=list(_DESCRICAO_ALVO),
        help='; '.join(f'{k} = {v}' for k, v in _DESCRICAO_ALVO.items()),
    )
    p.add_argument('--no-plot', action='store_true', help='Não gerar gráficos (mais rápido)')
    p.add_argument('--rebuild', action='store_true', help='Ignorar o cache e reprocessar as etapas 1-5')
    return p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='cvt',
        description=(
            'CurvantML - dois eixos: a REPRESENTAÇÃO da entrada (tab | seq) e o ALVO '
            '(risk | isl | velocity). O mesmo alvo roda nas duas representações, que é como '
            'se compara se a série bruta ganha das features agregadas.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Exemplos:\n'
            '  cvt tab risk           features agregadas -> Segura/Risco\n'
            '  cvt seq velocity       série bruta        -> velocidade (+ISL, multitarefa)\n'
            '  cvt tab isl            features agregadas -> faixa de ISL\n'
            '  cvt seq isl            série bruta        -> faixa de ISL\n'
            '  cvt preprocess         limpa os dados brutos\n'
        ),
    )
    sub = parser.add_subparsers(dest='repr')

    _add_alvo(sub, 'tab', 'Representação tabular: features agregadas por curva')
    _add_alvo(sub, 'seq', 'Representação de sequência: janela bruta de série temporal')

    pre = sub.add_parser('preprocess', help='Limpa os dados brutos (detecta o arquivo principal)')
    pre.add_argument('--input',  type=str, default=None, help='Arquivo bruto a limpar')
    pre.add_argument('--output', type=str, default=None, help='Arquivo limpo de saída')

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.repr is None:
        parser.print_help()
        return

    if args.repr == 'preprocess':
        from curvant.utils.preprocessing import executar_preprocessamento
        in_path = args.input or _input_bruto()
        print(f"[preprocess] Limpando os dados brutos: {in_path}")
        executar_preprocessamento(in_path, args.output)
        return

    plot     = not args.no_plot
    cfg      = carregar_config()
    feat_cfg = carregar_features_config()
    df_analysis, features_df = _carregar_features(
        cfg, feat_cfg, args.rebuild, mostrar_risco=(args.target == 'risk'),
    )

    print(f"\n[{args.repr} {args.target}] {_DESCRICAO_ALVO[args.target]}...")
    if args.repr == 'tab':
        # Só o tabular usa a whitelist chapada de features; a sequência define seus canais
        # por sensors/scalares_extras, lidos direto do features.yaml pelo treinador.
        configurar_features_ativas(resolver_features_flag(feat_cfg, 'tab'))
        train_tab.run(args.target, features_df, cfg, plot)
    else:
        train_seq.run(args.target, df_analysis, features_df, cfg, feat_cfg, plot)

    print("\nDone.")