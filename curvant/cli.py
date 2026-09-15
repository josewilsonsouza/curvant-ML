import argparse
import os

import numpy as np
import pandas as pd

from curvant.steps import (
    detect_curves, label_driving, extract_features, train_tab, train_seq,
)
from curvant.utils.config import carregar_config, carregar_features_config
from curvant.driving.features import configurar_features_ativas, resolver_features_flag

_RAW_CANDIDATOS = [
    'data/eletro_rjdf_serra_rjmgba_janeiro_agda.parquet',
    'data/eletro_rjdf_serra_rjmgba_janeiro.parquet',
    'data/eletro_rjdf_serra.parquet',
]

def _nome_base(valor: str) -> str:
    """Caminho ou nome -> nome base do dataset (sem pasta, sem .parquet, sem _clean)."""
    nome = os.path.basename(valor)
    if nome.endswith('.parquet'):
        nome = nome[: -len('.parquet')]
    if nome.endswith('_clean'):
        nome = nome[: -len('_clean')]
    return nome

def _resolver_dataset(cfg: dict, arg_data: str | None) -> tuple[str, bool, str]:
    """Qual arquivo usar: --data > config data.dataset > auto-detecção.

    Devolve (caminho, eh_limpo, nome_base), preferindo o _clean quando existir.
    """
    escolhido = arg_data or (cfg.get('data') or {}).get('dataset')
    if escolhido:
        nome = _nome_base(str(escolhido))
        for caminho, limpo in ((f'data/{nome}_clean.parquet', True), (f'data/{nome}.parquet', False)):
            if os.path.exists(caminho):
                return caminho, limpo, nome
        disponiveis = sorted({
            _nome_base(f) for f in os.listdir('data')
            if f.endswith('.parquet') and not f.startswith('.')
        })
        raise FileNotFoundError(
            f"Dataset '{nome}' não encontrado em data/. Disponíveis: {', '.join(disponiveis)}"
        )
    for bruto in _RAW_CANDIDATOS:
        caminho = f'data/{_nome_base(bruto)}_clean.parquet'
        if os.path.exists(caminho):
            return caminho, True, _nome_base(bruto)
    for bruto in _RAW_CANDIDATOS:
        if os.path.exists(bruto):
            return bruto, False, _nome_base(bruto)
    raise FileNotFoundError("Nenhum arquivo de dados encontrado em data/")


def _input_bruto(cfg: dict) -> str:
    escolhido = (cfg.get('data') or {}).get('dataset')
    if escolhido:
        caminho = f'data/{_nome_base(str(escolhido))}.parquet'
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo bruto do dataset não encontrado: {caminho}")
        return caminho
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


def _carregar_features(cfg: dict, feat_cfg: dict, rebuild: bool, mostrar_correcao: bool = False,
                       arg_data: str | None = None):
    """Carrega features do cache ou reconstrói o pipeline (etapas 1-5)."""
    data_path, is_clean, nome = _resolver_dataset(cfg, arg_data)
    cache_analysis = f'data/.cache_df_analysis_{nome}.parquet'
    cache_features = f'data/.cache_features_df_{nome}.parquet'

    usar_cache = (
        not rebuild
        and _cache_valido(cache_analysis, data_path)
        and _cache_valido(cache_features, data_path)
    )

    if usar_cache:
        print(f"[cache] Carregando intermediários de '{nome}' do cache (use --rebuild para reprocessar)...")
        df_analysis = pd.read_parquet(cache_analysis)
        features_df = pd.read_parquet(cache_features)
        print(f"  df_analysis: {len(df_analysis):,} linhas | features_df: {len(features_df):,} curvas")
        return df_analysis, features_df

    if not is_clean:
        print(f"  AVISO: dados limpos não encontrados. Execute 'cvt preprocess --data {data_path}' primeiro.")

    print("[1/5] Carregando dados...")
    print(f"  Usando: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} registros | {df['id_route'].nunique()} trajetos")

    print("\n[2/5] Detectando curvas...")
    dfs_curves = detect_curves.run(df, cfg)

    print("\n[3/5] Calculando aceleração centrípeta e absoluta...")
    R_MIN = cfg.get('curve_detection', {}).get('raio_min', 20.0)
    raio_clipado = dfs_curves['raio_curvatura'].clip(lower=R_MIN)
    dfs_curves['ctp_accel'] = (dfs_curves['vehicle_speed'] / 3.6) ** 2 / raio_clipado
    dfs_curves['abs_accel'] = np.sqrt(dfs_curves['accel_x'] ** 2 + dfs_curves['accel_y'] ** 2)

    print("\n[4/5] Caracterizando a condução...")
    df_analysis = label_driving.run(dfs_curves, cfg, mostrar_correcao=mostrar_correcao)

    print("\n[5/5] Identificando trechos curvos e extraindo features...")
    features_df = extract_features.run(df_analysis, cfg, feat_cfg, mostrar_correcao=mostrar_correcao)

    df_analysis.to_parquet(cache_analysis, index=False)
    features_df.to_parquet(cache_features, index=False)
    print(f"  [cache] Intermediários salvos em {cache_analysis} e {cache_features}")
    return df_analysis, features_df


_DESCRICAO_ALVO = {
    'velocity': 'velocidade crítica (v_critica)',
    'isl':      'isl_p95, com a faixa cortada da predição',
    'correcao': 'correção tardia dentro da curva (correcao_tardia_curva)',
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
    p.add_argument('--data', type=str, default=None,
                   help='Dataset a usar (nome ou caminho em data/); sobrescreve config data.dataset')
    return p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='cvt',
        description=(
            'CurvantML - dois eixos: a REPRESENTAÇÃO da entrada (tab | seq) e o ALVO '
            '(velocity | isl | correcao). O mesmo alvo roda nas duas representações, que é '
            'como se compara se a série bruta ganha das features agregadas.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Exemplos:\n'
            '  cvt tab velocity       features agregadas -> velocidade critica\n'
            '  cvt seq velocity       serie bruta        -> velocidade (+ISL, multitarefa)\n'
            '  cvt tab isl            features agregadas -> isl_p95 (+faixa)\n'
            '  cvt tab correcao       features agregadas -> correcao tardia\n'
            '  cvt preprocess         limpa os dados brutos\n'
        ),
    )
    sub = parser.add_subparsers(dest='repr')

    _add_alvo(sub, 'tab', 'Representação tabular: features agregadas por curva')
    _add_alvo(sub, 'seq', 'Representação de sequência: janela bruta de série temporal')

    pre = sub.add_parser('preprocess', help='Limpa os dados brutos (detecta o arquivo principal)')
    pre.add_argument('--data',   type=str, default=None, help='Arquivo bruto a limpar')
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
        in_path = args.data or _input_bruto(carregar_config())
        print(f"[preprocess] Limpando os dados brutos: {in_path}")
        executar_preprocessamento(in_path, args.output)
        return

    plot     = not args.no_plot
    cfg      = carregar_config()
    feat_cfg = carregar_features_config()
    df_analysis, features_df = _carregar_features(
        cfg, feat_cfg, args.rebuild, mostrar_correcao=(args.target == 'correcao'),
        arg_data=args.data,
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


if __name__ == '__main__':
    main()
