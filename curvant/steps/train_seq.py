"""
Representação de sequência: modelos sobre a janela bruta pré-curva (50 passos de tempo),
recortada de df_analysis. Contraparte de train_tab, com os mesmos três alvos.

Um alvo por subcomando (`cvt seq <alvo>`):
    risk      - Segura/Risco (manobra_combinado_curva), classificação binária
    isl       - faixa de ISL (baixo/medio/alto), classificação multiclasse
    velocity  - velocidade crítica (v_critica); com config.temporais.multitarefa, uma cabeça
                de classe prevê a faixa de ISL na mesma passada

Os modelos (clássicos ou neurais) saem de config.temporais.model, então a família do modelo é
independente da representação: dá para rodar XGBoost sobre a série e LSTM sobre a série.
"""

import pandas as pd

_COLUNA = {
    'risk':     'manobra_combinado_curva',
    'isl':      'isl_class',
    'velocity': 'v_critica',
}

_DIR = {
    'risk':     'results/seq/risk',
    'isl':      'results/seq/isl',
    'velocity': 'results/seq/velocity',
}

ALVOS = tuple(_COLUNA)


def run(
    target: str,
    df_analysis: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: dict,
    feat_cfg: dict,
    plot: bool,
) -> None:
    if target not in _COLUNA:
        raise ValueError(f"Alvo '{target}' não existe em `cvt seq`. Use um de: {list(ALVOS)}")

    from curvant.models import treinar_regressao_ts

    # Mesmo tratamento do tabular: curvas na faixa de histerese do limiar saem do alvo de risco.
    if target == 'risk' and 'manobra_indefinida_curva' in features_df.columns:
        n_desc = int(features_df['manobra_indefinida_curva'].sum())
        if n_desc:
            features_df = features_df[features_df['manobra_indefinida_curva'] == 0]
            print(f"  Histerese: {n_desc} curvas indefinidas descartadas ({len(features_df)} restantes)")

    treinar_regressao_ts(
        df_analysis, features_df, cfg, feat_cfg,
        plot=plot,
        outdir=_DIR[target],
        target=_COLUNA[target],
    )
