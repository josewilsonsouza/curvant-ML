"""
Representação de sequência: modelos sobre a janela bruta pré-curva (50 passos de tempo),
recortada de df_analysis. Contraparte de train_tab, com os mesmos três alvos.

Um alvo por subcomando (`cvt seq <alvo>`):
    velocity  - velocidade crítica (v_critica); com config.temporais.multitarefa, uma cabeça
                de classe prevê a faixa de ISL na mesma passada
    isl       - isl_p95, regressão
    correcao  - correção tardia dentro da curva, classificação binária

Os modelos (clássicos ou neurais) saem de config.temporais.model, então a família do modelo é
independente da representação: dá para rodar XGBoost sobre a série e LSTM sobre a série.
"""

import pandas as pd

_COLUNA = {
    'velocity': 'v_critica',
    'isl':      'isl_p95',
    'correcao': 'correcao_tardia_curva',
}

_DIR = {
    'velocity': 'results/seq/velocity',
    'isl':      'results/seq/isl',
    'correcao': 'results/seq/correcao',
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

    # Mesmo tratamento do tabular: curvas na faixa de histerese do limiar saem do treino.
    if target == 'correcao' and 'correcao_indefinida_curva' in features_df.columns:
        n_desc = int(features_df['correcao_indefinida_curva'].sum())
        if n_desc:
            features_df = features_df[features_df['correcao_indefinida_curva'] == 0]
            print(f"  Histerese: {n_desc} curvas indefinidas descartadas ({len(features_df)} restantes)")

    treinar_regressao_ts(
        df_analysis, features_df, cfg, feat_cfg,
        plot=plot,
        outdir=_DIR[target],
        target=_COLUNA[target],
    )
