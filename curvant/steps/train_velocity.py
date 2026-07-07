import pandas as pd

_DIR_VELOCIDADE = 'results/velocidade'

def run(df_analysis: pd.DataFrame, features_df: pd.DataFrame, cfg: dict, feat_cfg: dict, plot: bool) -> None:
    """Previsão da velocidade crítica (e ISL derivado) sobre a série temporal pré-curva."""
    from curvant.models import treinar_regressao_ts
    treinar_regressao_ts(df_analysis, features_df, cfg, feat_cfg, plot=plot, outdir=_DIR_VELOCIDADE)
