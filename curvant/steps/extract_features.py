import pandas as pd
from curvant.driving.curve_detection import identificar_trechos_curvos
from curvant.driving.features import extrair_features

def run(df_analysis: pd.DataFrame, cfg: dict, feat_cfg: dict, mostrar_correcao: bool = True) -> pd.DataFrame:
    """
    Extrai features e alvos por curva.

    Assume que a rota é pré-conhecida (ex: frota com rotas fixas, navegação GPS)
    e que o trajeto completo (lat, lon) está disponível.
    """
    df_analysis = df_analysis.copy()
    if 'correcao_tardia' in df_analysis.columns:
        df_analysis['correcao_tardia'] = df_analysis['correcao_tardia'].astype(int)

    dfs_trechos = identificar_trechos_curvos(df_analysis)
    ex = feat_cfg['extracao']
    features_df = extrair_features(
        dfs_trechos,
        precurva_distancia=ex.get('precurva_distancia'),
        precurva_desacel_confort=ex.get('precurva_desacel_confort', 2.5),
        precurva_distancia_min=ex.get('precurva_distancia_min', 50.0),
        precurva_distancia_max=ex.get('precurva_distancia_max', 400.0),
        lead_gap=ex.get('lead_gap', 0.0),
        vars_sensor=ex.get('vars_sensor'),
    )

    if mostrar_correcao:
        n_pos = int(features_df['correcao_tardia_curva'].sum())
        print(f"  {len(features_df)} curvas, com correção tardia: {n_pos} "
              f"({100*n_pos/len(features_df):.1f}%)")
    else:
        print(f"  {len(features_df)} amostras (curvas)")

    return features_df
