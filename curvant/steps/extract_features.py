import pandas as pd
from curvant.driving.curve_detection import identificar_trechos_curvos
from curvant.driving.features import extrair_features

def run(df_analysis: pd.DataFrame, cfg: dict, feat_cfg: dict, mostrar_risco: bool = True) -> pd.DataFrame:
    """
    Extrai features e alvos por curva.

    Assume que a rota é pré-conhecida (ex: frota com rotas fixas, navegação GPS)
    e que o trajeto completo (lat, lon) está disponível.
    """
    df_analysis = df_analysis.copy()
    for col in ['manobra_frenagem', 'manobra_ziguezague', 'manobra_combinado']:
        if col in df_analysis.columns:
            df_analysis[col] = df_analysis[col].astype(int)

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

    if mostrar_risco:
        n_perigosa = features_df['manobra_combinado_curva'].sum()
        n_segura   = (features_df['manobra_combinado_curva'] == 0).sum()
        print(f"  {len(features_df)} amostras — Risco: {n_perigosa} | Segura: {n_segura}")
    else:
        print(f"  {len(features_df)} amostras (curvas)")

    return features_df
