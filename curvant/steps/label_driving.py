import pandas as pd
from curvant.driving.risk_measures import caracterizar_conducao

def run(dfs_curves: pd.DataFrame, cfg: dict, mostrar_risco: bool = True) -> pd.DataFrame:
    """Classifica cada janela de curva com a taxonomia explícita de risco."""
    da = cfg['risk_measures']

    partes = []
    for _, dt in dfs_curves.groupby('id_route', sort=False):
        resultado = caracterizar_conducao(
            dt,
            kamm_alpha=da.get('kamm_alpha', 0.7),
            limiar_accel_lateral=da.get('limiar_accel_lateral', 2.0),
            zz_limiar_bearing=da.get('zigue_zague', {}).get('limiar_bearing', 15.0),
            zz_limiar_accel=da.get('zigue_zague', {}).get('limiar_ctp', 0.3),
            zz_min_mudancas=da.get('zigue_zague', {}).get('min_mudancas', 3),
        )
        if not resultado.empty:
            partes.append(resultado)

    df_analysis = pd.concat(partes, ignore_index=True)

    if mostrar_risco:
        n_perigosa = df_analysis['manobra_combinado'].sum()
        n_segura   = (~df_analysis['manobra_combinado']).sum()
        n_accel    = df_analysis['manobra_accel'].sum()
        n_lateral  = df_analysis['manobra_lateral'].sum()
        n_zz       = df_analysis['manobra_ziguezague'].sum()
        print(f"  Janelas — Risco: {n_perigosa} | Segura: {n_segura}")
        print(f"  Critérios — Accel: {n_accel} | Lateral: {n_lateral} | ZZ: {n_zz}")

    return df_analysis
