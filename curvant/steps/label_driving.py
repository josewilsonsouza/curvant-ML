import pandas as pd
from curvant.driving.risk_measures import caracterizar_conducao

def run(dfs_curves: pd.DataFrame, cfg: dict, mostrar_risco: bool = True) -> pd.DataFrame:
    """Classifica cada janela de curva com a taxonomia explícita de risco."""
    da = cfg['risk_measures']

    partes = []
    for _, dt in dfs_curves.groupby('id_route', sort=False):
        resultado = caracterizar_conducao(
            dt,
            limiar_desaceleracao=da.get('limiar_desaceleracao', 2.0),
            margem_histerese=da.get('margem_histerese', 0.0),
            zz_limiar_bearing=da.get('zigue_zague', {}).get('limiar_bearing', 15.0),
            zz_limiar_ctp=da.get('zigue_zague', {}).get('limiar_ctp', 0.3),
            zz_min_mudancas=da.get('zigue_zague', {}).get('min_mudancas', 3),
        )
        if not resultado.empty:
            partes.append(resultado)

    df_analysis = pd.concat(partes, ignore_index=True)

    if mostrar_risco:
        n_perigosa = df_analysis['manobra_combinado'].sum()
        n_segura   = (~df_analysis['manobra_combinado']).sum()
        n_frenagem = df_analysis['manobra_frenagem'].sum()
        n_zz       = df_analysis['manobra_ziguezague'].sum()
        print(f"  Janelas — Risco: {n_perigosa} | Segura: {n_segura}")
        print(f"  Critérios — Frenagem tardia: {n_frenagem} | Zigue-zague: {n_zz}")
        n_indef = df_analysis['manobra_indefinido'].sum()
        if n_indef:
            print(f"  Indefinidas (histerese): {n_indef} janelas na faixa do limiar")

    return df_analysis
