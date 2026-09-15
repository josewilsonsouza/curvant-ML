import pandas as pd
from curvant.driving.late_braking import rotular_correcao_tardia

def run(dfs_curves: pd.DataFrame, cfg: dict, mostrar_correcao: bool = True) -> pd.DataFrame:
    """Rotula cada segmento de curva com o critério de correção tardia."""
    da = cfg.get('correcao_tardia', {})

    partes = []
    for _, dt in dfs_curves.groupby('id_route', sort=False):
        resultado = rotular_correcao_tardia(
            dt,
            limiar_desaceleracao=da.get('limiar_desaceleracao', 2.0),
            margem_histerese=da.get('margem_histerese', 0.0),
        )
        if not resultado.empty:
            partes.append(resultado)

    df_analysis = pd.concat(partes, ignore_index=True)

    if mostrar_correcao:
        n_pos = int(df_analysis['correcao_tardia'].sum())
        print(f"  Pontos em curva com correção tardia: {n_pos}")
        n_indef = int(df_analysis['correcao_indefinida'].sum())
        if n_indef:
            print(f"  Indefinidos (histerese): {n_indef} pontos na faixa do limiar")

    return df_analysis
