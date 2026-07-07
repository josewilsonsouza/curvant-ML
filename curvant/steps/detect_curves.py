import pandas as pd
from curvant.driving.curve_detection import detectar_curvas, contar_curvas

def run(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Detecta curvas em cada trajeto e concatena."""
    sigma = cfg['curve_detection']['sigma']
    limite_raio = cfg['curve_detection']['limite_raio']
    sigma_gps = cfg['curve_detection'].get('sigma_gps', 0.0)

    partes = []
    for traj, dt in df.groupby('id_route', sort=False):
        try:
            partes.append(detectar_curvas(dt, sigma=sigma, limite_raio=limite_raio, sigma_gps=sigma_gps))
        except Exception as e:
            print(f"  [AVISO] Erro ao processar {traj}: {e}")

    dfs_curves = pd.concat(partes, ignore_index=True)

    contagem = contar_curvas(dfs_curves)
    print(f"  {sum(contagem.values())} curvas detectadas em {len(contagem)} trajetos")
    return dfs_curves
