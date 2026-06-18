import numpy as np
import pandas as pd

_G: float = 9.81  # aceleração gravitacional (m/s²)

# Limiares de classificação baseados em ISL_max da curva
_LIMIARES: list[tuple[float, str]] = [
    (0.5, 'baixo'),   # ISL < 0.5
    (0.8, 'medio'),   # 0.5 ≤ ISL < 0.8
]

_CLASS_MAP: dict[str, int] = {'baixo': 0, 'medio': 1, 'alto': 2}

def classificar_isl(isl: float) -> str:
    """Retorna 'baixo', 'medio' ou 'alto' para um valor de ISL."""
    for limite, classe in _LIMIARES:
        if isl < limite:
            return classe
    return 'alto'


def calcular_isl(df: pd.DataFrame, mu: float = 0.6) -> pd.DataFrame:
    """
    Adiciona a coluna 'isl' ponto a ponto ao DataFrame.

    Requer a coluna 'ctp_accel' (v²/R), em m/s².

    Parâmetros
    ----------
    df  : DataFrame com coluna 'ctp_accel'
    mu  : coeficiente de atrito estático (padrão 0.6 — asfalto seco)
    """
    df = df.copy()
    df['isl'] = df['ctp_accel'].abs() / (_G * mu)
    return df


def calcular_isl_por_curva(
    dfs_curves: pd.DataFrame,
    mu: float = 0.6,
) -> pd.DataFrame:
    """
    Agrega o ISL por trecho de curva (id_route, trecho_curvo).

    Considera apenas os pontos onde ``curva=True``.

    Retorna DataFrame com colunas:
        id_route, id_trecho_curvo, isl_mean, isl_max, isl_p95,
        isl_class, isl_class_num
    """
    df = calcular_isl(dfs_curves, mu=mu)
    curva_pts = df[df['curva'] == True]

    registros = []
    for (id_route, trecho_curvo), grupo in curva_pts.groupby(['id_route', 'trecho_curvo']):
        isl_vals = grupo['isl'].values
        isl_max = float(np.max(isl_vals))
        classe = classificar_isl(isl_max)
        registros.append({
            'id_route':       id_route,
            'id_trecho_curvo': trecho_curvo,
            'isl_mean':       float(np.mean(isl_vals)),
            'isl_max':        isl_max,
            'isl_p95':        float(np.percentile(isl_vals, 95)),
            'isl_class':      classe,
            'isl_class_num':  _CLASS_MAP[classe],
        })

    return pd.DataFrame(registros)


def resumo_isl(df_isl: pd.DataFrame) -> None:
    """Imprime distribuição das classes ISL e estatísticas básicas."""
    contagem = df_isl['isl_class'].value_counts().reindex(['baixo', 'medio', 'alto'], fill_value=0)
    print("Distribuição ISL por curva:")
    for classe, n in contagem.items():
        pct = n / len(df_isl) * 100
        print(f"  {classe:8s}: {n:4d} ({pct:.1f}%)")
    extra = ""
    if 'isl_p95' in df_isl.columns:
        extra = f"  p95: {df_isl['isl_p95'].mean():.3f}"
    print(f"  ISL_max — média: {df_isl['isl_max'].mean():.3f}{extra}  máx: {df_isl['isl_max'].max():.3f}")
