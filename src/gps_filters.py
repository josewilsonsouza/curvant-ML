"""
Filtros de suavização de coordenadas GPS (lat/lon ou Cartesianas x/y).

Todos os filtros trabalham por trajeto separadamente — nenhuma contaminação
entre rotas diferentes concatenadas no mesmo DataFrame.

Uso rápido:
    from src.gps_filters import aplicar_filtro

    df_suave = aplicar_filtro(df, metodo='kalman', cols=['lat', 'lon'], R=1e-5, Q=1e-6)
    df_suave = aplicar_filtro(df, metodo='savgol', window_length=11, polyorder=2)
    df_suave = aplicar_filtro(df, metodo='mediana', janela=5)  # pré-filtro anti-teleport

Recomendação de uso no pipeline:
    1. filtrar_mediana  — remove teleports (saltos impossíveis de GPS)
    2. filtrar_kalman   — suavização principal (usa dt temporal real)
    OU
    1. filtrar_savgol   — boa opção única: preserva bordas, remove ruído branco
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# ── Utilitário interno ────────────────────────────────────────────────────────

def _por_trajeto(df: pd.DataFrame, fn, cols: list[str], **kwargs) -> pd.DataFrame:
    """Aplica fn coluna a coluna em cada trajeto separado."""
    partes = []
    for _, grp in df.groupby('id_route', sort=False):
        grp = grp.copy()
        for col in cols:
            if col in grp.columns:
                grp[col] = fn(grp[col].values.astype(float), **kwargs)
        partes.append(grp)
    return pd.concat(partes, ignore_index=True)


# ── Filtros ───────────────────────────────────────────────────────────────────

def filtrar_mediana(
    df: pd.DataFrame,
    cols: list[str] = ['lat', 'lon'],
    janela: int = 5,
) -> pd.DataFrame:
    """
    Mediana móvel — robusto a outliers e teleports (saltos impossíveis de GPS).

    Recomendado como pré-filtro antes de qualquer suavização contínua.
    Não distorce curvas nítidas, mas remove picos isolados.

    janela : tamanho da janela (ímpar recomendado; menor = menos agressivo)
    """
    def _fn(arr):
        return pd.Series(arr).rolling(janela, center=True, min_periods=1).median().values

    return _por_trajeto(df, _fn, cols)


def filtrar_savgol(
    df: pd.DataFrame,
    cols: list[str] = ['lat', 'lon'],
    window_length: int = 11,
    polyorder: int = 2,
) -> pd.DataFrame:
    """
    Savitzky-Golay — ajusta um polinômio local por janela deslizante.

    Preserva melhor a forma das curvas (picos, bordas) do que médias móveis,
    porque o polinômio local acompanha a geometria real da trajetória.
    Boa opção única quando não há teleports.

    window_length : número de pontos na janela (deve ser ímpar, ≥ polyorder+1)
    polyorder     : grau do polinômio — 2 ou 3 funciona bem para GPS
    """
    def _fn(arr):
        wl = min(window_length, len(arr))
        if wl % 2 == 0:
            wl -= 1
        if wl < polyorder + 2:
            return arr
        return savgol_filter(arr, wl, polyorder)

    return _por_trajeto(df, _fn, cols)


def filtrar_gaussiano(
    df: pd.DataFrame,
    cols: list[str] = ['lat', 'lon'],
    sigma: float = 2.0,
) -> pd.DataFrame:
    """
    Filtro Gaussiano — suavização contínua com pesos em forma de sino.

    Mesmo filtro usado na curvatura dentro de curve_detection.py.
    Simples e eficaz para ruído branco; não é robusto a teleports.

    sigma : desvio padrão em número de pontos — maior = mais suave
    """
    return _por_trajeto(df, gaussian_filter1d, cols, sigma=sigma)


def filtrar_media_movel(
    df: pd.DataFrame,
    cols: list[str] = ['lat', 'lon'],
    janela: int = 5,
) -> pd.DataFrame:
    """
    Média móvel — mais simples, mas atenua e atrasa curvas nítidas.

    Útil para visualização; não recomendado antes do cálculo de curvatura
    porque amortece os ângulos reais das curvas.

    janela : número de pontos na janela
    """
    def _fn(arr):
        return pd.Series(arr).rolling(janela, center=True, min_periods=1).mean().values

    return _por_trajeto(df, _fn, cols)


def filtrar_kalman(
    df: pd.DataFrame,
    cols: list[str] = ['lat', 'lon'],
    R: float = 1e-5,
    Q: float = 1e-6,
) -> pd.DataFrame:
    """
    Filtro de Kalman com modelo de velocidade constante.

    Vantagem sobre os demais: usa o dt real entre pontos (coluna time_sec),
    tratando corretamente a irregularidade temporal do GPS (1–5 Hz variável).

    Modelo de estado: [posição, velocidade]
    Medição:          [posição]

    R : variância do ruído de medição (GPS)
        maior → menos confiança no GPS → trajetória mais suave
        típico para GPS veicular: 1e-5 a 1e-4
    Q : variância do ruído de processo (incerteza na velocidade)
        maior → o filtro acompanha mudanças mais rápidas
        típico: 1e-7 a 1e-5

    Se time_sec não estiver disponível, assume dt=1 entre pontos.
    """
    partes = []
    for _, grp in df.groupby('id_route', sort=False):
        grp = grp.copy().reset_index(drop=True)
        times = grp['time_sec'].values if 'time_sec' in grp.columns else np.arange(len(grp), dtype=float)

        for col in cols:
            if col in grp.columns:
                grp[col] = _kalman_vel_constante(grp[col].values.astype(float), times, R=R, Q=Q)

        partes.append(grp)

    return pd.concat(partes, ignore_index=True)


def _kalman_vel_constante(z: np.ndarray, times: np.ndarray, R: float, Q: float) -> np.ndarray:
    """
    Filtro de Kalman 1D — modelo de velocidade constante com dt variável.

    Estado:  x = [posição, velocidade]
    Transição: F(dt) = [[1, dt], [0, 1]]
    Medição: H = [1, 0]  (só observamos posição)
    """
    n = len(z)
    x = np.array([z[0], 0.0])   # posição inicial, velocidade inicial = 0
    P = np.eye(2)                # covariância inicial
    H = np.array([1.0, 0.0])    # vetor de medição

    resultado = np.empty(n)
    resultado[0] = z[0]

    for i in range(1, n):
        dt = float(times[i] - times[i - 1])
        if dt <= 0:
            dt = 1.0

        # ── Predict ──────────────────────────────────────────────────────────
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([0.5 * dt**2, dt])       # como a aceleração afeta o estado
        Q_mat = Q * np.outer(G, G)

        x = F @ x
        P = F @ P @ F.T + Q_mat

        # ── Update ────────────────────────────────────────────────────────────
        S = float(H @ P @ H) + R              # inovação na variância (escalar)
        K = (P @ H) / S                       # ganho de Kalman (vetor 2D)
        innov = z[i] - float(H @ x)           # inovação (escalar)

        x = x + K * innov
        P = (np.eye(2) - np.outer(K, H)) @ P

        resultado[i] = x[0]

    return resultado


# Dispatcher

_METODOS = {
    'mediana':     filtrar_mediana,
    'savgol':      filtrar_savgol,
    'gaussiano':   filtrar_gaussiano,
    'media_movel': filtrar_media_movel,
    'kalman':      filtrar_kalman,
}


def aplicar_filtro(
    df: pd.DataFrame,
    metodo: str = 'savgol',
    cols: list[str] = ['lat', 'lon'],
    **kwargs,
) -> pd.DataFrame:
    """
    Aplica o filtro escolhido sobre as colunas indicadas.

    metodo    : 'mediana' | 'savgol' | 'gaussiano' | 'media_movel' | 'kalman'
    cols      : colunas a filtrar (padrão: lat e lon; pode incluir 'x', 'y')

    Parâmetros por método:
      mediana      → janela (int, default 5)
      savgol       → window_length (int, ímpar, default 11), polyorder (int, default 2)
      gaussiano    → sigma (float, default 2.0)
      media_movel  → janela (int, default 5)
      kalman       → R (float, default 1e-5), Q (float, default 1e-6)

    Nota: filtros aplicados sobre lat/lon NÃO atualizam automaticamente as colunas
    Cartesianas x/y. Se x/y forem usadas no pipeline (curve_detection), passe
    cols=['lat', 'lon', 'x', 'y'] ou recalcule x/y após filtrar.
    """
    if metodo not in _METODOS:
        raise ValueError(f"Método {metodo!r} desconhecido. Opções: {list(_METODOS)}")

    return _METODOS[metodo](df, cols=cols, **kwargs)
