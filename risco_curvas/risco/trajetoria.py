"""Segmentação das viagens e detecção geométrica das curvas.

A viagem é dividida em segmentos contínuos sempre que o intervalo entre duas
leituras passa de ``lacuna_max_s``; nenhuma grandeza é calculada através de
um intervalo sem dados. Dentro de cada segmento, a curvatura sai das derivadas
numéricas das posições projetadas, suavizada por filtro gaussiano. São pontos
de curva os que ficam acima do percentil ``percentil_curva`` da curvatura da
viagem, e os trechos curvos são sequências desses pontos com o mesmo sentido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .config import Config
from .geo import projetar_xy, segundos


def segmentar(tempos: np.ndarray, lacuna_max_s: float) -> np.ndarray:
    """Identificador de segmento contínuo para cada leitura."""
    return np.r_[0, np.cumsum(np.diff(tempos) > lacuna_max_s)]


def curvatura(x: np.ndarray, y: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Curvatura suavizada (1/m) e sentido (+1 esquerda, -1 direita) ao longo dos pontos."""
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    cruzado = dx * ddy - dy * ddx
    denominador = (dx**2 + dy**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(denominador > 0, np.abs(cruzado) / denominador, 0.0)
    k = gaussian_filter1d(np.nan_to_num(k, nan=0.0, posinf=0.0), sigma)
    return k, np.sign(gaussian_filter1d(cruzado, sigma))


def sentido_estavel(sinais: np.ndarray, minimo: int) -> np.ndarray:
    """Absorve inversões de sentido mais curtas que ``minimo`` pontos no sentido anterior."""
    saida = sinais.copy()
    inicio = 0
    for i in range(1, len(saida) + 1):
        if i == len(saida) or saida[i] != saida[inicio]:
            if i - inicio < minimo and inicio > 0:
                saida[inicio:i] = saida[inicio - 1]
            inicio = i
    return saida


def _trechos(em_curva: np.ndarray, sinais: np.ndarray, proximo_id: int, minimo_sentido: int) -> tuple[np.ndarray, int]:
    sinais = sinais.astype(int).copy()
    if em_curva.any():
        sinais[em_curva] = sentido_estavel(sinais[em_curva], minimo_sentido)
    ids = np.zeros(len(em_curva), dtype=int)
    anterior, sentido = False, 0
    for i, curva in enumerate(em_curva):
        if not curva:
            proximo_id += anterior
            anterior = False
            continue
        if anterior and sinais[i] != sentido:
            proximo_id += 1
        ids[i], anterior, sentido = proximo_id, True, sinais[i]
    return ids, proximo_id + anterior


def _viagem(viagem: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    tc, col = cfg["trajetoria"], cfg["colunas"]
    viagem = viagem.dropna(subset=[col["latitude"], col["longitude"], "timestamp"]).sort_values("timestamp", kind="stable")
    if len(viagem) < tc["pontos_min_segmento"]:
        return None
    viagem = viagem.reset_index(drop=True)
    x, y = projetar_xy(viagem[col["latitude"]], viagem[col["longitude"]])
    viagem["x"], viagem["y"] = x, y
    viagem["segmento"] = segmentar(segundos(viagem["timestamp"]), tc["lacuna_max_s"])
    tamanho = viagem.groupby("segmento")["segmento"].transform("size")
    viagem = viagem[tamanho >= tc["pontos_min_segmento"]].reset_index(drop=True)
    if viagem.empty:
        return None

    viagem["prog_m"] = np.nan
    viagem["curvatura"] = np.nan
    viagem["sinal_curvatura"] = 0.0
    viagem["_nova_posicao"] = False
    for _, linhas in viagem.groupby("segmento").groups.items():
        seg = viagem.loc[linhas]
        passo = np.hypot(np.diff(seg["x"]), np.diff(seg["y"]))
        viagem.loc[linhas, "prog_m"] = np.r_[0.0, np.cumsum(passo)]
        nova_posicao = np.r_[True, passo > 0]
        k, sinal = curvatura(seg["x"].to_numpy()[nova_posicao], seg["y"].to_numpy()[nova_posicao], tc["sigma_curvatura"])
        mapa = np.cumsum(nova_posicao) - 1
        viagem.loc[linhas, "curvatura"] = k[mapa]
        viagem.loc[linhas, "sinal_curvatura"] = sinal[mapa]
        viagem.loc[linhas, "_nova_posicao"] = nova_posicao

    limiar = np.percentile(viagem.loc[viagem["_nova_posicao"].astype(bool), "curvatura"], tc["percentil_curva"])
    with np.errstate(divide="ignore"):
        viagem["raio_m"] = 1.0 / viagem["curvatura"]
    viagem["trecho"] = 0
    proximo = 1
    for _, linhas in viagem.groupby("segmento").groups.items():
        em_curva = (viagem.loc[linhas, "curvatura"] > limiar).to_numpy()
        ids, proximo = _trechos(em_curva, viagem.loc[linhas, "sinal_curvatura"].to_numpy(), proximo, tc["pontos_min_mesmo_sentido"])
        viagem.loc[linhas, "trecho"] = ids

    contagem = viagem.loc[viagem["trecho"] > 0].groupby("trecho")["raio_m"].agg(["size", "median"])
    validos = contagem.index[(contagem["size"] >= tc["pontos_min_trecho"]) & np.isfinite(contagem["median"])]
    viagem.loc[~viagem["trecho"].isin(validos), "trecho"] = 0
    viagem["curva"] = viagem["trecho"] > 0
    viagem["limiar_curvatura"] = limiar
    return viagem.drop(columns="_nova_posicao")


def detectar(base: pd.DataFrame, variaveis: list[str], cfg: Config) -> pd.DataFrame:
    """Segmenta cada viagem e marca curvas e trechos curvos."""
    col = cfg["colunas"]
    manter = ["conjunto", "id_viagem", "veiculo", "gravacao", "timestamp", "tempo_s"]
    manter += list(dict.fromkeys([col["latitude"], col["longitude"], col["velocidade"], *variaveis]))
    partes = [_viagem(v[manter], cfg) for _, v in base.groupby("id_viagem", sort=False)]
    saida = pd.concat([p for p in partes if p is not None], ignore_index=True)
    inicio = ["conjunto", "id_viagem", "veiculo", "gravacao", "timestamp", "tempo_s", "segmento", "x", "y", "prog_m",
              "curvatura", "raio_m", "sinal_curvatura", "curva", "trecho", "limiar_curvatura"]
    return saida[inicio + [c for c in saida.columns if c not in inicio]]
