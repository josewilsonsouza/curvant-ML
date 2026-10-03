"""Tratamento de valores inválidos e atípicos da base pareada.

Valores fora da faixa física de cada grandeza, saltos instantâneos de velocidade
e de posição e picos isolados nos sinais de sensores físicos viram ausentes.
Sinais comandados pelo motorista, como pedal e acelerador, não passam pelo
filtro de picos, porque neles um pulso curto é uma ação real. Os ausentes são
tratados depois, na interpolação sobre a grade espacial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .geo import haversine, segundos


def picos_hampel(valores: np.ndarray, janela: int, limiar: float) -> np.ndarray:
    """Marca picos isolados pelo filtro de Hampel.

    Um ponto é pico quando se afasta da mediana móvel centrada em mais de
    ``limiar`` vezes a escala robusta. A escala é o maior valor entre o MAD
    local e o MAD da série inteira, o que evita marcar degraus e pequenas
    oscilações de sinais quantizados.
    """
    serie = pd.Series(valores, dtype=float)
    if serie.notna().sum() < janela:
        return np.zeros(len(serie), dtype=bool)
    minimo = janela // 2 + 1
    mediana = serie.rolling(janela, center=True, min_periods=minimo).median()
    mad_local = (serie - mediana).abs().rolling(janela, center=True, min_periods=minimo).median()
    mad_global = np.nanmedian(np.abs(serie - np.nanmedian(serie)))
    if not mad_global > 0:
        mad_global = np.nanstd(serie)
    escala = 1.4826 * np.maximum(mad_local.to_numpy(), mad_global)
    with np.errstate(invalid="ignore"):
        return (np.abs(serie - mediana).to_numpy() > limiar * escala) & (escala > 0)


def _picos_de_variacao(valores: np.ndarray, tempos: np.ndarray, limite_por_s: float) -> np.ndarray:
    dt = np.diff(tempos)
    taxa = np.diff(valores) / np.where(dt > 0, dt, np.nan)
    entrada = np.r_[np.nan, taxa]
    saida = np.r_[taxa, np.nan]
    with np.errstate(invalid="ignore"):
        return (np.abs(entrada) > limite_por_s) & (np.abs(saida) > limite_por_s) & (np.sign(entrada) != np.sign(saida))


def _saltos_gps(lat: np.ndarray, lon: np.ndarray, tempos: np.ndarray, vmax: float) -> np.ndarray:
    valido = np.isfinite(lat) & np.isfinite(lon)
    marcas = np.zeros(len(lat), dtype=bool)
    idx = np.flatnonzero(valido)
    if len(idx) < 3:
        return marcas
    d = haversine(lat[idx[:-1]], lon[idx[:-1]], lat[idx[1:]], lon[idx[1:]])
    dt = np.diff(tempos[idx])
    with np.errstate(invalid="ignore", divide="ignore"):
        v = d / np.where(dt > 0, dt, np.nan)
    entrada, saida = np.r_[0.0, v], np.r_[v, 0.0]
    marcas[idx] = (entrada > vmax) & (saida > vmax)
    return marcas


def limpar(base: pd.DataFrame, colunas_sensor: list[str], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica as regras de limpeza por viagem e devolve a base e a contagem por regra."""
    lc, col = cfg["limpeza"], cfg["colunas"]
    base = base.copy()
    registro: list[dict] = []

    def anotar(regra: str, coluna: str, mascara: np.ndarray) -> None:
        registro.append({"regra": regra, "coluna": coluna, "valores": int(mascara.sum())})

    for coluna, (minimo, maximo) in lc["limites"].items():
        if coluna in base:
            fora = ~base[coluna].between(minimo, maximo) & base[coluna].notna()
            base.loc[fora, coluna] = np.nan
            anotar("fora da faixa física", coluna, fora.to_numpy())

    hampel = lc["hampel"]
    for _, linhas in base.groupby("id_viagem", sort=False).groups.items():
        viagem = base.loc[linhas]
        tempos = segundos(viagem["timestamp"])

        v = viagem[col["velocidade"]].to_numpy(float)
        marca = _picos_de_variacao(v, tempos, lc["variacao_max_velocidade_kmh_s"])
        base.loc[linhas[marca], col["velocidade"]] = np.nan
        anotar("salto instantâneo de velocidade", col["velocidade"], marca)

        lat, lon = viagem[col["latitude"]].to_numpy(float), viagem[col["longitude"]].to_numpy(float)
        marca = _saltos_gps(lat, lon, tempos, lc["velocidade_max_gps_ms"])
        base.loc[linhas[marca], [col["latitude"], col["longitude"]]] = np.nan
        anotar("salto de posição GPS", col["latitude"], marca)

        for coluna in colunas_sensor:
            if coluna in viagem and any(t in coluna for t in hampel["sinais"]):
                marca = picos_hampel(viagem[coluna].to_numpy(float), hampel["janela"], hampel["limiar"])
                base.loc[linhas[marca], coluna] = np.nan
                anotar("pico isolado (Hampel)", coluna, marca)

    resumo = pd.DataFrame(registro).groupby(["regra", "coluna"], as_index=False)["valores"].sum()
    resumo["fracao"] = resumo["valores"] / len(base)
    return base, resumo.sort_values(["regra", "valores"], ascending=[True, False]).reset_index(drop=True)
