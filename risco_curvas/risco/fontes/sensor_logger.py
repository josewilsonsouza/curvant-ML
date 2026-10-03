"""Leitura das gravações do Sensor Logger (celular com relógio pareado).

Cada gravação é um ``.zip`` com um CSV por sensor ou um ``.json`` com todas as
leituras. Os horários vêm em nanossegundos UTC e são convertidos para o horário
local sem fuso, a mesma referência dos registros OBD.
"""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

_IGNORAR = {"seconds_elapsed", "sensor"}


def listar_gravacoes(pasta_brutos: Path, conjuntos: list[str]) -> list[Path]:
    """Lista os arquivos de gravação de cada conjunto."""
    arquivos = []
    for conjunto in conjuntos:
        pasta = pasta_brutos / conjunto / "sensor_logger"
        arquivos += sorted(p for p in pasta.iterdir() if p.suffix in {".zip", ".json"})
    return arquivos


def _para_local(tempo_ns: pd.Series, fuso: str) -> pd.Series:
    utc = pd.to_datetime(pd.to_numeric(tempo_ns).astype("int64"), unit="ns", utc=True)
    return utc.dt.tz_convert(fuso).dt.tz_localize(None)


def _normalizar(tabela: pd.DataFrame, fuso: str) -> pd.DataFrame:
    tabela = tabela.rename(columns=str.lower)
    colunas = [c for c in tabela.columns if c not in _IGNORAR and c != "time"]
    saida = tabela[colunas].apply(pd.to_numeric, errors="coerce")
    saida.insert(0, "timestamp", _para_local(tabela["time"], fuso))
    return saida.dropna(axis=1, how="all").sort_values("timestamp", kind="stable").reset_index(drop=True)


def _ler(caminho: Path, sensores: tuple[str, ...], fuso: str) -> dict[str, pd.DataFrame]:
    saida: dict[str, pd.DataFrame] = {}
    if caminho.suffix == ".zip":
        with zipfile.ZipFile(caminho) as arquivo:
            for nome in arquivo.namelist():
                sensor = Path(nome).stem.lower()
                if sensor in sensores:
                    saida[sensor] = _normalizar(pd.read_csv(arquivo.open(nome)), fuso)
    else:
        with open(caminho, encoding="utf-8") as f:
            registros = pd.DataFrame(json.load(f))
        registros["sensor"] = registros["sensor"].str.lower()
        for sensor, tabela in registros.groupby("sensor"):
            if sensor in sensores:
                saida[sensor] = _normalizar(tabela.dropna(axis=1, how="all"), fuso)
    return saida


@lru_cache(maxsize=4)
def ler_gravacao(caminho: Path, sensores: tuple[str, ...], fuso: str) -> dict[str, pd.DataFrame]:
    """Lê os sensores pedidos de uma gravação, indexados pelo nome em minúsculas."""
    return _ler(caminho, sensores, fuso)


@lru_cache(maxsize=128)
def ler_localizacao(caminho: Path, fuso: str) -> pd.DataFrame:
    """Lê apenas o GPS do celular, usado para validar o pareamento."""
    return _ler(caminho, ("location",), fuso).get("location", pd.DataFrame())
