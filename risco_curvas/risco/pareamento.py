"""Pareamento entre os registros OBD e as gravações do Sensor Logger.

Uma gravação é aceita para uma viagem quando, nos instantes em que o veículo
está em movimento, a posição do GPS do celular fica a no máximo
``distancia_max_m`` da posição registrada pelo OBD. Com mais de uma gravação
aceita no mesmo instante, vale a de menor distância mediana. Os sensores são
amostrados no instante exato de cada leitura OBD.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .fontes.obd import ViagemOBD, ler_obd, listar_viagens
from .fontes.sensor_logger import ler_gravacao, ler_localizacao, listar_gravacoes
from .geo import haversine, segundos


@dataclass
class Candidata:
    caminho: Path
    distancia_m: float
    pontos: int
    cobertura: np.ndarray


def amostrar_sensor(
    sensor: pd.DataFrame,
    instantes: np.ndarray,
    angulares: tuple[str, ...] = (),
    janela_s: float = 1.0,
    lacuna_max_s: float = 3.0,
) -> pd.DataFrame:
    """Valores de um sensor nos instantes pedidos (segundos desde a época).

    Sensores esparsos, como o GPS, são interpolados linearmente entre leituras
    separadas por até ``lacuna_max_s``. Sensores densos recebem a média numa
    janela centrada de largura ``janela_s``. Colunas angulares são combinadas
    pelo seno e cosseno; ``bearing`` está em graus e as demais em radianos.
    """
    tempos = segundos(sensor["timestamp"])
    colunas = [c for c in sensor.columns if c != "timestamp"]
    if len(tempos) < 2:
        return pd.DataFrame(np.nan, index=range(len(instantes)), columns=colunas)

    esparso = np.median(np.diff(tempos)) > janela_s / 2
    if esparso:
        i = np.clip(np.searchsorted(tempos, instantes), 1, len(tempos) - 1)
        valido = (instantes >= tempos[0]) & (instantes <= tempos[-1]) & (tempos[i] - tempos[i - 1] <= lacuna_max_s)
        combinar = lambda v: np.interp(instantes, tempos, v)
    else:
        lo = np.searchsorted(tempos, instantes - janela_s / 2)
        hi = np.searchsorted(tempos, instantes + janela_s / 2, side="right")
        valido = hi > lo

        def combinar(v):
            finito = np.isfinite(v)
            soma = np.concatenate([[0.0], np.cumsum(np.where(finito, v, 0.0))])
            conta = np.concatenate([[0], np.cumsum(finito)])
            n = conta[hi] - conta[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(n > 0, (soma[hi] - soma[lo]) / n, np.nan)

    saida = {}
    for coluna in colunas:
        valores = sensor[coluna].to_numpy(dtype=float)
        if coluna in angulares:
            fator = np.pi / 180 if coluna == "bearing" else 1.0
            angulo = np.arctan2(combinar(np.sin(valores * fator)), combinar(np.cos(valores * fator))) / fator
            saida[coluna] = np.mod(angulo, 360) if coluna == "bearing" else angulo
        else:
            saida[coluna] = combinar(valores)
    resultado = pd.DataFrame(saida)
    resultado.loc[~valido, :] = np.nan
    return resultado


def _avaliar_candidatas(obd: pd.DataFrame, gravacoes: list[Path], cfg: Config) -> list[Candidata]:
    p, col = cfg["pareamento"], cfg["colunas"]
    instantes = segundos(obd["timestamp"])
    lat, lon = obd[col["latitude"]].to_numpy(float), obd[col["longitude"]].to_numpy(float)
    movendo = (obd[col["velocidade"]].to_numpy(float) >= p["velocidade_min_kmh"]) & np.isfinite(lat)

    candidatas = []
    for caminho in gravacoes:
        gps = ler_localizacao(caminho, cfg["fuso_horario"])
        if gps.empty:
            continue
        t_gps = segundos(gps["timestamp"])
        if t_gps[-1] < instantes[0] or t_gps[0] > instantes[-1]:
            continue
        pos = amostrar_sensor(gps[["timestamp", "latitude", "longitude"]], instantes, lacuna_max_s=p["lacuna_location_max_s"])
        distancia = haversine(lat, lon, pos["latitude"], pos["longitude"])
        uso = movendo & np.isfinite(distancia)
        mediana = float(np.median(distancia[uso])) if uso.sum() >= p["pontos_min"] else np.nan
        candidatas.append(Candidata(caminho, mediana, int(uso.sum()), pos["latitude"].notna().to_numpy()))
    return candidatas


def parear_viagem(viagem: ViagemOBD, obd: pd.DataFrame, gravacoes: list[Path], cfg: Config) -> tuple[pd.DataFrame | None, dict]:
    """Combina uma viagem OBD com as gravações aceitas e devolve a tabela pareada e o resumo."""
    p, sl = cfg["pareamento"], cfg["sensor_logger"]
    brutos = cfg.caminho("brutos")
    candidatas = _avaliar_candidatas(obd, gravacoes, cfg)
    aceitas = sorted((c for c in candidatas if c.distancia_m <= p["distancia_max_m"]), key=lambda c: c.distancia_m)

    resumo = {
        "id_viagem": viagem.id_viagem,
        "conjunto": viagem.conjunto,
        "linhas_obd": len(obd),
        "candidatas": "; ".join(f"{c.caminho.name} ({c.distancia_m:.1f} m)" for c in candidatas),
        "aceitas": "; ".join(c.caminho.name for c in aceitas),
    }
    if not aceitas:
        resumo["motivo_exclusao"] = "sem gravação sobreposta" if not candidatas else "posições do OBD e do celular não coincidem"
        return None, resumo

    dono = np.full(len(obd), -1)
    for k, c in enumerate(aceitas):
        dono[(dono < 0) & c.cobertura] = k

    instantes = segundos(obd["timestamp"])
    blocos = []
    for k, c in enumerate(aceitas):
        linhas = dono == k
        if not linhas.any():
            continue
        bloco = obd.loc[linhas].reset_index(drop=True)
        sensores = ler_gravacao(c.caminho, tuple(sl["sensores"]), cfg["fuso_horario"])
        partes = [bloco]
        for nome, tabela in sensores.items():
            valores = amostrar_sensor(tabela, instantes[linhas], tuple(sl["angulares"]), sl["janela_media_s"], p["lacuna_location_max_s"])
            partes.append(valores.add_prefix(f"sw_{nome}_"))
        bloco = pd.concat(partes, axis=1)
        bloco["gravacao"] = c.caminho.relative_to(brutos).as_posix()
        blocos.append(bloco)

    pareada = pd.concat(blocos, ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)
    duracao = (pareada["timestamp"].iloc[-1] - pareada["timestamp"].iloc[0]).total_seconds()
    resumo.update(
        linhas_pareadas=len(pareada),
        cobertura=len(pareada) / len(obd),
        distancia_mediana_m=min(c.distancia_m for c in aceitas),
        duracao_s=duracao,
    )
    if duracao < p["duracao_min_s"]:
        resumo["motivo_exclusao"] = "trecho pareado curto demais"
        return None, resumo

    pareada.insert(0, "veiculo", viagem.veiculo)
    pareada.insert(0, "id_viagem", viagem.id_viagem)
    pareada.insert(0, "conjunto", viagem.conjunto)
    pareada.insert(4, "tempo_s", (pareada["timestamp"] - pareada["timestamp"].iloc[0]).dt.total_seconds())
    return pareada, resumo


def parear(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pareia todas as viagens dos conjuntos configurados."""
    brutos = cfg.caminho("brutos")
    gravacoes = listar_gravacoes(brutos, cfg["conjuntos"])
    tabelas, resumos = [], []
    for viagem in listar_viagens(brutos, cfg["conjuntos"]):
        obd = ler_obd(viagem.caminho)
        if obd is None:
            resumos.append({"id_viagem": viagem.id_viagem, "conjunto": viagem.conjunto, "motivo_exclusao": "registro sem horário absoluto e sem GPS"})
            continue
        doconjunto = [g for g in gravacoes if g.parts[-3] == viagem.conjunto]
        tabela, resumo = parear_viagem(viagem, obd, doconjunto, cfg)
        resumos.append(resumo)
        if tabela is not None:
            tabelas.append(tabela)
    return pd.concat(tabelas, ignore_index=True), pd.DataFrame(resumos)
