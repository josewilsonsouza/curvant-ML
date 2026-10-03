"""Linha de comando: ``python -m risco <etapa> [opções]``.

Etapas, na ordem em que dependem umas das outras:

    base       pareia OBD e Sensor Logger a partir dos brutos, limpa e detecta curvas
    amostras   gera as amostras de cada horizonte
    avaliar    validação cruzada por viagem de referências e modelos
    relatorio  consolida resultados e compara as bases
    tudo       executa todas as etapas
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from . import amostras, avaliacao, limpeza, pareamento, relatorio, trajetoria, variaveis
from .config import BASES, Config, carregar, rotulo_horizonte

log = logging.getLogger("risco")
MODELOS = ["RF", "XGB", "LGB", "LSTM"]


def etapa_base(cfg: Config) -> None:
    pasta = cfg.pasta_base("corrigida")
    pasta.mkdir(parents=True, exist_ok=True)
    pareada, resumo = pareamento.parear(cfg)
    resumo.to_csv(pasta / "pareamento.csv", index=False)
    pareada.to_parquet(pasta / "base_pareada.parquet", index=False)
    log.info("pareamento: %d viagens aceitas de %d", resumo["motivo_exclusao"].isna().sum(), len(resumo))

    nomes = variaveis.todas(cfg)
    limpa, contagem = limpeza.limpar(pareada, nomes, cfg)
    contagem.to_csv(pasta / "limpeza.csv", index=False)
    curvas = trajetoria.detectar(limpa, nomes, cfg)
    curvas.to_parquet(pasta / "base_curvas.parquet", index=False)
    log.info("base de curvas: %d linhas, %d viagens", len(curvas), curvas["id_viagem"].nunique())


def etapa_amostras(cfg: Config, bases: list[str], horizontes: list[float]) -> None:
    nomes = variaveis.todas(cfg)
    for base in bases:
        curvas = amostras.carregar_base_curvas(cfg, base)
        for h in horizontes:
            conjunto = amostras.gerar(curvas, nomes, h, cfg)
            amostras.salvar(conjunto, cfg, base, h)
            log.info("amostras %s %s: %d", base, rotulo_horizonte(h), len(conjunto.amostras))


def etapa_avaliar(cfg: Config, bases: list[str], horizontes: list[float], modelos: list[str]) -> None:
    for base in bases:
        for h in horizontes:
            pasta = cfg.pasta_resultados(base) / rotulo_horizonte(h)
            pasta.mkdir(parents=True, exist_ok=True)
            saidas = avaliacao.avaliar(cfg, base, h, modelos)
            _mesclar(saidas["predicoes"], pasta / "predicoes.parquet")
            _mesclar(saidas["metricas_por_dobra"], pasta / "metricas_por_dobra.csv")
            saidas["cortes_por_dobra"].to_csv(pasta / "cortes_por_dobra.csv", index=False)
            log.info("avaliação %s %s concluída", base, rotulo_horizonte(h))


def _mesclar(novo: pd.DataFrame, destino) -> None:
    """Grava os resultados novos, mantendo os de modelos avaliados em outra execução."""
    if destino.exists():
        anterior = pd.read_parquet(destino) if destino.suffix == ".parquet" else pd.read_csv(destino)
        novo = pd.concat([anterior[~anterior["modelo"].isin(novo["modelo"].unique())], novo], ignore_index=True)
    if destino.suffix == ".parquet":
        novo.to_parquet(destino, index=False)
    else:
        novo.to_csv(destino, index=False)


def etapa_relatorio(cfg: Config, bases: list[str]) -> None:
    for base in bases:
        relatorio.resumir(cfg, base)
    relatorio.comparar_bases(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m risco", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etapa", choices=["base", "amostras", "avaliar", "relatorio", "tudo"])
    parser.add_argument("--base", choices=[*BASES, "ambas"], default="ambas")
    parser.add_argument("--horizontes", type=float, nargs="+")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, default=MODELOS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    pd.options.mode.copy_on_write = True
    cfg = carregar()
    bases = list(BASES) if args.base == "ambas" else [args.base]
    horizontes = args.horizontes or cfg["amostras"]["horizontes_s"]

    inicio = time.time()
    if args.etapa in ("base", "tudo"):
        etapa_base(cfg)
    if args.etapa in ("amostras", "tudo"):
        etapa_amostras(cfg, bases, horizontes)
    if args.etapa in ("avaliar", "tudo"):
        etapa_avaliar(cfg, bases, horizontes, args.modelos)
    if args.etapa in ("relatorio", "tudo"):
        etapa_relatorio(cfg, bases)
    log.info("concluído em %.1f min", (time.time() - inicio) / 60)


if __name__ == "__main__":
    main()
