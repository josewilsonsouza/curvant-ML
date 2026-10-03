"""Leitura da configuração e resolução de caminhos do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RAIZ = Path(__file__).resolve().parents[1]
BASES = ("original", "corrigida")


@dataclass(frozen=True)
class Config:
    """Configuração carregada de ``config.yaml``, com caminhos absolutos."""

    dados: dict[str, Any]

    def __getitem__(self, chave: str) -> Any:
        return self.dados[chave]

    def caminho(self, nome: str) -> Path:
        return RAIZ / self.dados["caminhos"][nome]

    def pasta_base(self, base: str) -> Path:
        if base not in BASES:
            raise ValueError(f"base desconhecida: {base}")
        return self.caminho(base)

    def pasta_resultados(self, base: str) -> Path:
        pasta = self.caminho("resultados") / base
        pasta.mkdir(parents=True, exist_ok=True)
        return pasta


def carregar(caminho: Path | None = None) -> Config:
    """Carrega ``config.yaml`` da raiz do projeto (ou do caminho informado)."""
    caminho = caminho or RAIZ / "config.yaml"
    with open(caminho, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def rotulo_horizonte(horizonte: float) -> str:
    """Nome curto do horizonte para arquivos: 1.5 -> ``h1p5``."""
    return "h" + f"{horizonte:g}".replace(".", "p")
