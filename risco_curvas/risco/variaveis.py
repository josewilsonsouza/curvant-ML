"""Variáveis de entrada por cenário de dispositivo e montagem das matrizes dos modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAIZ, Config

CENARIOS = ("OBD", "SW", "OBD_SW")
GEOMETRIA = ["raio_risco_m"]


def excluidas(cfg: Config) -> dict[str, str]:
    """Variável excluída e o motivo."""
    return {v: motivo for motivo, lista in cfg["variaveis"]["excluir"].items() for v in lista}


def cenarios(cfg: Config) -> dict[str, list[str]]:
    """Listas finais por cenário: a seleção validada sem as variáveis excluídas."""
    lista = pd.read_csv(RAIZ / cfg["variaveis"]["lista_base"]).sort_values(["cenario", "posicao"])
    fora = excluidas(cfg)
    return {c: [v for v in lista.loc[lista["cenario"] == c, "variavel"] if v not in fora] for c in CENARIOS}


def todas(cfg: Config) -> list[str]:
    """União das variáveis de todos os cenários, sem repetição."""
    return list(dict.fromkeys(v for lista in cenarios(cfg).values() for v in lista))


def tabular(sequencias: np.ndarray, nomes: list[str], selecao: list[str], geometria: pd.DataFrame, estatisticas: list[str]) -> pd.DataFrame:
    """Resume cada variável da janela de observação pelas estatísticas pedidas."""
    funcoes = {"mean": np.nanmean, "std": np.nanstd, "min": np.nanmin, "max": np.nanmax}
    blocos = {}
    with np.errstate(all="ignore"):
        for nome in selecao:
            serie = sequencias[:, :, nomes.index(nome)]
            for est in estatisticas:
                blocos[f"{nome}__{est}"] = funcoes[est](serie, axis=1)
    tabela = pd.DataFrame(blocos)
    return pd.concat([tabela, geometria[GEOMETRIA].reset_index(drop=True)], axis=1)


def sequencial(sequencias: np.ndarray, nomes: list[str], selecao: list[str], geometria: pd.DataFrame) -> np.ndarray:
    """Sequência das variáveis selecionadas com a geometria repetida como canal constante."""
    sensores = sequencias[:, :, [nomes.index(n) for n in selecao]]
    geo = np.repeat(geometria[GEOMETRIA].to_numpy(float)[:, None, :], sensores.shape[1], axis=1)
    return np.concatenate([sensores, geo], axis=2).astype(np.float32)
