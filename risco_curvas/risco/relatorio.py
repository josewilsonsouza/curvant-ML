"""Consolidação dos resultados por base e comparação entre as bases."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import amostras as am
from .config import BASES, Config, rotulo_horizonte
from .rotulos import LimiaresRisco

METRICAS = ["f1_alto", "recall_alto", "precisao_alto", "f1_medio", "f1_macro", "acuracia_balanceada"]
REFERENCIA = "Física v²/R"


def resumo_amostras(cfg: Config, base: str) -> pd.DataFrame:
    """Tamanho, viagens e distribuição de classes (cortes na base inteira) por horizonte."""
    linhas = []
    for h in cfg["amostras"]["horizontes_s"]:
        amostras, _, _ = am.carregar(cfg, base, h)
        lim = LimiaresRisco(cfg["rotulos"]["kmeans_n_init"], cfg["validacao"]["semente"]).fit(amostras["ae"])
        y = lim.transform(amostras["ae"])
        linha = {"horizonte_s": h, "amostras": len(amostras), "viagens": amostras["id_viagem"].nunique(),
                 "corte_baixo_medio": lim.cortes_ae_[0], "corte_medio_alto": lim.cortes_ae_[1],
                 "leituras_reais_obs_mediana": amostras["leituras_reais_obs"].median()}
        for i, classe in enumerate(cfg["rotulos"]["classes"]):
            linha[f"n_{classe}"] = int((y == i).sum())
        linha["viagens_com_alto"] = amostras.loc[y == 2, "id_viagem"].nunique()
        linhas.append(linha)
    return pd.DataFrame(linhas)


def resumir(cfg: Config, base: str) -> pd.DataFrame:
    """Média e desvio entre dobras de cada modelo, com o ganho sobre a referência física."""
    pasta = cfg.pasta_resultados(base)
    partes = []
    for h in cfg["amostras"]["horizontes_s"]:
        arquivo = pasta / rotulo_horizonte(h) / "metricas_por_dobra.csv"
        if arquivo.exists():
            partes.append(pd.read_csv(arquivo).assign(horizonte_s=h))
    if not partes:
        return pd.DataFrame()
    dobras = pd.concat(partes, ignore_index=True)
    resumo = dobras.groupby(["horizonte_s", "cenario", "modelo"])[METRICAS].agg(["mean", "std"])
    resumo.columns = [f"{m}_{e}".replace("_mean", "").replace("_std", "_dp") for m, e in resumo.columns]
    resumo = resumo.reset_index()
    ref = resumo.loc[resumo["modelo"] == REFERENCIA, ["horizonte_s", "f1_alto", "f1_macro"]]
    resumo = resumo.merge(ref.rename(columns={"f1_alto": "_ref_alto", "f1_macro": "_ref_macro"}), on="horizonte_s")
    resumo["ganho_f1_alto_sobre_fisica"] = resumo["f1_alto"] - resumo.pop("_ref_alto")
    resumo["ganho_f1_macro_sobre_fisica"] = resumo["f1_macro"] - resumo.pop("_ref_macro")
    resumo.to_csv(pasta / "resumo.csv", index=False)
    resumo_amostras(cfg, base).to_csv(pasta / "amostras.csv", index=False)
    _grafico(resumo, base, pasta / "f1_alto_por_horizonte.png")
    return resumo


def _grafico(resumo: pd.DataFrame, base: str, destino) -> None:
    cenarios = [c for c in ("OBD", "SW", "OBD_SW") if c in set(resumo["cenario"])]
    referencias = resumo[resumo["cenario"] == "nenhum"]
    fig, eixos = plt.subplots(1, len(cenarios), figsize=(4.6 * len(cenarios), 3.8), sharey=True, squeeze=False)
    for eixo, cenario in zip(eixos[0], cenarios):
        for modelo, dados in resumo[resumo["cenario"] == cenario].groupby("modelo"):
            eixo.errorbar(dados["horizonte_s"], dados["f1_alto"], yerr=dados["f1_alto_dp"], marker="o", capsize=3, label=modelo)
        for modelo, dados in referencias.groupby("modelo"):
            eixo.plot(dados["horizonte_s"], dados["f1_alto"], linestyle="--", color="0.3" if modelo == REFERENCIA else "0.7", label=modelo)
        eixo.set_title(cenario)
        eixo.set_xlabel("horizonte (s)")
        eixo.grid(alpha=0.3)
    eixos[0][0].set_ylabel("F1 da classe alto")
    eixos[0][-1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Base {base}: média e desvio entre dobras por viagem")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def comparar_bases(cfg: Config) -> pd.DataFrame:
    """Lado a lado das duas bases para cada horizonte, cenário e modelo."""
    tabelas = []
    for base in BASES:
        arquivo = cfg.pasta_resultados(base) / "resumo.csv"
        if arquivo.exists():
            tabelas.append(pd.read_csv(arquivo)[["horizonte_s", "cenario", "modelo", "f1_alto", "f1_macro", "ganho_f1_alto_sobre_fisica"]].assign(base=base))
    if len(tabelas) < 2:
        return pd.DataFrame()
    comparacao = pd.concat(tabelas).pivot_table(index=["horizonte_s", "cenario", "modelo"], columns="base")
    comparacao.columns = [f"{m}_{b}" for m, b in comparacao.columns]
    comparacao = comparacao.reset_index()
    comparacao.to_csv(cfg.caminho("resultados") / "comparacao_bases.csv", index=False)
    return comparacao
