"""Validação cruzada com dobras agrupadas por viagem.

Todas as amostras de uma viagem ficam na mesma dobra, de modo que o teste só
contém viagens que o modelo não viu. Os cortes de risco são reajustados em cada
dobra com o ``ae`` do treino. As dobras são estratificadas por um rótulo
provisório calculado na base inteira, usado apenas para equilibrar a
distribuição das classes entre dobras.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import amostras as am
from . import modelos, variaveis
from .config import Config
from .rotulos import LimiaresRisco

log = logging.getLogger(__name__)


def metricas(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Métricas globais e por classe."""
    rotulos = list(range(len(classes)))
    precisao, recall, f1, suporte = precision_recall_fscore_support(y, p, labels=rotulos, zero_division=0)
    saida = {
        "acuracia": accuracy_score(y, p),
        "acuracia_balanceada": balanced_accuracy_score(y, p),
        "f1_macro": f1_score(y, p, labels=rotulos, average="macro", zero_division=0),
    }
    for i, nome in enumerate(classes):
        saida |= {f"f1_{nome}": f1[i], f"precisao_{nome}": precisao[i], f"recall_{nome}": recall[i], f"suporte_{nome}": int(suporte[i])}
    return saida


def dobras(amostras: pd.DataFrame, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    v = cfg["validacao"]
    provisorio = LimiaresRisco(cfg["rotulos"]["kmeans_n_init"], v["semente"]).fit_transform(amostras["ae"])
    divisor = StratifiedGroupKFold(n_splits=v["dobras"], shuffle=True, random_state=v["semente"])
    return list(divisor.split(amostras, provisorio, amostras["id_viagem"]))


def avaliar(cfg: Config, base: str, horizonte: float, nomes_modelos: list[str]) -> dict[str, pd.DataFrame]:
    """Treina e avalia referências e modelos em todas as dobras de um horizonte."""
    amostras, X, nomes = am.carregar(cfg, base, horizonte)
    classes = cfg["rotulos"]["classes"]
    semente = cfg["validacao"]["semente"]
    cenarios = variaveis.cenarios(cfg)
    estatisticas = cfg["variaveis"]["estatisticas"]
    tabelas = {c: variaveis.tabular(X, nomes, v, amostras, estatisticas) for c, v in cenarios.items()}
    sequencias = {c: variaveis.sequencial(X, nomes, v, amostras) for c, v in cenarios.items()} if "LSTM" in nomes_modelos else {}
    fisicas = modelos.entradas_fisicas(amostras["v_fim_obs_kmh"], amostras["raio_risco_m"])
    ae = amostras["ae"].to_numpy()
    grupos = amostras["id_viagem"].to_numpy()

    predicoes, resultados, cortes = [], [], []

    def registrar(dobra, cenario, modelo, teste, y, p):
        predicoes.append(pd.DataFrame({"amostra": teste, "dobra": dobra, "cenario": cenario, "modelo": modelo, "y": y, "previsto": p}))
        resultados.append({"dobra": dobra, "cenario": cenario, "modelo": modelo, "n_teste": len(teste)} | metricas(y, p, classes))

    for k, (treino, teste) in enumerate(dobras(amostras, cfg)):
        limiares = LimiaresRisco(cfg["rotulos"]["kmeans_n_init"], semente).fit(ae[treino])
        y = limiares.transform(ae)
        cortes.append({"dobra": k, "corte_baixo_medio": limiares.cortes_ae_[0], "corte_medio_alto": limiares.cortes_ae_[1],
                       "viagens_teste": len(np.unique(grupos[teste]))} |
                      {f"n_{c}_treino": int((y[treino] == i).sum()) for i, c in enumerate(classes)})

        for nome in modelos.REFERENCIAS:
            ref = modelos.referencia(nome, cfg, semente).fit(fisicas[treino], y[treino])
            registrar(k, "nenhum", nome, teste, y[teste], ref.predict(fisicas[teste]))

        for cenario in cenarios:
            for nome in (m for m in nomes_modelos if m in modelos.TABULARES):
                pipe = modelos.tabular(nome, cfg, semente).fit(tabelas[cenario].iloc[treino], y[treino])
                registrar(k, cenario, nome, teste, y[teste], pipe.predict(tabelas[cenario].iloc[teste]))
            if "LSTM" in nomes_modelos:
                interno = GroupShuffleSplit(n_splits=1, test_size=cfg["validacao"]["fracao_validacao"], random_state=semente)
                ajuste, validacao = next(interno.split(treino, groups=grupos[treino]))
                ajuste, validacao = treino[ajuste], treino[validacao]
                seq = sequencias[cenario]
                rede = modelos.ClassificadorLSTM(cfg, semente).fit(seq[ajuste], y[ajuste], seq[validacao], y[validacao])
                registrar(k, cenario, "LSTM", teste, y[teste], rede.predict(seq[teste]))
            log.info("%s %s dobra %d cenário %s concluído", base, horizonte, k, cenario)

    return {
        "predicoes": pd.concat(predicoes, ignore_index=True),
        "metricas_por_dobra": pd.DataFrame(resultados),
        "cortes_por_dobra": pd.DataFrame(cortes),
    }
