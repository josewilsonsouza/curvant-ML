"""Geração das amostras de predição antecipada de risco em curva.

Cada amostra liga uma janela de risco de ``janela_risco_m`` dentro de uma curva
a uma janela de observação de ``janela_observacao_m`` que termina
``horizonte`` segundos antes do início do risco. As grandezas são levadas a uma
grade espacial de passo fixo, sempre dentro de um segmento contínuo da viagem;
um valor só é interpolado entre leituras reais separadas por até
``lacuna_max_variavel_s``.

A severidade da janela de risco é a excedência média de aceleração lateral::

    a_i = v_i² · |κ_i|
    ae  = Σ (a_i − a_lim) · Δs · [a_i ≥ a_lim] / L,   a_lim = (inclinação + μ) · g

Janelas com aceleração lateral acima de ``aceleracao_lateral_max_ms2`` não são
fisicamente alcançáveis por um automóvel e são descartadas.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, rotulo_horizonte
from .trajetoria import segmentar


@dataclass
class ConjuntoAmostras:
    """Metadados por amostra, sequências de observação e nomes das variáveis."""

    amostras: pd.DataFrame
    sequencias: np.ndarray
    variaveis: list[str]
    descartes: dict[str, int]


def carregar_base_curvas(cfg: Config, base: str) -> pd.DataFrame:
    """Base de curvas no formato comum, para a base ``original`` ou ``corrigida``."""
    if base == "corrigida":
        return pd.read_parquet(cfg.pasta_base(base) / "base_curvas.parquet")

    bruta = pd.read_parquet(cfg.pasta_base(base) / "base_curvas.parquet")
    tabela = bruta.rename(columns={"dataset": "conjunto", "time_sec": "tempo_s", "raio_curvatura": "raio_m"})
    viagem = bruta["id_route"].str.split("__OBD__").str[1].str.replace("__", "/", regex=False)
    tabela["id_viagem"] = tabela["conjunto"] + "/" + viagem
    tabela["trecho"] = np.where(bruta["curva"].astype(bool), bruta["trecho_curvo"], 0).astype(int)
    tabela["curva"] = tabela["trecho"] > 0
    tabela = tabela.sort_values(["id_viagem", "tempo_s"], kind="stable").reset_index(drop=True)
    tabela["segmento"] = tabela.groupby("id_viagem", sort=False)["tempo_s"].transform(
        lambda t: segmentar(t.to_numpy(), cfg["trajetoria"]["lacuna_max_s"])
    )
    return tabela


def _interpolar(t_alvo: np.ndarray, tempos: np.ndarray, valores: np.ndarray, lacuna_max: float) -> np.ndarray:
    finito = np.isfinite(valores)
    if finito.sum() < 2:
        return np.full(len(t_alvo), np.nan)
    t, v = tempos[finito], valores[finito]
    saida = np.interp(t_alvo, t, v, left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(t, t_alvo), 1, len(t) - 1)
    saida[(t[i] - t[i - 1]) > lacuna_max] = np.nan
    return saida


def _grade(segmento: pd.DataFrame, colunas: list[str], passo: float, lacuna_max: float) -> pd.DataFrame:
    unicos = segmento.groupby("prog_m")["tempo_s"].mean()
    grade = np.arange(unicos.index.min(), unicos.index.max() + passo, passo)
    tempos_grade = np.interp(grade, unicos.index.to_numpy(), unicos.to_numpy())
    tempos = segmento["tempo_s"].to_numpy(float)
    saida = {"prog_m": grade, "tempo_s": tempos_grade}
    for coluna in colunas:
        saida[coluna] = _interpolar(tempos_grade, tempos, segmento[coluna].to_numpy(float), lacuna_max)
    return pd.DataFrame(saida)


def _curvas(viagem: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ac = cfg["amostras"]
    curvas = (
        viagem[viagem["trecho"] > 0]
        .groupby(["segmento", "trecho"], as_index=False)
        .agg(inicio_m=("prog_m", "min"), fim_m=("prog_m", "max"), curvatura_mediana=("curvatura", "median"))
    )
    curvas = curvas[curvas["fim_m"] - curvas["inicio_m"] >= ac["janela_risco_m"]]
    if curvas.empty:
        return curvas
    limiar = curvas["curvatura_mediana"].abs().quantile(ac["quantil_curvas"])
    return curvas[curvas["curvatura_mediana"].abs() >= limiar]


def gerar(base_curvas: pd.DataFrame, variaveis: list[str], horizonte: float, cfg: Config) -> ConjuntoAmostras:
    """Gera as amostras de um horizonte a partir da base de curvas."""
    ac, col = cfg["amostras"], cfg["colunas"]
    passo = ac["passo_grade_m"]
    n_risco = int(round(ac["janela_risco_m"] / passo))
    n_obs = int(round(ac["janela_observacao_m"] / passo))
    a_lim = (ac["inclinacao_transversal"] + ac["mu"]) * ac["g"]
    velocidade = col["velocidade"]
    colunas_grade = list(dict.fromkeys([velocidade, "curvatura", "raio_m", *variaveis]))

    linhas, sequencias, descartes = [], [], Counter()
    for id_viagem, viagem in base_curvas.groupby("id_viagem", sort=False):
        curvas = _curvas(viagem, cfg)
        ocupado: dict[int, list[tuple[float, float]]] = {}
        for segmento, curvas_seg in curvas.groupby("segmento"):
            seg = viagem[viagem["segmento"] == segmento]
            grade = _grade(seg, colunas_grade, passo, ac["lacuna_max_variavel_s"])
            dist, tempo = grade["prog_m"].to_numpy(), grade["tempo_s"].to_numpy()
            v_ms = grade[velocidade].to_numpy() / 3.6
            a_lat = v_ms**2 * np.abs(grade["curvatura"].to_numpy())
            matriz = grade[variaveis].to_numpy(float)
            tempos_brutos = seg["tempo_s"].to_numpy(float)
            intervalos = ocupado.setdefault(segmento, [])

            for curva in curvas_seg.itertuples(index=False):
                for inicio in np.arange(curva.inicio_m, curva.fim_m - ac["janela_risco_m"] + 0.01, ac["passo_risco_m"]):
                    r0 = int(np.searchsorted(dist, inicio))
                    r1 = r0 + n_risco
                    if r1 > len(grade):
                        continue
                    t_risco = tempo[r0]
                    o1 = int(np.searchsorted(tempo, t_risco - horizonte, side="right"))
                    o0 = o1 - n_obs
                    if o0 < 0:
                        descartes["observação antes do início do segmento"] += 1
                        continue
                    horizonte_real = t_risco - tempo[o1 - 1]
                    if horizonte_real - horizonte > ac["erro_horizonte_max_s"]:
                        descartes["horizonte fora da tolerância"] += 1
                        continue
                    if dist[o1 - 1] + passo > inicio + 1e-6:
                        descartes["observação alcança a janela de risco"] += 1
                        continue
                    obs_ini, obs_fim = dist[o0], dist[o1 - 1] + passo
                    if any(obs_ini < f - 1e-6 and obs_fim > i + 1e-6 for i, f in intervalos):
                        descartes["observação sobreposta"] += 1
                        continue

                    janela_a = a_lat[r0:r1]
                    if not np.isfinite(janela_a).all():
                        descartes["janela de risco sem velocidade ou curvatura"] += 1
                        continue
                    raio_risco = float(np.nanmean(grade["raio_m"].to_numpy()[r0:r1]))
                    if not raio_risco >= ac["raio_min_m"]:
                        descartes["raio abaixo do mínimo"] += 1
                        continue
                    if janela_a.max() > ac["aceleracao_lateral_max_ms2"]:
                        descartes["aceleração lateral fisicamente implausível"] += 1
                        continue
                    obs = matriz[o0:o1]
                    if ac["exigir_todas_variaveis"] and np.isnan(obs).all(axis=0).any():
                        descartes["variável sem leitura na observação"] += 1
                        continue

                    excesso = janela_a >= a_lim
                    ae = float(np.sum((janela_a - a_lim) * passo * excesso) / ac["janela_risco_m"])
                    t_obs = (tempo[o0], tempo[o1 - 1])
                    v_fim = grade[velocidade].to_numpy()[o0:o1]
                    intervalos.append((obs_ini, obs_fim))
                    linhas.append({
                        "id_viagem": id_viagem,
                        "conjunto": viagem["conjunto"].iloc[0],
                        "segmento": int(segmento),
                        "trecho": int(curva.trecho),
                        "horizonte_s": horizonte,
                        "inicio_risco_m": float(inicio),
                        "inicio_obs_m": float(obs_ini),
                        "fim_obs_m": float(obs_fim),
                        "tempo_inicio_obs_s": float(t_obs[0]),
                        "tempo_fim_obs_s": float(t_obs[1]),
                        "tempo_inicio_risco_s": float(t_risco),
                        "horizonte_real_s": float(horizonte_real),
                        "leituras_reais_obs": int(np.sum((tempos_brutos >= t_obs[0]) & (tempos_brutos <= t_obs[1]))),
                        "raio_risco_m": raio_risco,
                        "v_fim_obs_kmh": float(v_fim[np.isfinite(v_fim)][-1]) if np.isfinite(v_fim).any() else np.nan,
                        "a_lat_max_ms2": float(janela_a.max()),
                        "ae": ae,
                    })
                    sequencias.append(obs)

    amostras = pd.DataFrame(linhas)
    seq = np.asarray(sequencias, dtype=np.float32).reshape(len(sequencias), n_obs, len(variaveis))
    return ConjuntoAmostras(amostras, seq, variaveis, dict(descartes))


def salvar(conjunto: ConjuntoAmostras, cfg: Config, base: str, horizonte: float) -> None:
    pasta = cfg.pasta_base(base) / "amostras"
    pasta.mkdir(parents=True, exist_ok=True)
    nome = rotulo_horizonte(horizonte)
    conjunto.amostras.to_parquet(pasta / f"{nome}.parquet", index=False)
    np.savez_compressed(pasta / f"{nome}_sequencias.npz", X=conjunto.sequencias, variaveis=np.array(conjunto.variaveis))
    pd.Series(conjunto.descartes, name="amostras").rename_axis("motivo").to_csv(pasta / f"{nome}_descartes.csv")


def carregar(cfg: Config, base: str, horizonte: float) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    pasta = cfg.pasta_base(base) / "amostras"
    nome = rotulo_horizonte(horizonte)
    dados = np.load(pasta / f"{nome}_sequencias.npz")
    return pd.read_parquet(pasta / f"{nome}.parquet"), dados["X"], [str(v) for v in dados["variaveis"]]
