"""Rotulagem do nível de risco por KMeans unidimensional em sqrt(ae)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class LimiaresRisco:
    """Três níveis de risco (baixo, médio, alto) a partir de ``ae``.

    O KMeans com k = 3 em ``sqrt(ae)`` é ajustado só com as amostras de treino;
    os centroides ordenados definem os cortes, e uma amostra recebe o nível do
    centroide mais próximo, o que equivale a cortar nos pontos médios.
    """

    def __init__(self, n_init: int = 50, semente: int = 42):
        self.n_init = n_init
        self.semente = semente

    def fit(self, ae: np.ndarray) -> "LimiaresRisco":
        raiz = np.sqrt(np.asarray(ae, dtype=float)).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, n_init=self.n_init, random_state=self.semente).fit(raiz)
        centros = np.sort(kmeans.cluster_centers_.ravel())
        self.centros_ae_ = centros**2
        self.cortes_ae_ = ((centros[:-1] + centros[1:]) / 2) ** 2
        return self

    def transform(self, ae: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.cortes_ae_, np.asarray(ae, dtype=float), side="right")

    def fit_transform(self, ae: np.ndarray) -> np.ndarray:
        return self.fit(ae).transform(ae)
