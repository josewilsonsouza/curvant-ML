"""Funções geométricas sobre coordenadas GPS."""

from __future__ import annotations

import numpy as np

RAIO_TERRA_M = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distância em metros entre pares de coordenadas em graus."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RAIO_TERRA_M * np.arcsin(np.sqrt(a))


def projetar_xy(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Projeção equiretangular local em metros, centrada na mediana dos pontos."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    lat0, lon0 = np.nanmedian(lat), np.nanmedian(lon)
    return (lon - lon0) * np.cos(lat0) * RAIO_TERRA_M, (lat - lat0) * RAIO_TERRA_M


def segundos(tempos) -> np.ndarray:
    """Converte ``datetime64`` em segundos (float) desde a época."""
    return np.asarray(tempos, dtype="datetime64[ns]").astype("int64") / 1e9
