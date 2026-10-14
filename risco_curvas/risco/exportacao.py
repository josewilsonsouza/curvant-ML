"""Exportação da base pareada corrigida no formato da base pareada original.

O arquivo gerado pode substituir ``base_bruta_obd_smartwatch_pareada_*.parquet``
nos notebooks que consomem a base original: mantém os nomes de colunas, os
identificadores de viagem e os atalhos (``lat``, ``vehicle_speed``,
``gps_speed`` etc.) esperados por eles. Os valores já passaram pela limpeza.
"""

from __future__ import annotations

import pandas as pd

from .config import Config

_PREFIXO_CONJUNTO = {"janeiro": "DATASET-JANEIRO", "mg_ba": "DATASET-MG-BA", "rj_mg": "DATASET-RJ-MG"}

_ATALHOS = {
    "lat": "obd_latitude_deg",
    "lon": "obd_longitude_deg",
    "vehicle_speed": "obd_velocidade_do_veiculo_km_h",
    "gps_speed": "obd_velocidade_do_gps_km_h",
    "engine_rpm": "obd_rpm_do_motor_rpm",
    "accel_x": "obd_aceleracao_x_m_s2",
    "accel_y": "obd_acelere_y_m_s2",
    "accel_z": "obd_aceleracao_z_m_s2",
    "fuel_rate": "obd_taxa_de_combustivel_l_hr",
    "time": "obd_time_sec",
}


def formato_original(limpa: pd.DataFrame) -> pd.DataFrame:
    """Converte a base pareada e limpa para o esquema da base pareada original."""
    caminho_relativo = limpa["id_viagem"].str.split("/", n=1).str[1]
    prefixo = limpa["conjunto"].map(_PREFIXO_CONJUNTO)

    saida = pd.DataFrame({
        "dataset": limpa["conjunto"],
        "id_route": prefixo + "__OBD__" + caminho_relativo.str.replace("/", "__", regex=False),
        "timestamp_segundo": limpa["timestamp"].dt.floor("s"),
        "timestamp": limpa["timestamp"],
        "vehicle": limpa["veiculo"],
        "source_file": prefixo + "/OBD/" + caminho_relativo + ".csv",
    })
    for atalho, coluna in _ATALHOS.items():
        saida[atalho] = limpa[coluna] if coluna in limpa else pd.NA
    saida["time_sec"] = limpa["obd_time_sec"]
    saida["smartwatch_source_file"] = limpa["gravacao"]
    saida["pareamento_tipo"] = "instante_exato_validado_por_gps"
    saida["tem_obd"] = True
    saida["tem_smartwatch"] = True

    sensores = [c for c in limpa.columns if c.startswith(("obd_", "sw_"))]
    return pd.concat([saida, limpa[sensores]], axis=1)


def exportar(limpa: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Grava ``base_bruta_obd_smartwatch_pareada_corrigida.parquet`` na pasta da base corrigida."""
    tabela = formato_original(limpa)
    tabela.to_parquet(cfg.pasta_base("corrigida") / "base_bruta_obd_smartwatch_pareada_corrigida.parquet", index=False)
    return tabela
