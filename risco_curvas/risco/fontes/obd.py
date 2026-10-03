"""Leitura dos registros OBD-II exportados pelo aplicativo de diagnóstico."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_FORMATO_INICIO = "%m/%d/%Y %I:%M:%S.%f %p"

_EQUIVALENCIAS = {
    "vehicle_speed": "velocidade_do_veiculo_km_h",
    "vehicle_speed_km_h": "velocidade_do_veiculo_km_h",
    "engine_rpm": "rpm_do_motor_rpm",
    "engine_rpm_rpm": "rpm_do_motor_rpm",
    "gps_speed_km_h": "velocidade_do_gps_km_h",
    "bearing_deg": "rolamento_deg",
    "absolute_throttle_position": "posicao_absoluta_do_acelerador",
    "absolute_throttle_position_b": "posicao_absoluta_do_acelerador_b",
    "relative_throttle_position": "posicao_relativa_do_acelerador",
    "accelerator_pedal_position_d": "posicao_d_do_pedal_do_acelerador",
    "commanded_throttle_actuator_control": "controle_do_atuador_do_acelerador_comandado",
    "fuel_level_input": "entrada_do_nivel_de_combustivel",
    "fuel_air_commanded_equivalence_ratio": "relacao_de_equivalencia_comandada_por_combustivel_ar",
    "instant_co2_rate": "taxa_instantanea_de_co2_g_km",
    "instant_fuel_economy": "economia_de_combustivel_instantanea_l_100_km",
    "accel_x_m_s2": "aceleracao_x_m_s2",
    "accel_y_m_s2": "acelere_y_m_s2",
    "accel_z_m_s2": "aceleracao_z_m_s2",
    "accel_grav_x_m_s2": "aceleracao_grav_x_m_s2",
    "accel_grav_y_m_s2": "aceleracao_grav_y_m_s2",
    "accel_grav_z_m_s2": "aceleracao_grav_z_m_s2",
    "rotation_rate_x": "taxa_de_rotacao_x_deg_s",
    "rotation_rate_y": "taxa_de_rotacao_y_deg_s",
    "rotation_rate_z": "taxa_de_rotacao_z_deg_s",
    "magnetometer_x_t": "magnetometro_x_t",
    "magnetometer_y_t": "magnetometro_y_t",
    "magnetometer_z_t": "magnetometro_z_t",
}

_PE_M = 0.3048
_CONVERSOES = {
    "_ft_s2": ("_m_s2", _PE_M),
    "altitude_ft": ("altitude_m", _PE_M),
    "precisao_horz_ft": ("precisao_horz_m", _PE_M),
    "taxa_instantanea_de_co2_lb_mile": ("taxa_instantanea_de_co2_g_km", 453.59237 / 1.609344),
}


def _padronizar(colunas: pd.Index) -> tuple[list[str], dict[str, float]]:
    nomes, fatores = [], {}
    for original in colunas:
        nome = _EQUIVALENCIAS.get(slug(original), slug(original))
        for sufixo, (novo, fator) in _CONVERSOES.items():
            if nome.endswith(sufixo):
                nome = nome[: -len(sufixo)] + novo
                fatores["obd_" + nome] = fator
        nomes.append("obd_" + nome)
    return nomes, fatores


def slug(texto: str) -> str:
    """Normaliza um cabeçalho para ``snake_case`` ASCII."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")


@dataclass(frozen=True)
class ViagemOBD:
    """Um arquivo OBD e a identificação da viagem que ele registra."""

    id_viagem: str
    conjunto: str
    veiculo: str | None
    caminho: Path


def listar_viagens(pasta_brutos: Path, conjuntos: list[str]) -> list[ViagemOBD]:
    """Lista os arquivos OBD de cada conjunto, em ``<conjunto>/obd/[motorista/veiculo/]*.csv``."""
    viagens = []
    for conjunto in conjuntos:
        raiz = pasta_brutos / conjunto / "obd"
        for caminho in sorted(raiz.rglob("*.csv")):
            relativo = caminho.relative_to(raiz).with_suffix("")
            partes = relativo.parts
            veiculo = partes[-2] if len(partes) >= 3 else None
            viagens.append(ViagemOBD(f"{conjunto}/{relativo.as_posix()}", conjunto, veiculo, caminho))
    return viagens


def ler_obd(caminho: Path) -> pd.DataFrame | None:
    """Lê um registro OBD com horário absoluto, em nomes e unidades padronizados.

    Cabeçalhos em inglês são mapeados para os nomes em português, e grandezas em
    unidades imperiais são convertidas para o SI. Retorna ``None`` para exportações sem a linha ``StartTime``, que não trazem
    horário absoluto nem posição e por isso não podem ser pareadas.
    """
    with open(caminho, encoding="utf-8-sig") as f:
        cabecalho = f.readline()
    if "StartTime" not in cabecalho:
        return None
    inicio = pd.to_datetime(cabecalho.split("=", 1)[1].strip(), format=_FORMATO_INICIO)

    dados = pd.read_csv(caminho, skiprows=1, low_memory=False)
    dados.columns, fatores = _padronizar(dados.columns)
    dados = dados.loc[:, ~dados.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    for coluna, fator in fatores.items():
        dados[coluna] *= fator

    dados.insert(0, "timestamp", inicio + pd.to_timedelta(dados["obd_time_sec"], unit="s"))
    sem_posicao = (dados["obd_latitude_deg"] == 0) & (dados["obd_longitude_deg"] == 0)
    dados.loc[sem_posicao, ["obd_latitude_deg", "obd_longitude_deg"]] = np.nan
    return dados.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
