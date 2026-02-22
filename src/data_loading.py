from huggingface_hub import login, list_repo_files, hf_hub_download
import os
import pandas as pd

DIR_HF = 'jwsouza13/routes_ML_inmetro'

COLS_ORIG = [
    'Time (sec)', ' Latitude (deg)', ' Longitude (deg)',
    ' Vehicle speed (km/h)', ' Accel X (m/s²)', ' Accel Y (m/s²)',
    ' Accel Z (m/s²)', ' Engine RPM (RPM)', ' Mass air flow rate (g/s)',
    ' Absolute throttle position (%)', ' Fuel level input (%)',
    ' GPS Speed (km/h)', ' Fuel rate (l/hr)',
    ' Alcohol fuel percentage (%)', ' Engine fuel rate (l/hr)',
    ' Vehicle Fuel Rate (g/s)', ' Fuel Remaining (l)',
    ' Fuel Level Input A (%)', ' Fuel Level Input B (%)',
    ' Instant fuel economy (l/100 km)', ' Calculated load value (%)',
    ' Fuel rail pressure (kPa)',
    ' Fuel rail pressure relative to manifold vacuum (kPa)',
    ' Accelerator pedal position D (%)',
    ' Accelerator pedal position E (%)',
]

COLS_RJDF = [
    'time', 'lat', 'lon', 'vehicle_speed', 'accel_x', 'accel_y', 'accel_z',
    'engine_rpm', 'mass_air_flow', 'absolute_throttle_pos', 'fuel_level',
    'gps_speed', 'fuel_rate', 'alcohol_fuel_percentage', 'engine_fuel_rate',
    'vehicle_fuel_rate', 'fuel_remaining', 'fuel_level_a', 'fuel_level_b',
    'instant_fuel_economy', 'calculated_load_value', 'fuel_rail_pressure_vacuum',
    'fuel_rail_pressure', 'accelerator_pedal_pos_d', 'accelerator_pedal_pos_e',
]

# Mapeamento de renomeação compartilhado por RJ-DF e SERRA
COL_RENAME = dict(zip(COLS_ORIG, COLS_RJDF))


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'routes_ML_inmetro')


def hf_login(token: str) -> None:
    """Autentica no Hugging Face Hub."""
    login(token=token)


# ── Carregamento local (routes_ML_inmetro/) ───────────────────────────────────

def _local_csvs(folder: str, data_dir: str = DATA_DIR) -> list[str]:
    """Lista os CSVs de uma pasta local DATASET-<folder>/."""
    path = os.path.join(data_dir, f'DATASET-{folder}')
    return sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')
    )


def load_eletronuclear_local(data_dir: str = DATA_DIR) -> list[pd.DataFrame]:
    """Carrega ELETRONUCLEAR da pasta local (ignora arquivos com coluna pos_lat)."""
    dfs = []
    for file in _local_csvs('ELETRONUCLEAR', data_dir):
        filename = os.path.basename(file)
        route_id = filename.replace('.csv', '')
        try:
            _, _, vehicle, *_ = filename.split('-')
        except ValueError:
            vehicle = 'unknown'
        df = pd.read_csv(file, sep=';')
        df = df.assign(id_route=route_id, vehicle=vehicle, loc_coleta='eletronuclear')
        if 'pos_lat' not in df.columns:
            dfs.append(df)
    return dfs


def load_rjdf_local(data_dir: str = DATA_DIR) -> list[pd.DataFrame]:
    """Carrega RJ-DF da pasta local com renomeação padronizada de colunas."""
    dfs = []
    for file in _local_csvs('RJ-DF', data_dir):
        filename = os.path.basename(file)
        df = pd.read_csv(file, skiprows=2)
        if df.columns.to_list() != COLS_ORIG:
            df = pd.read_csv(file, skiprows=1)
        df = df.rename(columns=COL_RENAME)
        df = df.assign(
            id_route=filename.replace('.csv', ''),
            vehicle='nivus',
            loc_coleta='rjdf',
        )
        dfs.append(df)
    return dfs


def load_serra_local(data_dir: str = DATA_DIR) -> list[pd.DataFrame]:
    """Carrega SERRA da pasta local com renomeação padronizada de colunas."""
    dfs = []
    for file in _local_csvs('SERRA', data_dir):
        filename = os.path.basename(file)
        df = pd.read_csv(file, skiprows=1)
        df = df.rename(columns=COL_RENAME)
        df = df.assign(
            id_route=filename.replace('.csv', ''),
            vehicle='jetta',
            loc_coleta='serra',
        )
        dfs.append(df)
    return dfs


def load_folder_files(folder: str, dir_hf: str = DIR_HF) -> list[str]:
    """
    Baixa todos os arquivos CSV de uma pasta específica no dataset do Hugging Face.
    folder: ELETRONUCLEAR, SERRA ou RJ-DF
    """
    all_files = list_repo_files(repo_id=dir_hf, repo_type='dataset', token=True)
    folder_files = [
        f for f in all_files
        if f.startswith(f'DATASET-{folder}/') and f.endswith('.csv')
    ]
    return [
        hf_hub_download(repo_id=dir_hf, filename=f, repo_type='dataset', token=True)
        for f in folder_files
    ]


def load_eletronuclear(dir_hf: str = DIR_HF) -> list[pd.DataFrame]:
    """Carrega os arquivos do dataset ELETRONUCLEAR (ignora arquivos com coluna pos_lat)."""
    dfs = []
    for file in load_folder_files('ELETRONUCLEAR', dir_hf):
        filename = os.path.basename(file)          # funciona em Windows e Linux
        route_id = filename.replace('.csv', '')
        try:
            _, _, vehicle, *_ = filename.split('-')  # obd-15-spin-... → vehicle='spin'
        except ValueError:
            vehicle = 'unknown'
        df = pd.read_csv(file, sep=';')
        df = df.assign(id_route=route_id, vehicle=vehicle, loc_coleta='eletronuclear')
        if 'pos_lat' not in df.columns:
            dfs.append(df)
    return dfs


def load_rjdf(dir_hf: str = DIR_HF) -> list[pd.DataFrame]:
    """Carrega os arquivos do dataset RJ-DF com renomeação padronizada de colunas."""
    dfs = []
    for file in load_folder_files('RJ-DF', dir_hf):
        filename = os.path.basename(file)
        df = pd.read_csv(file, skiprows=2)
        if df.columns.to_list() != COLS_ORIG:
            df = pd.read_csv(file, skiprows=1)
        df = df.rename(columns=COL_RENAME)
        df = df.assign(
            id_route=filename.replace('.csv', ''),
            vehicle='nivus',
            loc_coleta='rjdf',
        )
        dfs.append(df)
    return dfs


def load_serra(dir_hf: str = DIR_HF) -> list[pd.DataFrame]:
    """Carrega os arquivos do dataset SERRA com renomeação padronizada de colunas."""
    dfs = []
    for file in load_folder_files('SERRA', dir_hf):
        filename = os.path.basename(file)
        df = pd.read_csv(file, skiprows=1)
        df = df.rename(columns=COL_RENAME)
        df = df.assign(
            id_route=filename.replace('.csv', ''),
            vehicle='jetta',
            loc_coleta='serra',
        )
        dfs.append(df)
    return dfs
