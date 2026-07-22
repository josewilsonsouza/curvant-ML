"""Regenera o dataset bruto a partir dos CSVs originais do HuggingFace.

Versão limpa do tratamento_dos_dados.py (notebook do Colab), com uma diferença
deliberada: a deduplicação por [lat, lon, vehicle_speed] guardava UMA leitura de
acelerômetro por grupo colapsado, a última, não a maior. Aqui o conjunto de
linhas que sobrevive é o MESMO do dataset antigo (geometria, zigue-zague e
contagens idênticas), mas cada linha ganha `pico_abs_accel` e `pico_accel_y`:
o pico do acelerômetro sobre todas as leituras originais que ela representa.
São esses picos que os critérios de risco usam. Nas coletas cujo logger já era
1 Hz (rjmgba, janeiro, agda, serra) o pico é igual à leitura única e nada muda.

Uso:
    set HF_TOKEN=hf_...   (ou $env:HF_TOKEN='hf_...' no PowerShell)
    python data/regenerar_dataset.py

Saída: data/eletro_rjdf_serra_rjmgba_janeiro_agda_denso.parquet
"""

import math
import os

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO = 'jwsouza13/routes_ML_inmetro'
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'eletro_rjdf_serra_rjmgba_janeiro_agda_denso.parquet')

# chave da deduplicação original: as linhas que sobrevivem são as MESMAS do dataset
# antigo; a diferença é que o pico de acelerômetro do grupo descartado é preservado
CHAVE_DEDUP_ORIGINAL = ['lat', 'lon', 'vehicle_speed']

TOKEN = os.environ.get('HF_TOKEN')
if not TOKEN:
    raise SystemExit('Defina a variável de ambiente HF_TOKEN com um token do HuggingFace.')


def baixar_pasta(pasta: str) -> list[str]:
    todos = list_repo_files(repo_id=REPO, repo_type='dataset', token=TOKEN)
    arquivos = [f for f in todos if f.startswith(f'DATASET-{pasta}/') and f.endswith('.csv')]
    print(f'  DATASET-{pasta}: {len(arquivos)} CSVs')
    return [
        hf_hub_download(repo_id=REPO, filename=f, repo_type='dataset', token=TOKEN)
        for f in arquivos
    ]


# Renomeações por formato de log (idênticas ao tratamento original)

COLS_ORIG_RJDF = ['Time (sec)', ' Latitude (deg)', ' Longitude (deg)',
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
                  ' Accelerator pedal position E (%)']

COLS_RJDF = ['time', 'lat', 'lon', 'vehicle_speed', 'accel_x', 'accel_y', 'accel_z',
             'engine_rpm', 'mass_air_flow', 'absolute_throttle_pos', 'fuel_level',
             'gps_speed', 'fuel_rate', 'alcohol_fuel_percentage', 'engine_fuel_rate',
             'vehicle_fuel_rate', 'fuel_remaining', 'fuel_level_a', 'fuel_level_b',
             'instant_fuel_economy', 'calculated_load_value', 'fuel_rail_pressure_vacuum',
             'fuel_rail_pressure', 'accelerator_pedal_pos_d', 'accelerator_pedal_pos_e']

RENAME_RJDF = dict(zip(COLS_ORIG_RJDF, COLS_RJDF))

COLS_ORIG_RJMGBA = [
    'Time (sec)',
    ' Intake manifold absolute pressure', ' Intake manifold absolute pressure (kPa)', ' Pressão absoluta do coletor de admissão (kPa)',
    ' Engine RPM', ' Engine RPM (RPM)', ' RPM do motor (RPM)',
    ' Vehicle speed', ' Vehicle speed (km/h)', ' Velocidade do veículo (km/h)',
    ' Intake air temperature', ' Intake air temperature (°C)', ' Temperatura do ar de admissão (°C)',
    ' Absolute throttle position', ' Absolute throttle position (%)', ' Posição absoluta do acelerador (%)',
    ' Fuel level input', ' Fuel level input (%)', ' Entrada do nível de combustível (%)',
    ' Fuel/Air commanded equivalence ratio', ' Relação de equivalência comandada por combustível/ar',
    ' Relative throttle position', ' Relative throttle position (%)', ' Posição relativa do acelerador (%)',
    ' Absolute throttle position B', ' Absolute throttle position B (%)', ' Posição absoluta do acelerador B (%)',
    ' Accelerator pedal position D', ' Accelerator pedal position D (%)', ' Posição D do pedal do acelerador (%)',
    ' Accelerator pedal position E', ' Accelerator pedal position E (%)', ' Posição do pedal do acelerador E (%)',
    ' Commanded throttle actuator control', ' Commanded throttle actuator control (%)', ' Controle do atuador do acelerador comandado (%)',
    ' Alcohol fuel percentage', ' Alcohol fuel percentage (%)', ' Porcentagem de álcool combustível (%)',
    ' Instant Fuel Economy', ' Instant Fuel Economy (l/100 km)', ' Economia de combustível instantânea (l/100 km)',
    ' Total Fuel Economy', ' Total Fuel Economy (l/100 km)', ' Economia total de combustível (l/100 km)',
    ' Fuel Rate', ' Fuel Rate (l/hr)', ' Taxa de combustível (l/hr)',
    ' Instant CO2 rate', ' Instant CO2 rate (g/km)', ' Taxa instantânea de CO2 (g/km)',
    ' Total CO2', ' Total CO2 (kg)',
    ' CO2 flow', ' CO2 flow (g/s)', ' Fluxo de CO2 (g/s)',
    ' Trip Distance', ' Trip Distance (km)', ' Distância da viagem (km)',
    ' Trip Fuel', ' Trip Fuel (l)', ' Combustível de viagem (l)',
    ' Trip Fuel Economy', ' Trip Fuel Economy (l/100 km)', ' Economia de combustível de viagem (l/100 km)',
    ' Hard Brake Count', ' Contagem de freios rígidos',
    ' Hard Accel Count', ' Contagem de aceleração difícil',
    ' Latitude (deg)',
    ' Longitude (deg)',
    ' Altitude (m)',
    ' Bearing (deg)', ' Rolamento (deg)',
    ' GPS Speed (km/h)', ' Velocidade do GPS (km/h)',
    ' Horz Accuracy (m)', ' Precisão Horz (m)',
    ' Accel X (m/s)', ' Accel X (m/s²)', ' Aceleração X (m/s²)',
    ' Accel Y (m/s²)', ' Acelere Y (m/s²)',
    ' Accel Z (m/s²)', ' Aceleração Z (m/s²)',
    ' Accel (Grav) X (m/s²)', ' Aceleração (Grav) X (m/s²)',
    ' Accel (Grav) Y (m/s²)', ' Aceleração (Grav) Y (m/s²)',
    ' Accel (Grav) Z (m/s²)', ' Aceleração (Grav) Z (m/s²)',
    ' Rotation Rate X', ' Rotation Rate X (deg/s)', ' Taxa de rotação X (deg/s)',
    ' Rotation Rate Y', ' Rotation Rate Y (deg/s)', ' Taxa de rotação Y (deg/s)',
    ' Rotation Rate Z', ' Rotation Rate Z (deg/s)', ' Taxa de rotação Z (deg/s)',
    ' Magnetometer X (µT)', ' Magnetometer X', ' Magnetômetro X (µT)',
    ' Magnetometer Y (µT)', ' Magnetometer Y', ' Magnetômetro Y (µT)',
    ' Magnetometer Z (µT)', ' Magnetometer Z', ' Magnetômetro Z (µT)',
]

COLS_RJMGBA = [
    'time',
    'intake_manifold_abs_pressure', 'intake_manifold_abs_pressure', 'intake_manifold_abs_pressure',
    'engine_rpm', 'engine_rpm', 'engine_rpm',
    'vehicle_speed', 'vehicle_speed', 'vehicle_speed',
    'intake_air_temp', 'intake_air_temp', 'intake_air_temp',
    'absolute_throttle_pos', 'absolute_throttle_pos', 'absolute_throttle_pos',
    'fuel_level', 'fuel_level', 'fuel_level',
    'fuel_air_equivalence_ratio', 'fuel_air_equivalence_ratio',
    'relative_throttle_pos', 'relative_throttle_pos', 'relative_throttle_pos',
    'absolute_throttle_pos_b', 'absolute_throttle_pos_b', 'absolute_throttle_pos_b',
    'accelerator_pedal_pos_d', 'accelerator_pedal_pos_d', 'accelerator_pedal_pos_d',
    'accelerator_pedal_pos_e', 'accelerator_pedal_pos_e', 'accelerator_pedal_pos_e',
    'commanded_throttle_actuator_control', 'commanded_throttle_actuator_control', 'commanded_throttle_actuator_control',
    'alcohol_fuel_percentage', 'alcohol_fuel_percentage', 'alcohol_fuel_percentage',
    'instant_fuel_economy', 'instant_fuel_economy', 'instant_fuel_economy',
    'total_fuel_economy', 'total_fuel_economy', 'total_fuel_economy',
    'fuel_rate', 'fuel_rate', 'fuel_rate',
    'instant_co2_rate', 'instant_co2_rate', 'instant_co2_rate',
    'total_co2', 'total_co2',
    'co2_flow', 'co2_flow', 'co2_flow',
    'trip_distance', 'trip_distance', 'trip_distance',
    'trip_fuel', 'trip_fuel', 'trip_fuel',
    'trip_fuel_economy', 'trip_fuel_economy', 'trip_fuel_economy',
    'hard_brake_count', 'hard_brake_count',
    'hard_accel_count', 'hard_accel_count',
    'lat',
    'lon',
    'altitude',
    'bearing', 'bearing',
    'gps_speed', 'gps_speed',
    'horz_accuracy', 'horz_accuracy',
    'accel_x', 'accel_x', 'accel_x',
    'accel_y', 'accel_y',
    'accel_z', 'accel_z',
    'accel_grav_x', 'accel_grav_x',
    'accel_grav_y', 'accel_grav_y',
    'accel_grav_z', 'accel_grav_z',
    'rotation_rate_x', 'rotation_rate_x', 'rotation_rate_x',
    'rotation_rate_y', 'rotation_rate_y', 'rotation_rate_y',
    'rotation_rate_z', 'rotation_rate_z', 'rotation_rate_z',
    'magnetometer_x', 'magnetometer_x', 'magnetometer_x',
    'magnetometer_y', 'magnetometer_y', 'magnetometer_y',
    'magnetometer_z', 'magnetometer_z', 'magnetometer_z',
]

ZIP_RJMGBA = list(zip(COLS_ORIG_RJMGBA, COLS_RJMGBA))

COLS_ORIG_AGDA = [
    'Time (sec)', ' Intake manifold absolute pressure (kPa)', ' Engine RPM (RPM)', ' Vehicle speed (km/h)',
    ' Intake air temperature (°C)', ' Absolute throttle position (%)', ' Fuel level input (%)',
    ' Fuel/Air commanded equivalence ratio', ' Relative throttle position (%)', ' Absolute throttle position B (%)',
    ' Accelerator pedal position D (%)', ' Accelerator pedal position E (%)', ' Commanded throttle actuator control (%)',
    ' Alcohol fuel percentage (%)', ' Instant fuel economy (l/100 km)', ' Total fuel economy (l/100 km)',
    ' Fuel rate (l/hr)', ' Instant CO2 rate (g/km)', 'Total CO2 (kg)', ' CO2 flow (g/s)',
    ' Trip Distance (km)', ' Trip Fuel (l)', ' Trip Fuel Economy (l/100 km)', ' Hard Brake Count', ' Hard Accel Count',
    ' Latitude (deg)', ' Longitude (deg)', ' Altitude (m)', 'Bearing (deg)', ' GPS Speed (km/h)',
    ' Horz Accuracy (m)', ' Accel X (m/s²)', ' Accel Y (m/s²)', 'Accel Z (m/s²)', ' Accel (Grav) X (m/s²)',
    ' Accel (Grav) Y (m/s²)', ' Accel (Grav) Z (m/s²)', ' Rotation Rate X (deg/s)', ' Rotation Rate Y (deg/s)',
    ' Rotation Rate Z (deg/s)', ' Magnetometer X (µT)', ' Magnetometer Y (µT)', ' Magnetometer Z (µT)',
]

COLS_AGDA = [
    'time', 'intake_manifold_abs_pressure', 'engine_rpm', 'vehicle_speed',
    'intake_air_temp', 'absolute_throttle_pos', 'fuel_level',
    'fuel_air_equivalence_ratio', 'relative_throttle_pos', 'absolute_throttle_pos_b',
    'accelerator_pedal_pos_d', 'accelerator_pedal_pos_e', 'commanded_throttle_actuator_control',
    'alcohol_fuel_percentage', 'instant_fuel_economy', 'total_fuel_economy',
    'fuel_rate', 'instant_co2_rate', 'total_co2', 'co2_flow',
    'trip_distance', 'trip_fuel', 'trip_fuel_economy', 'hard_brake_count', 'hard_accel_count',
    'lat', 'lon', 'altitude', 'bearing', 'gps_speed',
    'horz_accuracy', 'accel_x', 'accel_y', 'accel_z', 'accel_grav_x',
    'accel_grav_y', 'accel_grav_z', 'rotation_rate_x', 'rotation_rate_y',
    'rotation_rate_z', 'magnetometer_x', 'magnetometer_y', 'magnetometer_z',
]

ZIP_AGDA = list(zip(COLS_ORIG_AGDA, COLS_AGDA))


def _renomear(df: pd.DataFrame, zip_cols: list) -> pd.DataFrame:
    atuais = df.columns.to_list()
    rename = {}
    for orig, novo in zip_cols:
        if orig in atuais and orig not in rename:
            rename[orig] = novo
    df = df.rename(columns=rename)
    return df.loc[:, ~df.columns.duplicated()]


def _aparar_velocidade_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Corta o início e o fim de cada trajeto onde a velocidade ainda é zero."""
    partes = []
    for rota in df['id_route'].unique():
        t = df[df['id_route'] == rota].copy()
        nz = t[t['vehicle_speed'] != 0]
        if nz.empty:
            print(f'  trajeto {rota} sem dados válidos, mantido como está')
            partes.append(t)
        else:
            partes.append(t.loc[nz.index.min():nz.index.max()])
    return pd.concat(partes)


def carregar_eletro() -> pd.DataFrame:
    dfs = []
    for arq in baixar_pasta('ELETRONUCLEAR'):
        df = pd.read_csv(arq, sep=';')
        nome = os.path.basename(arq)
        _, _, veiculo, _, _ = nome.split('-')
        df = df.assign(id_route=nome.replace('.csv', ''), vehicle=veiculo,
                       loc_coleta='eletronuclear')
        if 'pos_lat' not in df.columns:   # logs de ônibus (datalogger), descartados
            dfs.append(df)
    de = pd.concat(dfs)
    de = de.query('id_route not in ["obd-17-van-cicuito-t2", "obd-17-van-circuito-t1"]')
    de = de.query('vehicle_speed != 255')
    return _aparar_velocidade_zero(de)


def carregar_rjdf() -> pd.DataFrame:
    dfs = []
    for arq in baixar_pasta('RJ-DF'):
        df = pd.read_csv(arq, skiprows=2)
        if df.columns.to_list() != COLS_ORIG_RJDF:
            df = pd.read_csv(arq, skiprows=1)
        df = df.rename(columns=RENAME_RJDF)
        df = df.assign(id_route=os.path.basename(arq).replace('.csv', ''),
                       vehicle='nivus', loc_coleta='rjdf')
        dfs.append(df)
    d = pd.concat(dfs)
    d = d.query('id_route != "CSVLog_20240920_092511"')   # 116 pontos, quase tudo parado
    return _aparar_velocidade_zero(d)


def carregar_serra() -> pd.DataFrame:
    dfs = []
    for arq in baixar_pasta('SERRA'):
        df = pd.read_csv(arq, skiprows=1)
        df = df.rename(columns=RENAME_RJDF)
        df = df.assign(id_route=os.path.basename(arq).replace('.csv', ''),
                       vehicle='jetta', loc_coleta='serra')
        dfs.append(df)
    return _aparar_velocidade_zero(pd.concat(dfs))


def _carregar_estilo_rjmgba(pasta: str, veiculo: str, loc: str, zip_cols: list,
                            fallback_ponto_e_virgula: bool = False) -> pd.DataFrame:
    dfs = []
    for arq in baixar_pasta(pasta):
        df = pd.read_csv(arq, skiprows=1)
        if fallback_ponto_e_virgula and len(df.columns) == 1:
            df = pd.read_csv(arq, skiprows=2, sep=';')
        df = _renomear(df, zip_cols)
        df = df.assign(id_route=os.path.basename(arq).replace('.csv', ''),
                       vehicle=veiculo, loc_coleta=loc)
        dfs.append(df)
    return _aparar_velocidade_zero(pd.concat(dfs))


def latlon_para_cartesiano(lats, lons):
    lats_r = [math.radians(v) for v in lats]
    lons_r = [math.radians(v) for v in lons]
    lat0, lon0 = lats_r[0], lons_r[0]
    R = 6371000
    out = []
    for lat, lon in zip(lats_r, lons_r):
        if lat != lat0 or lon != lon0:
            out.append((R * (lon - lon0) * math.cos((lat + lat0) / 2), R * (lat - lat0)))
        else:
            out.append((0, 0))
    return out


def ajustes_finais(df: pd.DataFrame) -> pd.DataFrame | None:
    """Por trajeto: time_sec, dt, remoção de parados, deduplicação e coordenadas x/y."""
    df = df.dropna(subset=['lat', 'lon']).copy()
    if df.empty:
        return None

    if df['loc_coleta'].iloc[0] == 'eletronuclear':
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        df = df.set_index('timestamp').sort_index()
    else:
        df['time_sec'] = df['time']

    df['dt'] = df['time_sec'].diff().fillna(0)
    prev = df['vehicle_speed'].shift(1).fillna(0)
    if 'fuel_rate' in df.columns:
        df.loc[(prev == 0) & (df['fuel_rate'] == 0), 'dt'] = 0

    antes = len(df)
    df = df[~((df['vehicle_speed'] < 1) & (df['accel_x'].abs() < 0.005) & (df['accel_y'].abs() < 0.005))]

    # Mesma deduplicação do tratamento original (o conjunto de linhas fica idêntico,
    # preservando geometria, zigue-zague e contagens), mas antes de descartar as linhas
    # "duplicadas" o pico do acelerômetro delas é preservado em colunas próprias. São
    # esses picos que os critérios de Kamm e lateral usam; a leitura única que sobrevive
    # é a última do grupo, não a maior, e perde mais da metade dos picos reais.
    chave = [c for c in CHAVE_DEDUP_ORIGINAL if c in df.columns]
    df['_abs'] = np.sqrt(df['accel_x'] ** 2 + df['accel_y'] ** 2)
    df['_ay'] = df['accel_y'].abs()
    df['pico_abs_accel'] = df.groupby(chave)['_abs'].transform('max')
    df['pico_accel_y'] = df.groupby(chave)['_ay'].transform('max')
    df = df.drop_duplicates(subset=chave, keep='last')
    df = df.drop(columns=['_abs', '_ay'])
    df.reset_index(drop=True, inplace=True)

    x, y = zip(*latlon_para_cartesiano(df['lat'], df['lon']))
    df = df.assign(x=x, y=y)

    for col in list(df.columns):
        serie = df[col]
        if serie.notnull().sum() == 0 or bool(np.all(serie == 0)):
            df = df.drop(columns=col)

    df = df.dropna(subset=['x', 'y'])
    rota = df['id_route'].iloc[0] if not df.empty else '?'
    print(f'  {rota}: {antes} -> {len(df)} linhas')
    return df if not df.empty else None


def main() -> None:
    print('Baixando e carregando as coletas...')
    bases = [
        carregar_eletro(),
        carregar_rjdf(),
        carregar_serra(),
        _carregar_estilo_rjmgba('RJ-MG-BA', 'Cruze e Jetta', 'rjmgba', ZIP_RJMGBA),
        _carregar_estilo_rjmgba('JANEIRO', 'carro', 'janeiro', ZIP_RJMGBA),
        _carregar_estilo_rjmgba('AGDA', 'carro', 'agda', ZIP_AGDA, fallback_ponto_e_virgula=True),
    ]

    print('\nAjustes finais por trajeto (dedup por', CHAVE_DEDUP_ORIGINAL, ')...')
    partes = []
    for base in bases:
        for rota in base['id_route'].unique():
            res = ajustes_finais(base[base['id_route'] == rota])
            if res is not None:
                partes.append(res)

    df_all = pd.concat(partes)
    df_all.to_parquet(SAIDA, index=False)

    dt = df_all.groupby('id_route')['time_sec'].diff()
    print(f'\nSalvo em {SAIDA}')
    print(f'  {len(df_all):,} linhas | {df_all["id_route"].nunique()} trajetos '
          f'| dt mediano {dt.median():.3f} s')
    print(df_all.groupby('loc_coleta').size().rename('linhas').to_string())


if __name__ == '__main__':
    main()
