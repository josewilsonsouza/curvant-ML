import numpy as np
import pandas as pd

from src.isl import classificar_isl

_G         = 9.81
_MU_PADRAO = 0.6

# Encoding numérico da classe DNIT para feature prev_dnit_num
_DNIT_NUM = {'suave': 0, 'aberta': 1, 'media': 2, 'fechada': 3, 'muito_fechada': 4}


def calcular_estatisticas_por_trajeto(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatísticas médias de variáveis-chave agrupadas por trajeto."""
    vars_interesse = {
        'vehicle_speed': 'v',
        'theta_direcao': 'theta',
        'accel_x': 'a_x',
        'accel_y': 'a_y',
        'accel_z': 'a_z',
    }
    rows = []
    for id_route_atual, trajeto in df.groupby('id_route'):
        row = {
            'id_trajeto': id_route_atual,
            'numb_curve': len(trajeto['trecho_curvo'].unique()) - 1,
        }
        for var, alias in vars_interesse.items():
            if var in trajeto.columns:
                row[alias] = round(trajeto[var].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def extrair_features(
    data: pd.DataFrame,
    janela_tempo: int = 15,
    janela_distancia: float | None = None,
    janela_acel_confort: float = 2.5,
    janela_distancia_min: float = 50.0,
    janela_distancia_max: float = 400.0,
) -> pd.DataFrame:
    """
    Extrai features da janela pré-curva e targets de comportamento dentro da curva.

    Janela pré-curva
    ----------------
    Quando ``janela_distancia`` é fornecida, usa essa distância fixa (m).
    Caso contrário, calcula dinamicamente a distância de frenagem confortável
    com base na velocidade de aproximação: d = v² / (2 * a_confort), clipada
    em [janela_distancia_min, janela_distancia_max].

    Features (janela pré-curva)
    ---------------------------
    - mean / std / median / max / min / slope / cv de vehicle_speed, engine_rpm, accel_x, accel_y
    - distance_car_curve, n_perigo_*_janela
    - v_entry  : velocidade no primeiro ponto da curva (km/h)  [P4]
    - janela_raio_min, janela_raio_mean, janela_raio_last : raio de curvatura na janela
                                                            pré-curva (sem leakage)

    Features de contexto (calculadas post-hoc via shift, sem leakage)
    ------------------------------------------------------------------
    - n_curvas_antes, n_perigosas_antes, prop_perigosas_antes
    - prev_raio_min, prev_raio_mean, prev_dnit_num  : geometria da curva anterior  [P4]

    Targets de classificação
    ------------------------
    - manobra              : maioria dos pontos da curva é Perigosa (Li et al.)
    - manobra_velocidade   : 1 se v_entry > v_safe(R) — excesso de velocidade na entrada  [P2]

    Targets de regressão
    --------------------
    - isl_max, isl_mean, isl_alto, isl_class          : ISL dentro da curva
    - curve_accel_y_max, curve_accel_y_mean           : aceleração lateral |accel_y|  [P1]
    - curve_abs_accel_max, curve_abs_accel_mean        : aceleração total               [P1]
    - v_entry_ratio                                    : v_entry / v_safe(R)           [P2]

    Colunas auxiliares (excluídas das features dos modelos)
    -------------------------------------------------------
    - v_safe_dnit, curve_raio_min, curve_raio_mean, curve_dnit_num
    """
    dados_janela = []
    df = data.copy()
    df['conducao'] = df['conducao'].map({'Perigosa': 1, 'Segura': 0})
    df = df.sort_values(by=['id_route', 'time_sec'])

    vars_sensor = ['vehicle_speed', 'engine_rpm', 'accel_x', 'accel_y']

    for (id_route_atual, trecho_curvo), curva in df.groupby(['id_route', 'trecho_curvo']):
        if len(curva) <= 2:
            continue

        # ── Janela pré-curva ─────────────────────────────────────────────────
        dist_entrada = curva['distancia_acumulada'].min()

        if janela_distancia is not None:
            # distância fixa explícita
            d_janela = float(janela_distancia)
        else:
            # distância dinâmica: d = v² / (2 * a_confort), baseada na
            # velocidade média dos últimos pontos antes da curva
            pontos_antes = df[
                (df['id_route'] == id_route_atual)
                & (df['distancia_acumulada'] < dist_entrada)
            ].tail(5)
            v_ms = (
                float(pontos_antes['vehicle_speed'].mean()) / 3.6
                if not pontos_antes.empty else 60.0 / 3.6
            )
            d_janela = float(np.clip(
                v_ms ** 2 / (2.0 * janela_acel_confort),
                janela_distancia_min,
                janela_distancia_max,
            ))

        janela = df[
            (df['id_route'] == id_route_atual)
            & (df['distancia_acumulada'] < dist_entrada)
            & (df['distancia_acumulada'] >= dist_entrada - d_janela)
        ].copy()

        # Melhoria #2 — remove pontos pertencentes a uma curva anterior
        if 'curva' in janela.columns:
            janela = janela[~janela['curva']].copy()

        if janela.empty:
            continue

        row: dict = {
            'time_inicio':    janela['time_sec'].min(),
            'time_fim':       janela['time_sec'].max(),
            'id_route':       id_route_atual,
            'id_trecho_curvo': trecho_curvo,
        }

        t = janela['time_sec'].values - janela['time_sec'].values[0]

        # Metade tardia da janela (últimos 50% dos pontos) para capturar comportamento
        # de aproximação à curva separado do comportamento inicial
        mid = max(len(janela) // 2, 1)
        janela_tarde = janela.iloc[mid:]
        t_tarde = janela_tarde['time_sec'].values - janela_tarde['time_sec'].values[0] if len(janela_tarde) > 0 else np.array([0.0])

        for var in vars_sensor:
            vals = janela[var].values
            row[f'{var}_mean']   = float(np.mean(vals))
            row[f'{var}_std']    = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{var}_median'] = float(np.median(vals))
            row[f'{var}_max']    = float(np.max(vals))
            row[f'{var}_min']    = float(np.min(vals))
            row[f'{var}_slope']  = float(np.polyfit(t, vals, 1)[0]) if len(t) >= 2 else 0.0
            row[f'{var}_cv']     = (
                float(row[f'{var}_std'] / row[f'{var}_mean'])
                if row[f'{var}_mean'] != 0 else 0.0
            )
            # Metade tardia: mean e slope (captura frenagem / aceleração perto da curva)
            if len(janela_tarde) >= 2:
                vals_t = janela_tarde[var].values
                row[f'{var}_mean_tarde']  = float(np.mean(vals_t))
                row[f'{var}_slope_tarde'] = float(np.polyfit(t_tarde, vals_t, 1)[0])
            else:
                row[f'{var}_mean_tarde']  = row[f'{var}_mean']
                row[f'{var}_slope_tarde'] = row[f'{var}_slope']

        # Jerk (taxa de variação da aceleração) — detecta reações bruscas do motorista
        for var in ['accel_x', 'accel_y']:
            vals = janela[var].values
            dt_arr = np.diff(janela['time_sec'].values)
            dt_arr = np.where(dt_arr > 0, dt_arr, 1e-3)
            jerk = np.diff(vals) / dt_arr
            row[f'jerk_{var[-1]}_max'] = float(np.abs(jerk).max()) if len(jerk) > 0 else 0.0
            row[f'jerk_{var[-1]}_std'] = float(np.std(jerk))       if len(jerk) > 1 else 0.0

        row['distance_car_curve']          = float(
            janela['distancia_acumulada'].max() - janela['distancia_acumulada'].min()
        )
        # Velocidade cinemática prevista na entrada da curva assumindo desaceleração
        # de conforto constante ao longo da janela pré-curva.
        # v_pred² = max(0, v_mean² - 2·a·d)  — usa apenas dados da janela, sem leakage.
        _d_window = row['distance_car_curve']
        _v_mean_ms = float(janela['vehicle_speed'].mean()) / 3.6
        _v_pred_sq = max(0.0, _v_mean_ms ** 2 - 2.0 * janela_acel_confort * _d_window)
        row['v_pred_kinematica'] = round(float(np.sqrt(_v_pred_sq) * 3.6), 3)  # km/h

        # Contagem de pontos na janela pré-curva onde cada critério dispara,
        # calculados diretamente dos sensores (independente dos rótulos de
        # caracterização, que agora cobrem apenas os segmentos de curva).
        _kamm_lim = 0.7 * 0.6 * _G
        _accel_total_janela = np.sqrt(janela['accel_x'].values**2 + janela['accel_y'].values**2)
        row['n_perigo_accel_janela']   = int((_accel_total_janela > _kamm_lim).sum())
        row['n_perigo_lateral_janela'] = int((janela['accel_y'].abs() > 2.0).sum())

        # Raio de curvatura na janela pré-curva (F3).
        # A B-spline é ajustada ao trajeto inteiro: a curvatura aqui é influenciada pelos
        # pontos da curva à frente (via spline global + filtro Gaussiano).
        #
        # Modo 1 (rota conhecida): comportamento CORRETO — o trajeto completo está disponível
        # antes da viagem, então usar a geometria futura é legítimo.
        #
        # Modo 2 (rota desconhecida): comportamento INCORRETO — pontos futuros não estariam
        # disponíveis num dispositivo OBD em tempo real. Por isso pipeline.py zera F3 em Modo 2.
        if 'raio_curvatura' in janela.columns:
            raios_janela = janela['raio_curvatura'].replace(0, np.nan).dropna()
            if len(raios_janela) > 0:
                row['janela_raio_min']  = float(raios_janela.min())
                row['janela_raio_mean'] = float(raios_janela.mean())
                # último ponto da janela = mais próximo da curva
                last_valid = janela['raio_curvatura'].replace(0, np.nan).dropna()
                row['janela_raio_last'] = float(last_valid.iloc[-1])
            else:
                row['janela_raio_min'] = row['janela_raio_mean'] = row['janela_raio_last'] = np.nan
        else:
            row['janela_raio_min'] = row['janela_raio_mean'] = row['janela_raio_last'] = np.nan

        # ── Targets de classificação (Li et al.) ──────────────────────────────
        row['manobra_accel_curva']       = 1 if curva['manobra_accel'].any() else 0
        row['manobra_lateral_curva']     = 1 if curva['manobra_lateral'].any() else 0
        row['manobra_ziguezague_curva']  = 1 if curva['manobra_ziguezague'].any() else 0
        row['manobra_combinado_curva']   = 1 if curva['manobra_combinado'].any() else 0
        row['manobra']                   = row['manobra_combinado_curva']  # retrocompat

        # ── Pontos dentro da curva ────────────────────────────────────────────
        pts_curva = curva[curva['curva'] == True] if 'curva' in curva.columns else curva
        if pts_curva.empty:
            pts_curva = curva

        # P2 — velocidade de entrada e velocidade segura para o raio detectado
        curva_ord = pts_curva.sort_values('time_sec')
        v_entry    = float(curva_ord['vehicle_speed'].iloc[0])
        raio_entry = (
            float(curva_ord['raio_curvatura'].iloc[0])
            if 'raio_curvatura' in curva_ord.columns else np.nan
        )
        if not np.isnan(raio_entry) and raio_entry > 0:
            v_safe = float(np.sqrt(raio_entry * _G * _MU_PADRAO) * 3.6)
        else:
            v_safe = np.nan

        row['v_entry']            = v_entry
        row['v_safe_dnit']        = round(v_safe, 2) if not np.isnan(v_safe) else np.nan
        row['v_entry_ratio']      = round(v_entry / v_safe, 3) if (not np.isnan(v_safe) and v_safe > 0) else np.nan

        # Quanto o motorista desacelerou ao se aproximar da curva
        v_max_janela = float(janela['vehicle_speed'].max())
        v_mean_janela = row['vehicle_speed_mean']
        row['v_speed_drop']     = float(v_max_janela - v_entry)
        row['v_speed_drop_pct'] = float(row['v_speed_drop'] / v_max_janela) if v_max_janela > 0 else 0.0
        row['v_entry_vs_mean']  = float(v_entry / v_mean_janela) if v_mean_janela > 0 else 1.0
        # Proxy físico de ISL na entrada: v²/R_estimado (feature F3)
        _raio_est = row.get('janela_raio_min', np.nan)
        if pd.notna(_raio_est) and _raio_est > 0:
            row['v_entry_sq_over_raio_est'] = float((v_entry / 3.6) ** 2 / _raio_est)
        else:
            row['v_entry_sq_over_raio_est'] = np.nan
        row['manobra_velocidade'] = (
            int(v_entry > v_safe) if (not np.isnan(v_safe) and v_safe > 0) else np.nan
        )
        row['v_excess'] = row['manobra_velocidade']  # alias explícito da taxonomia

        # ISL cinemático: ISL = v²/(R·g·μ) = ctp_accel/(g·μ) — depende do raio GPS (B-spline).
        # Ruído no raio pode gerar ISL inflado em trechos retos ou subestimado em curvas reais.
        if 'ctp_accel' in pts_curva.columns:
            isl_vals = pts_curva['ctp_accel'].abs() / (_G * _MU_PADRAO)
            isl_max  = float(isl_vals.max())
            row['isl_mean']  = float(isl_vals.mean())
            row['isl_max']   = isl_max
            row['isl_class'] = classificar_isl(isl_max)
            row['isl_alto']  = 1 if isl_max >= 0.8 else 0
            # Velocidade real no ponto de pico ISL — target fisicamente fundamentado.
            # Prever v_critica equivale a prever ISL: ISL = v_critica²/(R×g×μ).
            idx_max          = isl_vals.idxmax()
            row['v_critica'] = float(pts_curva.loc[idx_max, 'vehicle_speed'])  # km/h
        else:
            row['isl_mean'] = row['isl_max'] = row['isl_class'] = row['isl_alto'] = np.nan
            row['v_critica'] = np.nan

        # ISL baseado no sensor OBD: ISL_sensor = |accel_y| / (g·μ)
        # Independe do raio GPS — usa a aceleração lateral medida diretamente.
        # Alternativa mais confiável para o ISL cinemático em curvas com GPS ruidoso.
        if 'accel_y' in pts_curva.columns:
            isl_s = pts_curva['accel_y'].abs() / (_G * _MU_PADRAO)
            row['isl_sensor_max']   = float(isl_s.max())
            row['isl_sensor_mean']  = float(isl_s.mean())
            row['isl_sensor_class'] = classificar_isl(float(isl_s.max()))
        else:
            row['isl_sensor_max'] = row['isl_sensor_mean'] = row['isl_sensor_class'] = np.nan

        # P1 — aceleração lateral e total dentro da curva (targets de regressão sem assumir μ)
        if 'accel_y' in pts_curva.columns:
            row['curve_accel_y_max']  = float(pts_curva['accel_y'].abs().max())
            row['curve_accel_y_mean'] = float(pts_curva['accel_y'].abs().mean())
        else:
            row['curve_accel_y_max'] = row['curve_accel_y_mean'] = np.nan

        if 'abs_accel' in pts_curva.columns:
            row['curve_abs_accel_max']  = float(pts_curva['abs_accel'].max())
            row['curve_abs_accel_mean'] = float(pts_curva['abs_accel'].mean())
        else:
            row['curve_abs_accel_max'] = row['curve_abs_accel_mean'] = np.nan

        # P4 — geometria desta curva (armazenada para uso como contexto da curva anterior)
        if 'raio_curvatura' in pts_curva.columns:
            row['curve_raio_min']  = float(pts_curva['raio_curvatura'].min())
            row['curve_raio_mean'] = float(pts_curva['raio_curvatura'].mean())
        else:
            row['curve_raio_min'] = row['curve_raio_mean'] = np.nan

        if 'classe_dnit' in pts_curva.columns:
            dnit_mode = pts_curva['classe_dnit'].mode()
            row['curve_dnit_num'] = _DNIT_NUM.get(
                dnit_mode.iloc[0] if not dnit_mode.empty else 'suave', 0
            )
        else:
            row['curve_dnit_num'] = 0

        # F4 — geometria real da curva seguinte.
        # Cópias de curve_raio_* com nome distinto para não conflitar com _COLS_EXCLUIR.
        row['f4_raio_min']  = row.get('curve_raio_min', np.nan)
        row['f4_raio_mean'] = row.get('curve_raio_mean', np.nan)
        row['f4_dnit_num']  = row.get('curve_dnit_num', 0)

        # ISL estimado na entrada: usa raio real + velocidade de entrada
        # isl = v²/(R × g × μ), μ=0.6 — mesmo cálculo do target isl_max
        # Em Modo 2 f4_raio_min=0, então isl_entry=0 (sem informação geométrica)
        _r = row.get('f4_raio_min', np.nan)
        if pd.notna(_r) and _r > 0:
            row['isl_entry_estimate'] = float((v_entry / 3.6) ** 2 / (_r * 9.81 * 0.6))
        else:
            row['isl_entry_estimate'] = 0.0

        dados_janela.append(row)

    df_out = (
        pd.DataFrame(dados_janela)
        .sort_values(['id_route', 'time_inicio'])
        .reset_index(drop=True)
    )

    # ── Contexto do trajeto (calculado via shift — sem leakage) ───────────────
    partes = []
    raio_min_global  = df_out['curve_raio_min'].median()
    raio_mean_global = df_out['curve_raio_mean'].median()

    for _, grupo in df_out.groupby('id_route', sort=False):
        grupo = grupo.copy()

        grupo['n_curvas_antes']    = np.arange(len(grupo))
        # n_perigosas_antes usa os labels do próprio pipeline (shift garante ausência de leakage
        # temporal), mas propaga ruído: um falso positivo precoce no trajeto inflaciona esta
        # feature para todas as curvas seguintes da mesma rota.
        grupo['n_perigosas_antes']    = grupo['manobra'].shift(1).fillna(0).cumsum().astype(int)
        grupo['prop_perigosas_antes'] = (
            grupo['n_perigosas_antes'] / grupo['n_curvas_antes'].replace(0, np.nan)
        ).fillna(0.0).round(3)

        # P4 — raio e classe DNIT da curva anterior (shift garante ausência de leakage)
        grupo['prev_raio_min']  = grupo['curve_raio_min'].shift(1).fillna(raio_min_global)
        grupo['prev_raio_mean'] = grupo['curve_raio_mean'].shift(1).fillna(raio_mean_global)
        grupo['prev_dnit_num']  = grupo['curve_dnit_num'].shift(1).fillna(0).astype(int)

        partes.append(grupo)

    return pd.concat(partes, ignore_index=True)
