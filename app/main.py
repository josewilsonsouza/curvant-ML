"""
CurvantML - Visualizador Interativo de Curvas
Uso: streamlit run app/main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import contextlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import altair as alt

from PIL import Image as _Image
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="CurvantML",
    layout="wide",
    page_icon=_Image.open(os.path.join(_ROOT, "images", "curvantML.png")),
)

_CACHE_DIR = "data/.cache_app"
_META_FILE = f"{_CACHE_DIR}/meta.json"
_FEATURES_FILE = f"{_CACHE_DIR}/features_df.parquet"
_DFAT_FILE = f"{_CACHE_DIR}/df_at.parquet"

_CANDIDATOS = [
    "data/eletro_rjdf_serra_rjmgba_janeiro_clean.parquet",
    "data/eletro_rjdf_serra_clean.parquet",
    "data/eletro_rjdf_serra_rjmgba_janeiro.parquet",
    "data/eletro_rjdf_serra.parquet",
]


def _cache_key(data_path: str) -> str:
    stat = os.stat(data_path)
    return f"{data_path}|{stat.st_size}|{stat.st_mtime}"


def _cache_valido(data_path: str) -> bool:
    if not all(os.path.exists(p) for p in [_META_FILE, _FEATURES_FILE, _DFAT_FILE]):
        return False
    with open(_META_FILE) as f:
        meta = json.load(f)
    return meta.get("key") == _cache_key(data_path)


def _salvar_cache(features_df: pd.DataFrame, df_at: pd.DataFrame, data_path: str) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    features_df.to_parquet(_FEATURES_FILE, index=False)
    df_at.to_parquet(_DFAT_FILE, index=False)
    with open(_META_FILE, "w") as f:
        json.dump({"key": _cache_key(data_path)}, f)


def _executar_pipeline(data_path: str, cfg: dict, feat_cfg: dict):
    from curvant.steps import detect_curves, label_driving
    from curvant.driving.curve_detection import identificar_trechos_curvos
    from curvant.driving.features import extrair_features

    df = pd.read_parquet(data_path)

    with contextlib.redirect_stdout(io.StringIO()):
        dfs_curves = detect_curves.run(df, cfg)

    raio_min = cfg["curve_detection"]["raio_min"]
    raio_clip = dfs_curves["raio_curvatura"].clip(lower=raio_min)
    dfs_curves["ctp_accel"] = (dfs_curves["vehicle_speed"] / 3.6) ** 2 / raio_clip

    with contextlib.redirect_stdout(io.StringIO()):
        df_analysis = label_driving.run(dfs_curves, cfg, mostrar_risco=False)

    df_analysis[["manobra_accel", "manobra_lateral", "manobra_ziguezague"]] = (
        df_analysis[["manobra_accel", "manobra_lateral", "manobra_ziguezague"]].astype(int)
    )

    with contextlib.redirect_stdout(io.StringIO()):
        df_at = identificar_trechos_curvos(df_analysis)

    ex = feat_cfg["extracao"]
    features_df = extrair_features(
        df_at,
        janela_distancia=ex.get("janela_distancia"),
        janela_acel_confort=ex.get("janela_acel_confort", 2.5),
        janela_distancia_min=ex.get("janela_distancia_min", 10.0),
        janela_distancia_max=ex.get("janela_distancia_max", 200.0),
        lead_gap=ex.get("lead_gap", 0.0),
        vars_sensor=ex.get("vars_sensor"),
    )
    return features_df, df_at


@st.cache_data(show_spinner=False)
def carregar_pipeline(rebuild: bool = False):
    from curvant.utils.config import carregar_config, carregar_features_config
    cfg = carregar_config()
    feat_cfg = carregar_features_config()
    data_path = next(p for p in _CANDIDATOS if os.path.exists(p))

    if not rebuild and _cache_valido(data_path):
        features_df = pd.read_parquet(_FEATURES_FILE)
        df_at = pd.read_parquet(_DFAT_FILE)
    else:
        with st.spinner("Rodando pipeline - primeira execução (~2 min)..."):
            features_df, df_at = _executar_pipeline(data_path, cfg, feat_cfg)
        _salvar_cache(features_df, df_at, data_path)

    return features_df, df_at, cfg, feat_cfg


def bearing_wrap(b2: float, b1: float) -> float:
    return (b2 - b1 + 180) % 360 - 180


def enriquecer_com_bearings(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona bearing e variação de bearing (wrap-corrected) ao DataFrame."""
    from curvant.driving.risk_measures import calcular_bearing
    df = df.copy().reset_index(drop=True)
    lats, lons = df["lat"].tolist(), df["lon"].tolist()
    bearings = [np.nan] + [
        calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ]
    df["bearing"] = bearings
    df["delta_bearing"] = [np.nan] + [
        abs(bearing_wrap(bearings[i], bearings[i - 1]))
        for i in range(1, len(bearings))
    ]
    df["var_vel_acum"] = df["vehicle_speed"].diff().abs().cumsum().fillna(0)
    return df


def cor_manobra(label: str) -> str:
    return "#d73027" if label == "Perigosa" else "#2b8a3e"


# Dados

features_df, df_at, cfg, feat_cfg = carregar_pipeline(rebuild=False)
loc_map = df_at[["id_route", "loc_coleta"]].drop_duplicates().set_index("id_route")["loc_coleta"]
features_df["_local"] = features_df["id_route"].map(loc_map)

# Sidebar - filtros

st.sidebar.image(os.path.join(_ROOT, "images", "curvantML.png"), width=180)
st.sidebar.markdown("Visualizador de manobras em curvas")
if st.sidebar.button("Reprocessar dados", use_container_width=True):
    carregar_pipeline.clear()
    features_df, df_at, cfg, feat_cfg = carregar_pipeline(rebuild=True)
    st.rerun()
st.sidebar.divider()

visao = st.sidebar.radio(
    "Visualização",
    ["Curva selecionada", "Visão geral do local"],
    horizontal=False,
)

st.sidebar.divider()

classe_filter = st.sidebar.radio(
    "Filtrar por classe",
    ["Todas", "Segura", "Perigosa"],
    horizontal=True,
)
if classe_filter == "Segura":
    subset = features_df[features_df["manobra"] == 0]
elif classe_filter == "Perigosa":
    subset = features_df[features_df["manobra"] == 1]
else:
    subset = features_df

_CRITERIOS_MAPA = {
    "Kamm (aceleração total)": "manobra_accel_curva",
    "Lateral (accel_y + DNIT)": "manobra_lateral_curva",
    "Zigue-zague": "manobra_ziguezague_curva",
}
criterios_sel = st.sidebar.multiselect(
    "Filtrar por critério de risco",
    options=list(_CRITERIOS_MAPA.keys()),
    placeholder="Todos os critérios",
)
if criterios_sel:
    cols_sel = [_CRITERIOS_MAPA[c] for c in criterios_sel]
    mask = subset[[c for c in cols_sel if c in subset.columns]].any(axis=1)
    subset = subset[mask]

n_per = int((subset["manobra"] == 1).sum())
n_seg = int((subset["manobra"] == 0).sum())
st.sidebar.markdown(f"**{n_per + n_seg}** curvas — 🔴 {n_per} Perigosa · 🟢 {n_seg} Segura")
st.sidebar.divider()

locais_disp = sorted(subset["_local"].unique())
local_sel = st.sidebar.selectbox("Local de coleta", locais_disp)

subset_local = subset[subset["_local"] == local_sel]
rotas_disp = sorted(subset_local["id_route"].unique())

if visao == "Curva selecionada":
    rota_sel = st.sidebar.selectbox("Trajeto", rotas_disp)
    sub_rota = subset_local[subset_local["id_route"] == rota_sel]
    curvas_disp = sorted(sub_rota["id_trecho_curvo"].unique())

    def fmt_curva(tc):
        row = sub_rota[sub_rota["id_trecho_curvo"] == tc].iloc[0]
        label = "🔴" if row["manobra"] == 1 else "🟢"
        return f"{label} Curva {tc}"

    curva_sel = st.sidebar.selectbox("Curva", curvas_disp, format_func=fmt_curva)
else:
    rota_sel = rotas_disp[0]
    sub_rota = subset_local[subset_local["id_route"] == rota_sel]
    curvas_disp = sorted(sub_rota["id_trecho_curvo"].unique())
    curva_sel = curvas_disp[0] if curvas_disp else None

# Dados da curva selecionada

feat_row = sub_rota[sub_rota["id_trecho_curvo"] == curva_sel].iloc[0]
manobra_label = "Perigosa" if feat_row["manobra"] == 1 else "Segura"

curva_pts_raw = df_at[
    (df_at["id_route"] == rota_sel) & (df_at["trecho_curvo"] == curva_sel)
].copy()
inicio_curva = curva_pts_raw["time_sec"].min()

pre_pts_raw = df_at[
    (df_at["id_route"] == rota_sel)
    & (df_at["time_sec"] >= feat_row["time_inicio"])
    & (df_at["time_sec"] <= feat_row["time_fim"])
].copy()

curva_pts = enriquecer_com_bearings(curva_pts_raw)
pre_pts = pre_pts_raw.copy()

curva_pts["segmento"] = "Curva"
pre_pts["segmento"] = "Pré-curva"
todos = pd.concat([pre_pts, curva_pts], ignore_index=True)
todos["t_rel"] = todos["time_sec"] - inicio_curva
todos = enriquecer_com_bearings(todos)

# Curva selecionada

if visao == "Curva selecionada":
    cor = cor_manobra(manobra_label)
    st.markdown(
        f"<h2>Curva <b>#{curva_sel}</b> &nbsp;&nbsp;"
        f"<span style='background:{cor};color:white;padding:4px 14px;"
        f"border-radius:6px;font-size:1rem'>{manobra_label}</span></h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"Trajeto: `{rota_sel}`")

    dnit = curva_pts["classe_dnit"].mode()[0] if "classe_dnit" in curva_pts.columns else "N/A"
    raio_med = curva_pts["raio_curvatura"].median()
    vel_media = curva_pts["vehicle_speed"].mean()
    vel_max = curva_pts["vehicle_speed"].max()
    accel_lat_max = curva_pts["accel_y"].abs().max()
    var_vel = curva_pts["vehicle_speed"].diff().abs().sum()
    ctp_max = curva_pts["ctp_accel"].abs().max()
    dur_curva = curva_pts["time_sec"].max() - curva_pts["time_sec"].min()

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Vel. média", f"{vel_media:.0f} km/h")
    c2.metric("Vel. máx", f"{vel_max:.0f} km/h")
    c3.metric("Accel. lateral máx", f"{accel_lat_max:.2f} m/s²")
    c4.metric("Var. vel. acumulada", f"{var_vel:.1f} km/h",
              help="Soma de |Δv| ponto a ponto.")
    c5.metric("Accel. centrípeta máx", f"{ctp_max:.2f} m/s²",
              help="v²/R — força que empurra para fora da curva.")
    c6.metric("Raio med. / DNIT", f"{raio_med:.0f}m / {dnit}")
    c7.metric("Duração da curva", f"{dur_curva:.0f}s")

    st.markdown("**Critérios ativos durante a curva:**")
    _rm = cfg["risk_measures"]
    _kamm_lim = _rm["kamm_alpha"] * cfg["physics"]["mu"] * 9.81
    _criterios = [
        (
            "manobra_accel_curva", "Aceleração anormal (Kamm)",
            f"Círculo de Kamm: √(aₓ²+a_y²) > {_kamm_lim:.2f} m/s² "
            f"({_rm['kamm_alpha']} × μ{cfg['physics']['mu']} × g) em pelo menos um ponto da curva.",
        ),
        (
            "manobra_lateral_curva", "Direção perigosa (lateral)",
            f"|accel_y| > {_rm['limiar_accel_lateral']} m/s² em curva aberta ou mais fechada (R ≤ 500 m).",
        ),
        (
            "manobra_ziguezague_curva", "Zigue-zague",
            f"≥ {_rm['zigue_zague']['min_mudancas']} alternâncias de direção com "
            f"|Δθ| > {_rm['zigue_zague']['limiar_bearing']}° "
            f"e ctp_accel > {_rm['zigue_zague']['limiar_ctp']} m/s².",
        ),
    ]
    ca, cb, cc_ = st.columns(3)
    for col, (flag_col, nome, descricao) in zip([ca, cb, cc_], _criterios):
        ativo = bool(feat_row[flag_col])
        col.metric(label=nome, value="🔴 Ativo" if ativo else "⚪ Inativo", help=descricao)

    st.divider()
    st.subheader("Mapa do trecho")
    lead_gap = feat_cfg["extracao"].get("lead_gap", 0)
    st.caption(
        f"O trecho azul é a **janela de features** usada para predição. "
        f"Termina {lead_gap} m antes da entrada da curva (lead_gap). "
        f"O modelo faz a predição nesse ponto, antes de o veículo entrar na curva."
    )
    lat_c = todos["lat"].mean()
    lon_c = todos["lon"].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=17, tiles="CartoDB positron")

    traj_completo = df_at[df_at["id_route"] == rota_sel][["lat", "lon"]].dropna()
    if len(traj_completo) > 1:
        folium.PolyLine(
            traj_completo.values.tolist(),
            color="#aaaaaa", weight=1.5, opacity=0.4,
            tooltip="Trajeto completo",
        ).add_to(m)

    if len(pre_pts) > 1:
        folium.PolyLine(
            pre_pts[["lat", "lon"]].values.tolist(),
            color="#2166ac", weight=4, opacity=0.7,
            tooltip="Pré-curva (janela de features)",
        ).add_to(m)

    for _, row in pre_pts.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5, color="#2166ac", fill=True,
            fill_color="#4393c3", fill_opacity=0.8,
            tooltip=(
                f"Pré-curva | t={row['time_sec']:.0f}s<br>"
                f"Vel: {row['vehicle_speed']:.1f} km/h<br>"
                f"RPM: {row['engine_rpm']:.0f}"
            ),
        ).add_to(m)

    if len(curva_pts) > 1:
        folium.PolyLine(
            curva_pts[["lat", "lon"]].values.tolist(),
            color=cor, weight=4, opacity=0.8,
        ).add_to(m)

    for _, row in curva_pts.iterrows():
        c_pt = "#d73027" if row.get("conducao") == "Perigosa" else "#2b8a3e"
        criterios_pt = []
        if row.get("aceleracao_anormal", 0):
            criterios_pt.append("Accel")
        if row.get("zigue_zague", 0):
            criterios_pt.append("ZZ")
        tip = (
            f"<b>{row.get('conducao','?')}</b><br>"
            f"t={row['time_sec']:.0f}s | Vel: {row['vehicle_speed']:.1f} km/h<br>"
            f"Accel.lat: {row['accel_y']:.2f} m/s²<br>"
            f"DNIT: {row.get('classe_dnit','?')}"
        )
        if criterios_pt:
            tip += f"<br>Critérios: {', '.join(criterios_pt)}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7, color=c_pt, fill=True,
            fill_color=c_pt, fill_opacity=0.9,
            tooltip=folium.Tooltip(tip),
        ).add_to(m)

    folium.Marker(
        location=[curva_pts.iloc[0]["lat"], curva_pts.iloc[0]["lon"]],
        icon=folium.Icon(color="red" if manobra_label == "Perigosa" else "green",
                         icon="flag", prefix="fa"),
        tooltip="Início da curva",
    ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:12px;left:12px;z-index:9999;
                background:white;padding:8px 12px;border-radius:8px;
                font-size:12px;border:1px solid #ccc;line-height:1.6">
      <b>Legenda</b><br>
      <span style="color:#2166ac">&#9644;</span> Pré-curva (features)<br>
      <span style="color:#d73027">●</span> Perigosa &nbsp;
      <span style="color:#2b8a3e">●</span> Segura
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width="100%", height=500, returned_objects=[])

    st.subheader("Sensores ao longo do tempo")
    if todos.empty:
        st.warning("Sem dados de sensor para esta curva.")
    else:
        COLOR_SCALE = alt.Scale(domain=["Pré-curva", "Curva"], range=["#2166ac", cor])
        COLOR_ENC = alt.Color("segmento:N", scale=COLOR_SCALE,
                              legend=alt.Legend(title="Segmento", orient="top"))
        rule_df = pd.DataFrame({"t": [0], "label": ["início da curva"]})
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="black", strokeDash=[5, 4], strokeWidth=1.5)
            .encode(x=alt.X("t:Q"))
        )
        rule_lbl = (
            alt.Chart(rule_df)
            .mark_text(align="left", dx=4, dy=-5, fontSize=10, color="black")
            .encode(x="t:Q", text="label:N", y=alt.value(5))
        )

        def linha(campo, titulo, unidade="", height=160):
            base = alt.Chart(todos).mark_line(
                point=alt.OverlayMarkDef(size=30), strokeWidth=2,
            ).encode(
                x=alt.X("t_rel:Q", axis=alt.Axis(
                    title="Tempo relativo ao início da curva (s)", labelFontSize=10)),
                y=alt.Y(f"{campo}:Q", scale=alt.Scale(zero=False),
                        axis=alt.Axis(
                            title=f"{titulo} ({unidade})" if unidade else titulo,
                            labelFontSize=10)),
                color=COLOR_ENC,
                tooltip=["t_rel:Q", f"{campo}:Q", "segmento:N",
                         "vehicle_speed:Q", "conducao:N"],
            )
            return (base + rule + rule_lbl).properties(height=height, title=titulo)

        ch_speed = linha("vehicle_speed", "Velocidade", "km/h")
        ch_accy = linha("accel_y", "Accel. lateral", "m/s²")
        lim_ctp = cfg["risk_measures"]["zigue_zague"]["limiar_ctp"]
        ch_ctp = (
            linha("ctp_accel", "Accel. centrípeta v²/R", "m/s²")
            + alt.Chart(pd.DataFrame({"y": [lim_ctp]}))
            .mark_rule(color="#e6550d", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y=alt.Y("y:Q"))
        ).properties(height=160, title="Accel. centrípeta v²/R — gate zigue-zague")
        lim_bear = cfg["risk_measures"]["zigue_zague"]["limiar_bearing"]
        ch_bear = (
            linha("delta_bearing", "Variação de bearing |Δθ|", "°")
            + alt.Chart(pd.DataFrame({"y": [lim_bear]}))
            .mark_rule(color="#e6550d", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y=alt.Y("y:Q"))
        ).properties(height=160, title="Variação de bearing |Δθ| — critério zigue-zague")

        g_esq, g_dir = st.columns(2, gap="medium")
        with g_esq:
            st.altair_chart(ch_speed, width="stretch")
            st.altair_chart(ch_ctp, width="stretch")
        with g_dir:
            st.altair_chart(ch_accy, width="stretch")
            st.altair_chart(ch_bear, width="stretch")

    st.divider()
    st.subheader("Distribuição de classes neste trajeto")
    rota_feat = features_df[features_df["id_route"] == rota_sel].copy()
    rota_feat["Classe"] = rota_feat["manobra"].map({0: "Segura", 1: "Perigosa"})
    rota_feat["Critério"] = rota_feat.apply(
        lambda r: ("Accel" if r["manobra_accel_curva"] else "")
        + ("+" if r["manobra_accel_curva"] and r["manobra_ziguezague_curva"] else "")
        + ("ZZ" if r["manobra_ziguezague_curva"] else "")
        or "Nenhum",
        axis=1,
    )
    col_dist, col_crit = st.columns(2)
    with col_dist:
        dist_df = rota_feat["Classe"].value_counts().reset_index()
        dist_df.columns = ["Classe", "Contagem"]
        st.altair_chart(
            alt.Chart(dist_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Classe:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Contagem:Q"),
                color=alt.Color("Classe:N",
                    scale=alt.Scale(domain=["Segura", "Perigosa"],
                                    range=["#2b8a3e", "#d73027"]), legend=None),
                tooltip=["Classe:N", "Contagem:Q"],
            ).properties(title=f"Curvas em {rota_sel}", height=180),
            width="stretch",
        )
    with col_crit:
        crit_df = rota_feat[rota_feat["Classe"] == "Perigosa"]["Critério"].value_counts().reset_index()
        crit_df.columns = ["Critério", "Contagem"]
        if not crit_df.empty:
            st.altair_chart(
                alt.Chart(crit_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#d73027")
                .encode(
                    x=alt.X("Critério:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Contagem:Q"),
                    tooltip=["Critério:N", "Contagem:Q"],
                ).properties(title="Critérios das manobras Perigosa", height=180),
                width="stretch",
            )

    with st.expander("Dados brutos da curva selecionada"):
        cols_show = [
            "time_sec", "vehicle_speed", "engine_rpm",
            "accel_x", "accel_y", "ctp_accel",
            "conducao", "aceleracao_anormal", "zigue_zague",
            "raio_curvatura", "classe_dnit", "bearing", "delta_bearing",
        ]
        cols_ok = [c for c in cols_show if c in curva_pts.columns]
        st.dataframe(curva_pts[cols_ok].round(3), width="stretch")

    with st.expander("Features extraídas desta curva"):
        st.dataframe(feat_row.to_frame().T.round(3), width="stretch")

# Tab 2: Visão geral do local

else:  # Visão geral do local
    st.markdown(f"<h2>Local de coleta: <b>{local_sel}</b></h2>", unsafe_allow_html=True)

    local_feat = subset[subset["_local"] == local_sel].copy()
    local_feat["Classe"] = local_feat["manobra"].map({0: "Segura", 1: "Perigosa"})

    n_tot = len(local_feat)
    n_per_loc = int(local_feat["manobra"].sum())
    n_seg_loc = n_tot - n_per_loc
    v1, v2, v3 = st.columns(3)
    v1.metric("Total de curvas", n_tot)
    v2.metric("🔴 Perigosa", n_per_loc)
    v3.metric("🟢 Segura", n_seg_loc)

    st.subheader("Mapa de todas as curvas do local")
    st.warning("🚔Cada círculo representa uma curva completa (ponto de entrada). Selecione uma curva na opção **Curva selecionada** para ver todos os pontos do trecho.")

    entry_rows = []
    for _, fr in local_feat.iterrows():
        route_id = fr["id_route"]
        tc = fr["id_trecho_curvo"]
        grp = df_at[(df_at["id_route"] == route_id) & (df_at["trecho_curvo"] == tc)]
        if grp.empty:
            continue
        first = grp.sort_values("distancia_acumulada").iloc[0]
        entry_rows.append({
            "lat": first["lat"], "lon": first["lon"],
            "tc": tc, "route": route_id,
            "manobra": int(fr["manobra"]),
            "vel_media": grp["vehicle_speed"].mean(),
            "raio_med": grp["raio_curvatura"].median(),
            "dnit": grp["classe_dnit"].mode()[0] if "classe_dnit" in grp.columns else "?",
        })

    if entry_rows:
        lats = [r["lat"] for r in entry_rows]
        lons = [r["lon"] for r in entry_rows]
        m2 = folium.Map(
            location=[np.mean(lats), np.mean(lons)],
            zoom_start=13, tiles="CartoDB positron",
        )

        for route_id in local_feat["id_route"].unique():
            traj = df_at[df_at["id_route"] == route_id][["lat", "lon"]].dropna()
            if len(traj) > 1:
                folium.PolyLine(
                    traj.values.tolist(),
                    color="#aaaaaa", weight=1.5, opacity=0.5,
                    tooltip=route_id,
                ).add_to(m2)

        for er in entry_rows:
            c_er = "#d73027" if er["manobra"] == 1 else "#2b8a3e"
            label_er = "Perigosa" if er["manobra"] == 1 else "Segura"
            folium.CircleMarker(
                location=[er["lat"], er["lon"]],
                radius=8, color=c_er, fill=True,
                fill_color=c_er, fill_opacity=0.85,
                tooltip=folium.Tooltip(
                    f"<b>{label_er}</b> — Curva {er['tc']}<br>"
                    f"Trajeto: {er['route']}<br>"
                    f"Vel. média: {er['vel_media']:.0f} km/h<br>"
                    f"Raio med.: {er['raio_med']:.0f} m — DNIT: {er['dnit']}"
                ),
            ).add_to(m2)

        m2.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        legend2 = """
        <div style="position:fixed;bottom:12px;left:12px;z-index:9999;
                    background:white;padding:8px 12px;border-radius:8px;
                    font-size:12px;border:1px solid #ccc;line-height:1.6">
          <b>Legenda</b><br>
          <span style="color:#d73027">●</span> Perigosa &nbsp;
          <span style="color:#2b8a3e">●</span> Segura<br>
          <span style="color:#aaaaaa">&#9644;</span> Trajeto
        </div>"""
        m2.get_root().html.add_child(folium.Element(legend2))
        st_folium(m2, width="100%", height=560, returned_objects=[])
    else:
        st.warning("Sem pontos de entrada de curva disponíveis para este local.")

    st.subheader("Tabela de curvas")
    tbl = local_feat[["id_route", "id_trecho_curvo", "Classe",
                       "manobra_accel_curva", "manobra_lateral_curva",
                       "manobra_ziguezague_curva"]].copy()
    tbl.columns = ["Trajeto", "Curva", "Classe", "Kamm", "Lateral", "Zigue-zague"]
    st.dataframe(tbl.reset_index(drop=True), width="stretch", height=300)
