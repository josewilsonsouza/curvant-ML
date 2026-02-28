"""
CurvantML — Visualizador Interativo de Curvas
Uso: streamlit run app.py
"""
import io
import contextlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import altair as alt

st.set_page_config(
    page_title="CurvantML — Visualizador de Curvas",
    layout="wide",
    page_icon="🚗",
)

# ── Pipeline (cacheado em memória) ────────────────────────────────────────────

@st.cache_data(show_spinner="Rodando pipeline — primeira execução (~2 min)...")
def carregar_pipeline():
    import os
    from run_experiments import carregar_config, etapa_curvas, etapa_analise_conducao
    from src.curve_detection import identificar_trechos_curvos
    from src.features import extrair_features

    cfg = carregar_config()

    clean_path = "data/eletro_rjdf_serra_clean.parquet"
    raw_path   = "data/eletro_rjdf_serra.parquet"
    data_path  = clean_path if os.path.exists(clean_path) else raw_path
    df = pd.read_parquet(data_path)

    with contextlib.redirect_stdout(io.StringIO()):
        dfs_curves = etapa_curvas(df, cfg)

    dfs_curves["ctp_accel"] = (
        (dfs_curves["vehicle_speed"] / 3.6) ** 2 / dfs_curves["raio_curvatura"]
    )

    with contextlib.redirect_stdout(io.StringIO()):
        df_analysis = etapa_analise_conducao(dfs_curves, cfg, plot=False)

    df_analysis[["aceleracao_anormal", "direcao_perigosa", "zigue_zague"]] = (
        df_analysis[["aceleracao_anormal", "direcao_perigosa", "zigue_zague"]].astype(int)
    )

    with contextlib.redirect_stdout(io.StringIO()):
        df_at = identificar_trechos_curvos(df_analysis)
        features_df = extrair_features(df_at, janela_tempo=cfg["features"]["janela_tempo"])

    return features_df, df_at, cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def bearing_wrap(b2: float, b1: float) -> float:
    return (b2 - b1 + 180) % 360 - 180


def enriquecer_com_bearings(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona bearing e variação de bearing (wrap-corrected) ao DataFrame."""
    from src.driving_analysis import calcular_bearing
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


# ── Dados ─────────────────────────────────────────────────────────────────────

features_df, df_at, cfg = carregar_pipeline()

# ── Sidebar — filtros ─────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/48/car.png", width=120)
st.sidebar.title("CurvantML")
st.sidebar.markdown("Visualizador de manobras em curvas")
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

n_per = int((subset["manobra"] == 1).sum())
n_seg = int((subset["manobra"] == 0).sum())
st.sidebar.markdown(f"**{n_per + n_seg}** curvas — 🔴 {n_per} Perigosa · 🟢 {n_seg} Segura")
st.sidebar.divider()

rotas_disp = sorted(subset["id_route"].unique())
rota_sel = st.sidebar.selectbox("Trajeto", rotas_disp)

sub_rota = subset[subset["id_route"] == rota_sel]
curvas_disp = sorted(sub_rota["id_trecho_curvo"].unique())

# Mostrar label ao lado de cada curva
def fmt_curva(tc):
    row = sub_rota[sub_rota["id_trecho_curvo"] == tc].iloc[0]
    label = "🔴" if row["manobra"] == 1 else "🟢"
    return f"{label} Curva {tc}"

curva_sel = st.sidebar.selectbox(
    "Curva",
    curvas_disp,
    format_func=fmt_curva,
)

# ── Dados da curva selecionada ────────────────────────────────────────────────

feat_row = sub_rota[sub_rota["id_trecho_curvo"] == curva_sel].iloc[0]
manobra_label = "Perigosa" if feat_row["manobra"] == 1 else "Segura"

curva_pts_raw = df_at[
    (df_at["id_route"] == rota_sel) & (df_at["trecho_curvo"] == curva_sel)
].copy()
inicio_curva = curva_pts_raw["time_sec"].min()

pre_pts_raw = df_at[
    (df_at["id_route"] == rota_sel)
    & (df_at["time_sec"] < inicio_curva)
    & (df_at["time_sec"] >= inicio_curva - cfg["features"]["janela_tempo"])
].copy()

curva_pts = enriquecer_com_bearings(curva_pts_raw)
pre_pts = pre_pts_raw.copy()

curva_pts["segmento"] = "Curva"
pre_pts["segmento"] = "Pré-curva"
todos = pd.concat([pre_pts, curva_pts], ignore_index=True)
todos["t_rel"] = todos["time_sec"] - inicio_curva
todos = enriquecer_com_bearings(todos)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────

cor = cor_manobra(manobra_label)
st.markdown(
    f"<h2>Curva <b>#{curva_sel}</b> &nbsp;&nbsp;"
    f"<span style='background:{cor};color:white;padding:4px 14px;"
    f"border-radius:6px;font-size:1rem'>{manobra_label}</span></h2>",
    unsafe_allow_html=True,
)
st.caption(f"Trajeto: `{rota_sel}`")

# ── Métricas rápidas ──────────────────────────────────────────────────────────

dnit = curva_pts["classe_dnit"].mode()[0] if "classe_dnit" in curva_pts.columns else "N/A"
raio_med = curva_pts["raio_curvatura"].median()
vel_media = curva_pts["vehicle_speed"].mean()
vel_max = curva_pts["vehicle_speed"].max()
accel_lat_max = curva_pts["accel_y"].abs().max()
var_vel = curva_pts["vehicle_speed"].diff().abs().sum()
rpm_med = curva_pts["engine_rpm"].mean()
ctp_max = curva_pts["ctp_accel"].abs().max()
dur_curva = curva_pts["time_sec"].max() - curva_pts["time_sec"].min()

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Vel. média", f"{vel_media:.0f} km/h")
c2.metric("Vel. máx", f"{vel_max:.0f} km/h")
c3.metric("Accel. lateral máx", f"{accel_lat_max:.2f} m/s²")
c4.metric("Var. vel. acumulada", f"{var_vel:.1f} km/h",
          help="Soma de |Δv| ponto a ponto. >20 km/h → aceleração anormal")
c5.metric("Accel. centrípeta máx", f"{ctp_max:.2f} m/s²",
          help="v²/R — força que empurra para fora da curva")
c6.metric("Raio med. / DNIT", f"{raio_med:.0f}m / {dnit}")
c7.metric("Duração da curva", f"{dur_curva:.0f}s")

# Critérios
st.markdown("**Critérios ativos durante a curva:**")
ca, cb, cc_ = st.columns(3)
for col, flag_col, nome in [
    (ca, "manobra_accel_perigo", "Aceleração anormal"),
    (cb, "manobra_dir_perigosa", "Direção perigosa (desabilitado)"),
    (cc_, "manobra_zigue_zague", "Zigue-zague"),
]:
    ativo = bool(feat_row[flag_col])
    icone = "🔴" if ativo else "⚪"
    col.markdown(f"{icone} **{nome}**")

st.divider()

# ── Layout principal: mapa + gráficos ─────────────────────────────────────────

col_map, col_charts = st.columns([1, 1.6], gap="medium")

# ── Mapa Folium ───────────────────────────────────────────────────────────────

with col_map:
    st.subheader("Mapa do trecho")
    lat_c = todos["lat"].mean()
    lon_c = todos["lon"].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=17, tiles="CartoDB positron")

    # Linha de contexto: todo o trajeto (cinza, fino)
    traj_completo = df_at[df_at["id_route"] == rota_sel][["lat", "lon"]].dropna()
    if len(traj_completo) > 1:
        folium.PolyLine(
            traj_completo.values.tolist(),
            color="#aaaaaa", weight=1.5, opacity=0.4,
            tooltip="Trajeto completo",
        ).add_to(m)

    # Linha da pré-curva (azul)
    if len(pre_pts) > 1:
        folium.PolyLine(
            pre_pts[["lat", "lon"]].values.tolist(),
            color="#2166ac", weight=4, opacity=0.7,
            tooltip="Pré-curva (janela de features)",
        ).add_to(m)

    # Pontos pré-curva
    for _, row in pre_pts.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="#2166ac",
            fill=True,
            fill_color="#4393c3",
            fill_opacity=0.8,
            tooltip=(
                f"Pré-curva | t={row['time_sec']:.0f}s<br>"
                f"Vel: {row['vehicle_speed']:.1f} km/h<br>"
                f"RPM: {row['engine_rpm']:.0f}"
            ),
        ).add_to(m)

    # Linha da curva
    if len(curva_pts) > 1:
        folium.PolyLine(
            curva_pts[["lat", "lon"]].values.tolist(),
            color=cor, weight=4, opacity=0.8,
        ).add_to(m)

    # Pontos da curva — coloridos por conducao de cada ponto
    for _, row in curva_pts.iterrows():
        c_pt = "#d73027" if row.get("conducao") == "Perigosa" else "#2b8a3e"
        criterios = []
        if row.get("aceleracao_anormal", 0):
            criterios.append("Accel")
        if row.get("zigue_zague", 0):
            criterios.append("ZZ")
        tip = (
            f"<b>{row.get('conducao','?')}</b><br>"
            f"t={row['time_sec']:.0f}s | Vel: {row['vehicle_speed']:.1f} km/h<br>"
            f"Accel.lat: {row['accel_y']:.2f} m/s²<br>"
            f"DNIT: {row.get('classe_dnit','?')}"
        )
        if criterios:
            tip += f"<br>Critérios: {', '.join(criterios)}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color=c_pt,
            fill=True,
            fill_color=c_pt,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tip),
        ).add_to(m)

    # Marcador de início da curva
    folium.Marker(
        location=[curva_pts.iloc[0]["lat"], curva_pts.iloc[0]["lon"]],
        icon=folium.Icon(color="red" if manobra_label == "Perigosa" else "green",
                         icon="flag", prefix="fa"),
        tooltip="Início da curva",
    ).add_to(m)

    # Legenda
    legend = """
    <div style="position:fixed;bottom:12px;left:12px;z-index:9999;
                background:white;padding:8px 12px;border-radius:8px;
                font-size:12px;border:1px solid #ccc;line-height:1.6">
      <b>Legenda</b><br>
      <span style="color:#2166ac">&#9644;</span> Pré-curva (features)<br>
      <span style="color:#d73027">●</span> Perigosa &nbsp;
      <span style="color:#2b8a3e">●</span> Segura
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    st_folium(m, width=None, height=480, returned_objects=[])

# ── Gráficos Altair ───────────────────────────────────────────────────────────

with col_charts:
    st.subheader("Sensores ao longo do tempo")

    if todos.empty:
        st.warning("Sem dados de sensor para esta curva.")
    else:
        COLOR_SCALE = alt.Scale(
            domain=["Pré-curva", "Curva"],
            range=["#2166ac", cor],
        )
        COLOR_ENC = alt.Color("segmento:N", scale=COLOR_SCALE,
                              legend=alt.Legend(title="Segmento", orient="top"))

        # Linha vertical em t=0 (início da curva)
        rule_df = pd.DataFrame({"t": [0], "label": ["início da curva"]})
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="black", strokeDash=[5, 4], strokeWidth=1.5)
            .encode(x=alt.X("t:Q"))
        )
        rule_lbl = (
            alt.Chart(rule_df)
            .mark_text(align="left", dx=4, dy=-5, fontSize=10, color="black")
            .encode(x="t:Q", text="label:N",
                    y=alt.value(5))
        )

        def linha(campo, titulo, unidade="", height=100):
            base = alt.Chart(todos).mark_line(
                point=alt.OverlayMarkDef(size=30),
                strokeWidth=2,
            ).encode(
                x=alt.X("t_rel:Q",
                        axis=alt.Axis(title="Tempo relativo ao início da curva (s)",
                                      labelFontSize=10)),
                y=alt.Y(f"{campo}:Q",
                        axis=alt.Axis(title=f"{titulo} ({unidade})" if unidade else titulo,
                                      labelFontSize=10)),
                color=COLOR_ENC,
                tooltip=["t_rel:Q", f"{campo}:Q", "segmento:N",
                         "vehicle_speed:Q", "conducao:N"],
            )
            return (base + rule + rule_lbl).properties(height=height, title=titulo)

        # Velocidade — com faixa de limite por DNIT
        ch_speed = linha("vehicle_speed", "Velocidade", "km/h", 120)

        # Variação acumulada de velocidade (critério de aceleração anormal)
        ch_varvel = linha("var_vel_acum", "Variação vel. acumulada $\sum|\Delta v|$", "km/h", 90)
        limiar_accel = cfg["driving_analysis"]["var_velocidade_max"]
        limiar_df = pd.DataFrame({"y": [limiar_accel]})
        limiar_rule = (
            alt.Chart(limiar_df)
            .mark_rule(color="#e6550d", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y=alt.Y("y:Q"))
        )
        limiar_lbl = (
            alt.Chart(limiar_df)
            .mark_text(align="left", dx=4, fontSize=10, color="#e6550d")
            .encode(y="y:Q", text=alt.value(f"limiar {limiar_accel} km/h"))
        )
        ch_varvel = (ch_varvel + limiar_rule + limiar_lbl).properties(
            height=90, title="Variação vel. acumulada $\sum|\Delta v|$ — critério aceleração anormal"
        )

        # Aceleração lateral (accel_y)
        ch_accy = linha("accel_y", "Accel. lateral accel_y", "m/s²", 100)

        # Aceleração centrípeta (v²/R) — zigue-zague
        ch_ctp = linha("ctp_accel", "Accel. centrípeta v²/R", "m/s²", 100)
        lim_ctp_df = pd.DataFrame(
            {"y": [cfg["driving_analysis"]["zigue_zague"]["limiar_accel_lateral"]]}
        )
        lim_ctp_rule = (
            alt.Chart(lim_ctp_df)
            .mark_rule(color="#e6550d", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y=alt.Y("y:Q"))
        )
        ch_ctp = (ch_ctp + lim_ctp_rule).properties(
            height=100, title="Aceleração centrípeta v²/R — gate do zigue-zague"
        )

        # Variação de bearing |Δbearing| — zigue-zague
        ch_bear = linha("delta_bearing", "Variação de bearing |Δθ|", "°", 100)
        lim_bear_df = pd.DataFrame(
            {"y": [cfg["driving_analysis"]["zigue_zague"]["limiar_bearing"]]}
        )
        lim_bear_rule = (
            alt.Chart(lim_bear_df)
            .mark_rule(color="#e6550d", strokeDash=[4, 3], strokeWidth=1.5)
            .encode(y=alt.Y("y:Q"))
        )
        ch_bear = (ch_bear + lim_bear_rule).properties(
            height=100, title="Variação de bearing |Δθ| — critério zigue-zague"
        )

        # RPM
        ch_rpm = linha("engine_rpm", "RPM do motor", "rpm", 90)

        chart = alt.vconcat(
            ch_speed, ch_varvel, ch_accy, ch_ctp, ch_bear, ch_rpm,
            spacing=6,
        ).resolve_scale(color="shared")

        st.altair_chart(chart, width='stretch')

# ── Contexto do trajeto ───────────────────────────────────────────────────────

st.divider()
st.subheader("Distribuição de classes neste trajeto")

rota_feat = features_df[features_df["id_route"] == rota_sel].copy()
rota_feat["Classe"] = rota_feat["manobra"].map({0: "Segura", 1: "Perigosa"})
rota_feat["Critério"] = rota_feat.apply(
    lambda r: ("Accel" if r["manobra_accel_perigo"] else "")
    + ("+" if r["manobra_accel_perigo"] and r["manobra_zigue_zague"] else "")
    + ("ZZ" if r["manobra_zigue_zague"] else "")
    or "Nenhum",
    axis=1,
)

col_dist, col_crit = st.columns(2)

with col_dist:
    dist_df = rota_feat["Classe"].value_counts().reset_index()
    dist_df.columns = ["Classe", "Contagem"]
    bar_dist = (
        alt.Chart(dist_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Classe:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Contagem:Q"),
            color=alt.Color(
                "Classe:N",
                scale=alt.Scale(domain=["Segura", "Perigosa"],
                                range=["#2b8a3e", "#d73027"]),
                legend=None,
            ),
            tooltip=["Classe:N", "Contagem:Q"],
        )
        .properties(title=f"Curvas em {rota_sel}", height=180)
    )
    st.altair_chart(bar_dist, width='stretch')

with col_crit:
    crit_df = rota_feat[rota_feat["Classe"] == "Perigosa"]["Critério"].value_counts().reset_index()
    crit_df.columns = ["Critério", "Contagem"]
    if not crit_df.empty:
        bar_crit = (
            alt.Chart(crit_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#d73027")
            .encode(
                x=alt.X("Critério:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Contagem:Q"),
                tooltip=["Critério:N", "Contagem:Q"],
            )
            .properties(title="Critérios das manobras Perigosa", height=180)
        )
        st.altair_chart(bar_crit, width='stretch')

# ── Dados brutos ──────────────────────────────────────────────────────────────

with st.expander("Dados brutos da curva selecionada"):
    cols_show = [
        "time_sec", "vehicle_speed", "engine_rpm",
        "accel_x", "accel_y", "ctp_accel",
        "conducao", "aceleracao_anormal", "zigue_zague",
        "raio_curvatura", "classe_dnit",
        "bearing", "delta_bearing",
    ]
    cols_ok = [c for c in cols_show if c in curva_pts.columns]
    st.dataframe(curva_pts[cols_ok].round(3), width='stretch')

with st.expander("Features extraídas desta curva"):
    st.dataframe(feat_row.to_frame().T.round(3), width='stretch')
