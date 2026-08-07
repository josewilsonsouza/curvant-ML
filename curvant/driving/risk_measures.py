"""Critérios de conduta de risco em curva, medidos só com GPS e velocidade.

Os critérios antigos de Kamm e de aceleração lateral foram removidos em jul/2026.
Eles vinham do acelerômetro, que é do celular usado na coleta e não da central do
veículo. A análise em docs/ANALISE_DADOS.md mostra que esse sensor não acompanha a
dinâmica do carro: a correlação entre a leitura lateral e a aceleração centrípeta
calculada pela física fica em 0,055, o sinal não inverte entre curvas para a
esquerda e para a direita, e a melhor reorientação linear possível recupera 0,3%
do sinal. Como controle, a aceleração longitudinal calculada pelo GPS concorda com
a calculada pelo OBD a 0,712, então a grandeza é mensurável e o problema é do
sensor, não do método.

O que sobrou mede conduta, não exigência da curva: o motorista precisou corrigir
depois de já estar na curva, seja freando forte (frenagem tardia) ou alternando a
direção (zigue-zague). A velocidade de entrada comparada à velocidade segura ficou
de fora de propósito: ela é reproduzível com F1 0,91 a partir da velocidade de
aproximação e do raio, que são as próprias features, então prever esse rótulo não
ensinaria nada. Essa dimensão continua coberta pelos alvos de ISL e v_critica.
"""
import numpy as np
import pandas as pd

_DNIT_RISCO: dict[str, int] = {
    'suave': 0, 'aberta': 1, 'media': 2, 'fechada': 3, 'muito_fechada': 4,
}

_ACCEL_MAX_PLAUSIVEL = 6.0  # m/s² - acima disso é falha de leitura da velocidade, não frenagem


def calcular_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Ângulo de direção (bearing) entre dois pontos GPS, em graus [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _desaceleracao_maxima(janela: pd.DataFrame) -> float:
    """Maior desaceleração (m/s², positiva) observada DENTRO do segmento de curva.

    Calculada pela variação da velocidade entre pontos consecutivos do próprio
    segmento, então mede a frenagem depois que o carro já entrou na curva. Frear
    antes da curva é condução prudente e não entra aqui.

    A velocidade é o sensor confiável do conjunto: a aceleração longitudinal obtida
    dela pelo GPS e pelo OBD concorda a 0,712, o que valida tanto a grandeza quanto
    o cálculo.
    """
    if not {'vehicle_speed', 'dt'} <= set(janela.columns) or len(janela) < 2:
        return 0.0

    v_ms = janela['vehicle_speed'].values.astype(float) / 3.6
    dt   = janela['dt'].values.astype(float)[1:]
    dv   = np.diff(v_ms)

    with np.errstate(divide='ignore', invalid='ignore'):
        a_long = np.where(dt > 0, dv / dt, np.nan)

    a_long = a_long[np.isfinite(a_long) & (np.abs(a_long) <= _ACCEL_MAX_PLAUSIVEL)]
    if a_long.size == 0:
        return 0.0

    return float(max(-a_long.min(), 0.0))


def _detectar_zigue_zague(
    janela: pd.DataFrame,
    limiar_bearing: float = 15.0,
    limiar_ctp: float = 0.3,
    min_mudancas: int = 3,
    vel_min_kmh: float = 5.0,
) -> bool:
    # Em velocidades baixas, o espaçamento GPS (~1–3 m) é da mesma ordem que o erro de
    # posição (~3–5 m), gerando mudanças de bearing fictícias mesmo em linha reta.
    if 'vehicle_speed' in janela.columns:
        janela = janela[janela['vehicle_speed'] >= vel_min_kmh]

    lats = janela['lat'].tolist()
    lons = janela['lon'].tolist()
    ctp  = janela['ctp_accel'].tolist()
    if len(lats) < 2:
        return False

    bearings = np.array([
        calcular_bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
        for i in range(1, len(lats))
    ])

    contador     = 0
    ultimo_sinal = 0  # sinal da última mudança qualificada; 0 = nenhuma ainda

    for i in range(1, len(bearings)):
        # preserva o sinal: positivo = virou à direita, negativo = à esquerda
        mudanca = (bearings[i] - bearings[i - 1] + 180) % 360 - 180
        if abs(mudanca) > limiar_bearing and abs(ctp[i]) > limiar_ctp:
            sinal = int(np.sign(mudanca))
            if sinal != ultimo_sinal:   # alternância real de direção
                contador     += 1
                ultimo_sinal  = sinal
                if contador >= min_mudancas:
                    return True
            # mesma direção que a anterior: curva contínua, não zigue-zague

    return False


def caracterizar_janela(
    janela: pd.DataFrame,
    limiar_desaceleracao: float = 2.0,
    zz_limiar_bearing: float = 15.0,
    zz_limiar_ctp: float = 0.3,
    zz_min_mudancas: int = 3,
    margem_histerese: float = 0.0,
) -> dict:
    """Aplica os dois critérios de conduta a um segmento de curva. Retorna dict de bools.

    Critério 1 — Frenagem tardia:
        max(-dv/dt) dentro da curva > limiar_desaceleracao
        O motorista freou forte depois de já estar na curva, sinal de que não
        antecipou o que vinha pela frente.

    Critério 2 — Zigue-zague:
        alternâncias de direção acima do limiar de bearing, com curvatura mínima.
        O motorista corrigiu a trajetória mais de uma vez dentro da curva.

    Com margem_histerese > 0, uma desaceleração a menos dessa fração do limiar marca
    o critério como indefinido: tão perto do limiar, o rótulo binário é decidido pelo
    ruído da medida e não pelo comportamento. O rótulo em si não muda; a flag
    *_indef permite descartar essas janelas do treino. O zigue-zague é contagem de
    alternâncias, sem limiar contínuo, então não participa da histerese.
    """
    desaceleracao     = _desaceleracao_maxima(janela)
    manobra_frenagem  = bool(desaceleracao > limiar_desaceleracao)

    m = margem_histerese
    frenagem_indef = bool(
        m > 0
        and limiar_desaceleracao * (1 - m) < desaceleracao < limiar_desaceleracao * (1 + m)
    )

    manobra_ziguezague = _detectar_zigue_zague(
        janela, zz_limiar_bearing, zz_limiar_ctp, zz_min_mudancas
    )

    # Descritivo da geometria, mantido para as análises; não entra em nenhum critério.
    risco_dnit = (
        int(janela['classe_dnit'].map(_DNIT_RISCO).max())
        if 'classe_dnit' in janela.columns else 3
    )

    return {
        'manobra_frenagem_indef': frenagem_indef,
        'manobra_frenagem':       manobra_frenagem,
        'manobra_ziguezague':     manobra_ziguezague,
        'risco_dnit':             risco_dnit,
    }


def caracterizar_conducao(
    df: pd.DataFrame,
    limiar_desaceleracao: float = 2.0,
    margem_histerese: float = 0.0,
    **zz_kwargs,
) -> pd.DataFrame:
    """
    Caracteriza risco por segmento de curva (trecho contíguo curva=True).

    Para cada segmento, avalia os dois critérios sobre os pontos do segmento.
    Os rótulos são atribuídos apenas aos pontos do segmento; pontos fora de
    curvas recebem False/Segura.

    O combinado é indefinido quando nenhum critério é claramente positivo (fora
    da faixa de histerese) e a frenagem caiu dentro dela: nesse caso o OR
    poderia flipar com o ruído da medida.
    """
    _cols = [
        'manobra_frenagem', 'manobra_ziguezague', 'manobra_combinado',
        'manobra_frenagem_indef', 'manobra_indefinido',
    ]

    df_out = df.sort_values('time_sec').copy()
    for col in _cols:
        df_out[col] = False
    df_out['conducao']   = 'Segura'
    df_out['risco_dnit'] = 0
    df_out['id_janela']  = 0

    df_out['_bloco'] = (df_out['curva'] != df_out['curva'].shift()).cumsum()

    id_janela = 1
    for _, bloco in df_out.groupby('_bloco', sort=False):
        if not bool(bloco['curva'].iloc[0]):
            continue

        if len(bloco) < 2:
            continue

        r    = caracterizar_janela(
            bloco, limiar_desaceleracao,
            margem_histerese=margem_histerese, **zz_kwargs,
        )
        comb  = r['manobra_frenagem'] or r['manobra_ziguezague']
        claro = (
            (r['manobra_frenagem'] and not r['manobra_frenagem_indef'])
            or r['manobra_ziguezague']
        )
        indef = (not claro) and r['manobra_frenagem_indef']

        idx = bloco.index
        df_out.loc[idx, 'manobra_frenagem']       = r['manobra_frenagem']
        df_out.loc[idx, 'manobra_ziguezague']     = r['manobra_ziguezague']
        df_out.loc[idx, 'manobra_combinado']      = comb
        df_out.loc[idx, 'manobra_frenagem_indef'] = r['manobra_frenagem_indef']
        df_out.loc[idx, 'manobra_indefinido']     = indef
        df_out.loc[idx, 'conducao']               = 'Perigosa' if comb else 'Segura'
        df_out.loc[idx, 'risco_dnit']             = r['risco_dnit']
        df_out.loc[idx, 'id_janela']              = id_janela
        id_janela += 1

    return df_out.drop(columns=['_bloco'])


def caracterizar_todos_trajetos(
    dfs_curves: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Aplica caracterizar_conducao em cada trajeto e concatena."""
    partes = [
        caracterizar_conducao(grupo, **kwargs)
        for _, grupo in dfs_curves.groupby('id_route', sort=False)
    ]
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
