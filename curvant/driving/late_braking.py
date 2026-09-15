"""Correção tardia: o motorista freou forte depois de já estar dentro da curva.

É o único critério de comportamento do projeto. Houve outros, todos removidos, e o
motivo de cada remoção está em docs/CORRECAO_TARDIA.md. O resumo é que os critérios
vindos do acelerômetro mediam vibração do celular, e o de zigue-zague media o oposto
deste aqui: frear dentro da curva derruba o v²/R que aquele critério acompanhava.
"""
import numpy as np
import pandas as pd

_ACCEL_MAX_PLAUSIVEL = 6.0  # m/s² - acima disso é falha de leitura da velocidade, não frenagem


def _desaceleracao_maxima(janela: pd.DataFrame) -> float:
    """Maior desaceleração observada dentro do segmento de curva, em m/s² positivos.

    Sai da variação da velocidade entre pontos consecutivos do próprio segmento, então
    mede a frenagem depois que o carro já entrou. Frear antes da curva é condução
    prudente e não entra aqui. A velocidade é o sensor confiável do conjunto: GPS e OBD
    concordam bem na aceleração longitudinal, o que valida grandeza e cálculo.
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


def avaliar_segmento(
    janela: pd.DataFrame,
    limiar_desaceleracao: float = 2.0,
    margem_histerese: float = 0.0,
) -> dict:
    """Aplica o critério a um segmento de curva.

    Dispara quando a maior desaceleração passa do limiar. Com margem_histerese, uma
    desaceleração colada no limiar marca o segmento como indefinido: ali o rótulo
    binário seria decidido pela quantização da velocidade do OBD, não pelo
    comportamento. O rótulo não muda, mas a flag permite tirar o segmento do treino.
    """
    desaceleracao = _desaceleracao_maxima(janela)
    correcao      = bool(desaceleracao > limiar_desaceleracao)

    m = margem_histerese
    indefinida = bool(
        m > 0
        and limiar_desaceleracao * (1 - m) < desaceleracao < limiar_desaceleracao * (1 + m)
    )

    return {'correcao_tardia': correcao, 'correcao_indefinida': indefinida}


def rotular_correcao_tardia(
    df: pd.DataFrame,
    limiar_desaceleracao: float = 2.0,
    margem_histerese: float = 0.0,
) -> pd.DataFrame:
    """
    Rotula cada segmento de curva (trecho contíguo com curva=True) de um trajeto.

    Os rótulos são atribuídos aos pontos do segmento; pontos fora de curvas ficam
    False.
    """
    df_out = df.sort_values('time_sec').copy()
    for col in ('correcao_tardia', 'correcao_indefinida'):
        df_out[col] = False
    df_out['id_janela'] = 0

    df_out['_bloco'] = (df_out['curva'] != df_out['curva'].shift()).cumsum()

    id_janela = 1
    for _, bloco in df_out.groupby('_bloco', sort=False):
        if not bool(bloco['curva'].iloc[0]) or len(bloco) < 2:
            continue

        r   = avaliar_segmento(bloco, limiar_desaceleracao, margem_histerese)
        idx = bloco.index
        df_out.loc[idx, 'correcao_tardia']     = r['correcao_tardia']
        df_out.loc[idx, 'correcao_indefinida'] = r['correcao_indefinida']
        df_out.loc[idx, 'id_janela']           = id_janela
        id_janela += 1

    return df_out.drop(columns=['_bloco'])
