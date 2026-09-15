"""
Representação tabular: modelos sobre as features já resumidas por curva.

Um alvo por subcomando (`cvt tab <alvo>`):
    velocity  velocidade crítica, regressão
    isl       isl_p95, regressão; a faixa sai do corte da predição
    correcao  correção tardia dentro da curva, classificação binária

É a contraparte de train_seq: mesmos alvos e mesmo corte por rota, mas a entrada aqui
é o vetor de features da curva em vez da janela bruta de série temporal.
"""

import numpy as np
import pandas as pd

from curvant.driving.isl import classificar_isl
from curvant.steps import train_correcao

_DIR = {
    'velocity': 'results/tab/velocity',
    'isl':      'results/tab/isl',
    'correcao': 'results/tab/correcao',
}

ALVOS = tuple(_DIR)


def _faixas_isl(valores: np.ndarray) -> np.ndarray:
    """Corta o isl_p95 previsto nas três faixas, com o mesmo critério do rótulo."""
    return np.array([classificar_isl(float(v)) for v in valores])


def run(target: str, features_df: pd.DataFrame, cfg: dict, plot: bool) -> None:
    if target not in _DIR:
        raise ValueError(f"Alvo '{target}' não existe em `cvt tab`. Use um de: {list(ALVOS)}")

    if target == 'correcao':
        train_correcao.classicos(features_df, cfg, plot)
    elif target == 'isl':
        _isl(features_df, cfg, plot)
    else:
        _velocity(features_df, cfg, plot)


def _isl(features_df, cfg, plot) -> None:
    from curvant.models import treinar_regressao

    ml = cfg['ml']
    df = features_df.dropna(subset=['isl_p95'])
    faixas = pd.Series(_faixas_isl(df['isl_p95'].values)).value_counts()
    print(f"  Faixas medidas - baixo: {faixas.get('baixo', 0)} | "
          f"medio: {faixas.get('medio', 0)} | alto: {faixas.get('alto', 0)}")

    treinar_regressao(
        df,
        target='isl_p95',
        plot=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        outdir=_DIR['isl'],
        unidade='',
        classes_derivadas=_faixas_isl,
    )


def _velocity(features_df, cfg, plot) -> None:
    from curvant.models import treinar_regressao

    ml = cfg['ml']
    treinar_regressao(
        features_df.dropna(subset=['v_critica']),
        target='v_critica',
        plot=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        outdir=_DIR['velocity'],
        unidade='km/h',
    )
