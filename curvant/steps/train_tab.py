"""
Representação tabular: modelos sobre as features agregadas por curva (features_df).

Um alvo por subcomando (`cvt tab <alvo>`):
    risk      - Segura/Risco (manobra_combinado_curva) + um modelo por critério
    isl       - faixa de ISL (baixo/medio/alto), classificação multiclasse
    velocity  - velocidade crítica (v_critica), regressão

É a contraparte de train_seq: mesmos alvos, mesma divisão por rota, mas a entrada aqui é o
vetor de features da curva, não a janela bruta de série temporal.
"""

import pandas as pd

from curvant.steps import train_risk

_ISL_ENCODE = {'baixo': 0, 'medio': 1, 'alto': 2}

_DIR = {
    'risk':     'results/tab/risk',
    'isl':      'results/tab/isl',
    'velocity': 'results/tab/velocity',
}

ALVOS = tuple(_DIR)


def run(target: str, features_df: pd.DataFrame, cfg: dict, plot: bool) -> None:
    if target not in _DIR:
        raise ValueError(f"Alvo '{target}' não existe em `cvt tab`. Use um de: {list(ALVOS)}")

    if target == 'risk':
        _risk(features_df, cfg, plot)
    elif target == 'isl':
        _isl(features_df, cfg, plot)
    else:
        _velocity(features_df, cfg, plot)


def _risk(features_df, cfg, plot) -> None:
    train_risk.classicos(features_df, cfg, plot)
    print("\n[tab risk] Um modelo por critério de risco...")
    train_risk.criterios_separados(features_df, cfg, plot)


def _isl(features_df, cfg, plot) -> None:
    from curvant.models import aplicar_modelos_ml

    ml = cfg['ml']
    df = features_df.copy()
    if not pd.api.types.is_numeric_dtype(df['isl_class']):
        df['isl_class'] = df['isl_class'].map(_ISL_ENCODE)
    df = df.dropna(subset=['isl_class'])

    dist = df['isl_class'].value_counts().sort_index()
    print(f"  Faixas - baixo: {dist.get(0, 0)} | medio: {dist.get(1, 0)} | alto: {dist.get(2, 0)}")

    aplicar_modelos_ml(
        df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=ml.get('pca_n_components'),
        target='isl_class',
        f1_average='macro',          # 3 classes: macro não favorece a faixa majoritária
        labels=['baixo', 'medio', 'alto'],
        outdir=_DIR['isl'],
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
