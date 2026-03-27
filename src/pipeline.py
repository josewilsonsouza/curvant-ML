"""
CurvantML — Etapas do pipeline de experimentos.
Funções reutilizáveis por scripts CLI e pelo app Streamlit.
"""

import pandas as pd

from src.curve_detection import detectar_curvas, identificar_trechos_curvos
from src.driving_analysis import detectar_conducao_perigosa
from src.features import extrair_features
from graphics.visualization import plotar_trajeto_conducao
from utils.data import contar_curvas


def etapa_curvas(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Etapa 2: detecta curvas em cada trajeto e concatena."""
    sigma = cfg['curve_detection']['sigma']
    limite_raio = cfg['curve_detection']['limite_raio']

    partes = []
    for traj in df['id_route'].unique():
        dt = df.query(f'id_route == "{traj}"')
        try:
            partes.append(detectar_curvas(dt, sigma=sigma, limite_raio=limite_raio))
        except Exception as e:
            print(f"  [AVISO] Erro ao processar {traj}: {e}")

    dfs_curves = pd.concat(partes, ignore_index=True)

    contagem = contar_curvas(dfs_curves)
    print(f"  {sum(contagem.values())} curvas detectadas em {len(contagem)} trajetos")
    return dfs_curves


def etapa_analise_conducao(dfs_curves: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Etapa 4: classifica janelas de 10s como Segura/Perigosa."""
    da = cfg['driving_analysis']

    partes = []
    for traj in dfs_curves['id_route'].unique():
        dt = dfs_curves.query(f'id_route == "{traj}"')
        partes.append(detectar_conducao_perigosa(
            dt,
            janela_tempo=da['janela_tempo'],
            var_velocidade_max=da['var_velocidade_max'],
            velocidade_max_direcao=da['velocidade_max_direcao'],
            angulo_max_direcao=da['angulo_max_direcao'],
        ))

    df_analysis = pd.concat(partes, ignore_index=True)

    n_perigosa = (df_analysis['conducao'] == 'Perigosa').sum()
    n_segura   = (df_analysis['conducao'] == 'Segura').sum()
    print(f"  Janelas — Perigosa: {n_perigosa} | Segura: {n_segura}")

    if plot:
        for traj in df_analysis['id_route'].unique():
            plotar_trajeto_conducao(df_analysis.query(f'id_route == "{traj}"'))

    return df_analysis


def etapa_features(df_analysis: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Etapas 5-6: identifica trechos curvos e extrai features estatísticas."""
    df_analysis = df_analysis.copy()
    df_analysis[['aceleracao_anormal', 'direcao_perigosa', 'zigue_zague']] = (
        df_analysis[['aceleracao_anormal', 'direcao_perigosa', 'zigue_zague']].astype(int)
    )

    dfs_trechos = identificar_trechos_curvos(df_analysis)
    features_df = extrair_features(dfs_trechos, janela_tempo=cfg['features']['janela_tempo'])

    n_perigosa = features_df['manobra'].sum()
    n_segura   = (features_df['manobra'] == 0).sum()
    print(f"  {len(features_df)} amostras — Perigosa: {n_perigosa} | Segura: {n_segura}")

    crit_cols = {
        'manobra_accel_perigo': 'Aceleração anormal',
        'manobra_dir_perigosa': 'Direção perigosa',
        'manobra_zigue_zague':  'Zigue-zague',
    }
    tab = features_df.groupby('manobra')[list(crit_cols.keys())].sum().rename(columns=crit_cols)
    tab.index = tab.index.map({0: 'Segura', 1: 'Perigosa'})
    tab.insert(0, 'Condução', features_df.groupby('manobra').size().rename({0: 'Segura', 1: 'Perigosa'}))
    tab.index.name = 'Manobra'

    print(tab.to_string())
    tab.to_latex('results/tab_result.tex', index=True)

    return features_df


def etapa_ml_classico(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Etapa 7: treina e avalia modelos clássicos com validação cruzada k-fold."""
    from src.models import aplicar_modelos_ml

    ml = cfg['ml']
    pca = ml.get('pca_n_components')
    if pca is not None:
        print(f"  PCA ativado: n_components={pca}")
    resultados = aplicar_modelos_ml(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=pca,
    )
    print(resultados.to_string(index=False))
    return resultados


def etapa_isl_modelo(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """
    Etapa ISL: treina modelos para prever o Índice de Segurança Lateral
    antes de o veículo entrar na curva.

    Usa as features da janela pré-curva (já extraídas) para prever ``isl_alto``
    (1 = ISL ≥ 0.8 — risco lateral alto).
    """
    from src.models import treinar_modelo_isl
    from src.isl import resumo_isl

    resumo_isl(features_df.dropna(subset=['isl_max']))

    ml = cfg['ml']
    resultados = treinar_modelo_isl(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=ml.get('pca_n_components'),
    )
    print(resultados.to_string(index=False))
    return resultados


def etapa_mlp_sklearn(features_df: pd.DataFrame, cfg: dict) -> None:
    """MLP sklearn com GridSearchCV."""
    from src.models import treinar_mlp_sklearn

    ml = cfg['ml']
    treinar_mlp_sklearn(
        features_df,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        pca_n_components=ml.get('pca_n_components'),
    )


def etapa_keras(features_df: pd.DataFrame, cfg: dict, plot: bool) -> None:
    """Treina MLP Keras, GRU e LSTM."""
    from src.models import (
        construir_gru, construir_lstm, construir_mlp_keras,
        preparar_dados_gru, preparar_dados_keras, preparar_dados_lstm,
    )
    from graphics.visualization import plotar_curva_treinamento

    nn = cfg['neural_networks']

    print("\n  MLP Keras...")
    mlp_cfg = nn['mlp_keras']
    X_train, X_test, y_train, y_test = preparar_dados_keras(features_df)
    modelo = construir_mlp_keras(X_train.shape[1])
    history = modelo.fit(
        X_train, y_train,
        epochs=mlp_cfg['epochs'],
        batch_size=mlp_cfg['batch_size'],
        validation_split=mlp_cfg['validation_split'],
        verbose=0,
    )
    _, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"  MLP Keras — Acurácia teste: {acc:.4f}")
    if plot:
        plotar_curva_treinamento(history)

    print("\n  GRU...")
    gru_cfg = nn['gru']
    X_train, X_test, y_train, y_test, n_feat = preparar_dados_gru(
        features_df, janela_tempo=gru_cfg['janela_tempo']
    )
    modelo = construir_gru(gru_cfg['janela_tempo'], n_feat)
    history = modelo.fit(
        X_train, y_train,
        epochs=gru_cfg['epochs'],
        batch_size=gru_cfg['batch_size'],
        validation_split=gru_cfg['validation_split'],
        verbose=0,
    )
    _, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"  GRU — Acurácia teste: {acc:.4f}")
    if plot:
        plotar_curva_treinamento(history)

    print("\n  LSTM...")
    lstm_cfg = nn['lstm']
    X_train, X_test, y_train, y_test, n_feat = preparar_dados_lstm(
        features_df, janela_tempo=lstm_cfg['janela_tempo']
    )
    modelo = construir_lstm(lstm_cfg['janela_tempo'], n_feat)
    history = modelo.fit(
        X_train, y_train,
        epochs=lstm_cfg['epochs'],
        batch_size=lstm_cfg['batch_size'],
        validation_split=lstm_cfg['validation_split'],
        verbose=0,
    )
    _, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"  LSTM — Acurácia teste: {acc:.4f}")
    if plot:
        plotar_curva_treinamento(history)
