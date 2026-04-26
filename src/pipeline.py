"""
CurvantML — Etapas do pipeline de experimentos.
Funções reutilizáveis por scripts CLI e pelo app Streamlit.
"""

import os

import numpy as np
import pandas as pd

from src.characterization import caracterizar_conducao
from src.curve_detection import detectar_curvas, identificar_trechos_curvos
from src.features import extrair_features
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
    """Etapa 4: classifica janelas de 10s com a taxonomia explícita de risco."""
    da = cfg['driving_analysis']

    partes = []
    for traj in dfs_curves['id_route'].unique():
        dt = dfs_curves.query(f'id_route == "{traj}"')
        resultado = caracterizar_conducao(
            dt,
            janela_tempo=da['janela_tempo'],
            var_velocidade_max=da['var_velocidade_max'],
            limiar_accel_lateral=da.get('limiar_accel_lateral', 3.0),
            zz_limiar_bearing=da.get('zigue_zague', {}).get('limiar_bearing', 15.0),
            zz_limiar_accel=da.get('zigue_zague', {}).get('limiar_accel_lateral', 0.3),
            zz_min_mudancas=da.get('zigue_zague', {}).get('min_mudancas', 3),
        )
        if not resultado.empty:
            partes.append(resultado)

    df_analysis = pd.concat(partes, ignore_index=True)

    n_perigosa = df_analysis['manobra_combinado'].sum()
    n_segura   = (~df_analysis['manobra_combinado']).sum()
    n_accel    = df_analysis['manobra_accel'].sum()
    n_lateral  = df_analysis['manobra_lateral'].sum()
    n_zz       = df_analysis['manobra_ziguezague'].sum()
    print(f"  Janelas — Perigosa: {n_perigosa} | Segura: {n_segura}")
    print(f"  Critérios — Accel: {n_accel} | Lateral: {n_lateral} | ZZ: {n_zz}")

    return df_analysis


def etapa_features(df_analysis: pd.DataFrame, cfg: dict, modo: str = 'modo1') -> pd.DataFrame:
    """
    Etapas 5-6: extrai features e alvos por curva.

    modo='modo1' — inclui F4 (geometria da curva seguinte)
    modo='modo2' — zera F4 e modo_rota_conhecida=0
    """
    df_analysis = df_analysis.copy()
    for col in ['manobra_accel', 'manobra_lateral', 'manobra_ziguezague', 'manobra_combinado']:
        if col in df_analysis.columns:
            df_analysis[col] = df_analysis[col].astype(int)

    dfs_trechos = identificar_trechos_curvos(df_analysis)
    features_df = extrair_features(
        dfs_trechos,
        janela_tempo=cfg['features']['janela_tempo'],
        janela_distancia=cfg['features'].get('janela_distancia'),
    )

    if modo == 'modo2':
        for col in ['f4_raio_min', 'f4_raio_mean', 'f4_dnit_num']:
            if col in features_df.columns:
                features_df[col] = np.nan
        features_df['modo_rota_conhecida'] = 0

    n_perigosa = features_df['manobra_combinado_curva'].sum()
    n_segura   = (features_df['manobra_combinado_curva'] == 0).sum()
    print(f"  {len(features_df)} amostras — Perigosa: {n_perigosa} | Segura: {n_segura}")

    crit_cols = {
        'manobra_accel_curva':      'Aceleração anormal',
        'manobra_lateral_curva':    'Direção perigosa',
        'manobra_ziguezague_curva': 'Zigue-zague',
    }
    tab = features_df.groupby('manobra_combinado_curva')[list(crit_cols.keys())].sum().rename(columns=crit_cols)
    tab.index = tab.index.map({0: 'Segura', 1: 'Perigosa'})
    tab.insert(0, 'Condução', features_df.groupby('manobra_combinado_curva').size().rename({0: 'Segura', 1: 'Perigosa'}))
    tab.index.name = 'Manobra'
    print(tab.to_string())
    os.makedirs('results', exist_ok=True)
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


def etapa_ml_otimizado(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Treina modelos clássicos com Optuna (XGB + RF) e split por rota."""
    from src.models import aplicar_modelos_ml_otimizados

    ml  = cfg['ml']
    opt = cfg.get('optuna', {})
    resultados = aplicar_modelos_ml_otimizados(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        n_trials_xgb=opt.get('n_trials', 50),
        n_trials_rf=opt.get('n_trials', 30),
    )
    print(resultados.to_string(index=False))
    return resultados


def _etapa_regressao(features_df: pd.DataFrame, cfg: dict, plot: bool, target: str) -> pd.DataFrame:
    """Etapa genérica de regressão para qualquer target contínuo."""
    from src.models import treinar_regressao

    ml = cfg['ml']
    resultados = treinar_regressao(
        features_df,
        target=target,
        plot=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=ml.get('pca_n_components'),
        cap_percentil=ml.get('isl_max_cap_percentil'),
    )
    print(resultados.to_string(index=False))
    return resultados


def etapa_accel_regressao(features_df, cfg, plot):
    """P1 — Regressão aceleração lateral (curve_accel_y_max, sensor direto, sem assumir μ)."""
    print("\n  Regressão — curve_accel_y_max")
    r1 = _etapa_regressao(features_df, cfg, plot, target='curve_accel_y_max')
    print("\n  Regressão — curve_abs_accel_max")
    r2 = _etapa_regressao(features_df, cfg, plot, target='curve_abs_accel_max')
    return r1, r2


def etapa_manobra_velocidade(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """P2 — Classificação por velocidade de entrada vs. velocidade segura para o raio."""
    from src.models import aplicar_modelos_ml

    df_v = features_df.dropna(subset=['manobra_velocidade']).copy()
    df_v['manobra_velocidade'] = df_v['manobra_velocidade'].astype(int)

    n_perigosa = df_v['manobra_velocidade'].sum()
    n_segura   = (df_v['manobra_velocidade'] == 0).sum()
    print(f"  v_entry label — Perigosa (v>v_safe): {n_perigosa} | Segura: {n_segura}")

    ml = cfg['ml']
    resultados = aplicar_modelos_ml(
        df_v,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=ml.get('pca_n_components'),
        target='manobra_velocidade',
        f1_average='macro',
    )
    print(resultados.to_string(index=False))
    return resultados


def etapa_isl_modelo(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """
    Etapa ISL: treina modelos para classificar o Índice de Segurança Lateral
    antes de o veículo entrar na curva.

    Target: ``isl_class`` — 3 classes ordinais (baixo / medio / alto).
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
        isl_max_cap_percentil=ml.get('isl_max_cap_percentil'),
    )
    print(resultados.to_string(index=False))
    return resultados


def etapa_pytorch(features_df: pd.DataFrame, cfg: dict) -> None:
    """Treina o MLP multi-tarefa PyTorch com split por id_route."""
    from src.models_pytorch import treinar_multitask_mlp

    nn_cfg = cfg.get('neural_networks', {})
    pt_cfg = nn_cfg.get('multitask_mlp', {})

    treinar_multitask_mlp(
        features_df,
        epochs=pt_cfg.get('epochs', 100),
        batch_size=pt_cfg.get('batch_size', 32),
        lr=pt_cfg.get('lr', 1e-3),
        test_size=cfg['ml']['test_size'],
        random_state=cfg['ml']['random_state'],
    )


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

    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=0,
    )

    print("\n  MLP Keras...")
    mlp_cfg = nn['mlp_keras']
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_dados_keras(
        features_df, val_size=mlp_cfg['validation_split'],
    )
    modelo = construir_mlp_keras(X_train.shape[1])
    history = modelo.fit(
        X_train, y_train,
        epochs=mlp_cfg['epochs'],
        batch_size=mlp_cfg['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
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
