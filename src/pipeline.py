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
            janela_aproximacao=da.get('janela_aproximacao', 5),
            kamm_alpha=da.get('kamm_alpha', 0.7),
            limiar_accel_lateral=da.get('limiar_accel_lateral', 2.0),
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
    print(f"  Janelas — Risco: {n_perigosa} | Segura: {n_segura}")
    print(f"  Critérios — Accel: {n_accel} | Lateral: {n_lateral} | ZZ: {n_zz}")

    return df_analysis


def etapa_features(df_analysis: pd.DataFrame, cfg: dict, modo: str = 'modo1') -> pd.DataFrame:
    """
    Etapas 5-6: extrai features e alvos por curva.

    modo='modo1' — rota pré-conhecida (ex: frota com rotas fixas, navegação GPS).
        O trajeto completo (lat, lon) está disponível antes da viagem.
        B-spline ajustada sobre todos os pontos é legítima — o sistema pode
        "ver" a curva à frente porque a rota já está mapeada.
        Features disponíveis: F1 + F2 + F3 + F4 + F5.

    modo='modo2' — rota desconhecida, somente dados OBD acumulados até o instante atual.
        Apenas pontos já percorridos estão disponíveis para o modelo.
        F4 (geometria real da curva) é zerrada porque exige conhecimento futuro.
        F3 (raio estimado via B-spline) também é zerrada: a B-spline foi ajustada
        sobre o trajeto inteiro (incluindo pontos além da posição atual), o que
        constituiria look-ahead num dispositivo OBD em tempo real.
        Features disponíveis: F1 + F2 (sem raio) + F5.
    """
    df_analysis = df_analysis.copy()
    for col in ['manobra_accel', 'manobra_lateral', 'manobra_ziguezague', 'manobra_combinado']:
        if col in df_analysis.columns:
            df_analysis[col] = df_analysis[col].astype(int)

    dfs_trechos = identificar_trechos_curvos(df_analysis)
    ft = cfg['features']
    features_df = extrair_features(
        dfs_trechos,
        janela_tempo=ft['janela_tempo'],
        janela_distancia=ft.get('janela_distancia'),
        janela_acel_confort=ft.get('janela_acel_confort', 2.5),
        janela_distancia_min=ft.get('janela_distancia_min', 50.0),
        janela_distancia_max=ft.get('janela_distancia_max', 400.0),
    )

    if modo == 'modo2':
        # F4 — geometria real da curva seguinte (requer rota pré-conhecida)
        _cols_f4 = ['f4_raio_min', 'f4_raio_mean', 'f4_dnit_num']
        # F3 — raio estimado via B-spline global: usa pontos futuros do trajeto,
        #       indisponíveis num cenário de rota desconhecida em tempo real
        _cols_f3 = ['janela_raio_min', 'janela_raio_mean', 'janela_raio_last',
                    'v_entry_sq_over_raio_est']
        for col in _cols_f4 + _cols_f3:
            if col in features_df.columns:
                features_df[col] = np.nan
        features_df['modo_rota_conhecida'] = 0

    n_perigosa = features_df['manobra_combinado_curva'].sum()
    n_segura   = (features_df['manobra_combinado_curva'] == 0).sum()
    print(f"  {len(features_df)} amostras — Risco: {n_perigosa} | Segura: {n_segura}")

    crit_cols = {
        'manobra_accel_curva':      'Aceleração anormal',
        'manobra_lateral_curva':    'Aceleração lateral',
        'manobra_ziguezague_curva': 'Zigue-zague',
    }
    tab = features_df.groupby('manobra_combinado_curva')[list(crit_cols.keys())].sum().rename(columns=crit_cols)
    tab.index = tab.index.map({0: 'Segura', 1: 'Risco'})
    tab.insert(0, 'Total curvas', features_df.groupby('manobra_combinado_curva').size().rename({0: 'Segura', 1: 'Risco'}))
    tab.index.name = 'Classificação'
    print(tab.to_string())
    os.makedirs('results', exist_ok=True)
    tab.to_latex(
        'results/tab_result.tex',
        index=True,
        caption='Distribuição dos critérios de risco por curva — contagem de curvas Segura e Risco em que cada critério foi ativado.',
        label='tab:criterios_risco',
        position='h',
        column_format='lcccc',
    )

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
        n_trials_xgb=opt.get('n_trials_xgb', opt.get('n_trials', 50)),
        n_trials_rf=opt.get('n_trials_rf', opt.get('n_trials', 30)),
        timeout=opt.get('timeout'),
    )
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
    return resultados


def etapa_accel_regressao(features_df, cfg, plot):
    """P1 — Regressão de aceleração dentro da curva. Targets definidos em ml.regression_targets."""
    targets = cfg['ml'].get('regression_targets', ['curve_accel_y_max', 'curve_abs_accel_max'])
    resultados = []
    for target in targets:
        print(f"\n  Regressão — {target}")
        resultados.append(_etapa_regressao(features_df, cfg, plot, target=target))
    return resultados


def etapa_manobra_velocidade(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """P2 — Classificação por velocidade de entrada vs. velocidade segura para o raio."""
    from src.models import aplicar_modelos_ml

    df_v = features_df.dropna(subset=['manobra_velocidade']).copy()
    df_v['manobra_velocidade'] = df_v['manobra_velocidade'].astype(int)

    n_perigosa = df_v['manobra_velocidade'].sum()
    n_segura   = (df_v['manobra_velocidade'] == 0).sum()
    print(f"  v_entry label — Risco (v>v_safe): {n_perigosa} | Segura: {n_segura}")

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
    return resultados


def etapa_regressao_ts(df_analysis: pd.DataFrame, features_df: pd.DataFrame, cfg: dict, plot: bool) -> None:
    """Regressão de série temporal (GRU / LSTM / CNN1D) sobre dados brutos da janela pré-curva."""
    from src.models_ts import treinar_regressao_ts
    treinar_regressao_ts(df_analysis, features_df, cfg, plot=plot)


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
        dropout=pt_cfg.get('dropout', 0.3),
        test_size=cfg['ml']['test_size'],
        random_state=cfg['ml']['random_state'],
    )


def etapa_mlp_sklearn(features_df: pd.DataFrame, cfg: dict) -> None:
    """MLP sklearn com GridSearchCV."""
    from src.models import treinar_mlp_sklearn

    ml     = cfg['ml']
    sk_cfg = cfg.get('neural_networks', {}).get('mlp_sklearn', {})
    treinar_mlp_sklearn(
        features_df,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        pca_n_components=ml.get('pca_n_components'),
        hidden_layer_sizes=sk_cfg.get('hidden_layer_sizes'),
        activation=sk_cfg.get('activation', 'relu'),
        solver=sk_cfg.get('solver', 'adam'),
        alpha=sk_cfg.get('alpha', 0.001),
        max_iter=sk_cfg.get('max_iter', 2000),
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
