"""
CurvantML - Etapas do pipeline de experimentos.
Funções reutilizáveis por scripts CLI e pelo app Streamlit.

Cada flag de alvo escreve seus resultados numa subpasta própria de results/.
"""

import os

import numpy as np
import pandas as pd

from curvant.driving.risk_measures import caracterizar_conducao
from curvant.driving.curve_detection import detectar_curvas, identificar_trechos_curvos, contar_curvas
from curvant.driving.features import extrair_features, configurar_features_desativadas
from curvant.models.montecarlo import aplicar_mc_features

_DIR_RISCO       = 'results/risco'
_DIR_ISL         = 'results/isl'
_DIR_VELOCIDADE  = 'results/velocidade'
_DIR_MULTITAREFA = 'results/multitarefa'


def etapa_curvas(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Etapa 2: detecta curvas em cada trajeto e concatena."""
    sigma = cfg['curve_detection']['sigma']
    limite_raio = cfg['curve_detection']['limite_raio']
    sigma_gps = cfg['curve_detection'].get('sigma_gps', 0.0)

    partes = []
    for traj, dt in df.groupby('id_route', sort=False):
        try:
            partes.append(detectar_curvas(dt, sigma=sigma, limite_raio=limite_raio, sigma_gps=sigma_gps))
        except Exception as e:
            print(f"  [AVISO] Erro ao processar {traj}: {e}")

    dfs_curves = pd.concat(partes, ignore_index=True)

    contagem = contar_curvas(dfs_curves)
    print(f"  {sum(contagem.values())} curvas detectadas em {len(contagem)} trajetos")
    return dfs_curves


def etapa_analise_conducao(dfs_curves: pd.DataFrame, cfg: dict, mostrar_risco: bool = True) -> pd.DataFrame:
    """Etapa 4: classifica janelas com a taxonomia explícita de risco."""
    da = cfg['risk_measures']

    partes = []
    for _, dt in dfs_curves.groupby('id_route', sort=False):
        resultado = caracterizar_conducao(
            dt,
            kamm_alpha=da.get('kamm_alpha', 0.7),
            limiar_accel_lateral=da.get('limiar_accel_lateral', 2.0),
            zz_limiar_bearing=da.get('zigue_zague', {}).get('limiar_bearing', 15.0),
            zz_limiar_accel=da.get('zigue_zague', {}).get('limiar_ctp', 0.3),
            zz_min_mudancas=da.get('zigue_zague', {}).get('min_mudancas', 3),
        )
        if not resultado.empty:
            partes.append(resultado)

    df_analysis = pd.concat(partes, ignore_index=True)

    if mostrar_risco:
        n_perigosa = df_analysis['manobra_combinado'].sum()
        n_segura   = (~df_analysis['manobra_combinado']).sum()
        n_accel    = df_analysis['manobra_accel'].sum()
        n_lateral  = df_analysis['manobra_lateral'].sum()
        n_zz       = df_analysis['manobra_ziguezague'].sum()
        print(f"  Janelas — Risco: {n_perigosa} | Segura: {n_segura}")
        print(f"  Critérios — Accel: {n_accel} | Lateral: {n_lateral} | ZZ: {n_zz}")

    return df_analysis


def etapa_features(df_analysis: pd.DataFrame, cfg: dict, mostrar_risco: bool = True) -> pd.DataFrame:
    """
    Etapas 5-6: extrai features e alvos por curva.

    Assume que a rota é pré-conhecida (ex: frota com rotas fixas, navegação GPS)
    e que o trajeto completo (lat, lon) está disponível.
    """
    df_analysis = df_analysis.copy()
    for col in ['manobra_accel', 'manobra_lateral', 'manobra_ziguezague', 'manobra_combinado']:
        if col in df_analysis.columns:
            df_analysis[col] = df_analysis[col].astype(int)

    dfs_trechos = identificar_trechos_curvos(df_analysis)
    ft = cfg['features']
    configurar_features_desativadas(ft.get('desativar'))
    features_df = extrair_features(
        dfs_trechos,
        janela_distancia=ft.get('janela_distancia'),
        janela_acel_confort=ft.get('janela_acel_confort', 2.5),
        janela_distancia_min=ft.get('janela_distancia_min', 50.0),
        janela_distancia_max=ft.get('janela_distancia_max', 400.0),
        lead_gap=ft.get('lead_gap', 0.0),
    )

    if mostrar_risco:
        n_perigosa = features_df['manobra_combinado_curva'].sum()
        n_segura   = (features_df['manobra_combinado_curva'] == 0).sum()
        print(f"  {len(features_df)} amostras — Risco: {n_perigosa} | Segura: {n_segura}")
    else:
        print(f"  {len(features_df)} amostras (curvas)")

    mc_cfg = cfg.get('montecarlo', {})
    rnd    = cfg.get('ml', {}).get('random_state', 42)
    features_df = aplicar_mc_features(
        features_df, df_analysis,
        n_sim=mc_cfg.get('n_sim', 1000),
        k_ultimos=mc_cfg.get('k_ultimos', 10),
        sigma_min=mc_cfg.get('sigma_min', 0.5),
        random_state=rnd,
    )

    return features_df


def _tabela_criterios_risco(features_df: pd.DataFrame, outdir: str) -> None:
    """Tabela da distribuição dos 3 critérios de risco por curva (alvo manobra)."""
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
    os.makedirs(outdir, exist_ok=True)
    tab.to_latex(
        os.path.join(outdir, 'criterios_risco.tex'),
        index=True,
        caption='Distribuição dos critérios de risco por curva — contagem de curvas Segura e Risco em que cada critério foi ativado.',
        label='tab:criterios_risco',
        position='h',
        column_format='lcccc',
    )


def etapa_criterios_separados(features_df: pd.DataFrame, cfg: dict) -> None:
    """Treina XGBoost para cada critério de risco individualmente e compara F1."""
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import cross_val_score, GroupKFold
    from xgboost import XGBClassifier
    from curvant.models import _split_por_rota

    ml = cfg['ml']
    criterios = {
        'manobra_combinado_curva':  'Combinado',
        'manobra_accel_curva':      'Kamm (aceleração total)',
        'manobra_lateral_curva':    'Lateral (accel_y + DNIT)',
        'manobra_ziguezague_curva': 'Zigue-zague',
    }

    feature_cols = None

    print(f"\n  {'Critério':<30}  {'Pos':>5}  {'Neg':>5}  {'scale_w':>8}  {'CV F1':>7}  {'Teste F1':>9}  {'Teste Acc':>10}")
    print(f"  {'-'*80}")

    for target, label in criterios.items():
        if target not in features_df.columns:
            continue

        X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
            features_df, target=target,
            test_size=ml['test_size'], random_state=ml['random_state'],
        )

        if feature_cols is None:
            from curvant.driving.features import colunas_features
            feature_cols = colunas_features(features_df)

        pos_train = int(y_train.sum())
        neg_train = len(y_train) - pos_train
        spw = round(neg_train / pos_train, 1) if pos_train > 0 else 1.0

        clf = XGBClassifier(
            n_estimators=200, random_state=ml['random_state'],
            eval_metric='logloss', scale_pos_weight=spw,
        )
        cv    = GroupKFold(n_splits=ml['cv_folds'])
        cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, groups=groups_train, scoring='f1').mean()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        pos = int(y_test.sum())
        neg = len(y_test) - pos

        print(f"  {label:<30}  {pos:>5}  {neg:>5}  {spw:>8.1f}  {cv_f1:>7.3f}  {f1:>9.3f}  {acc:>10.3f}")

        importancias = clf.feature_importances_
        top5_idx = importancias.argsort()[::-1][:5]
        top5 = [(feature_cols[i], importancias[i]) for i in top5_idx]
        print(f"    Top features: " + "  |  ".join(f"{n} ({v:.3f})" for n, v in top5))


def etapa_ml_classico(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Modelos clássicos para o alvo de risco (manobra), validação por rota."""
    from curvant.models import aplicar_modelos_ml

    _tabela_criterios_risco(features_df, _DIR_RISCO)
    ml = cfg['ml']
    pca = ml.get('pca_n_components')
    if pca is not None:
        print(f"  PCA ativado: n_components={pca}")
    return aplicar_modelos_ml(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=pca,
        outdir=_DIR_RISCO,
    )


def etapa_ml_otimizado(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """Modelos de risco com Optuna (XGB + RF) e split por rota."""
    from curvant.models import aplicar_modelos_ml_otimizados

    _tabela_criterios_risco(features_df, _DIR_RISCO)
    ml  = cfg['ml']
    opt = cfg.get('optuna', {})
    return aplicar_modelos_ml_otimizados(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        n_trials_xgb=opt.get('n_trials_xgb', opt.get('n_trials', 50)),
        n_trials_rf=opt.get('n_trials_rf', opt.get('n_trials', 30)),
        n_trials_lr=opt.get('n_trials_lr', 20),
        timeout=opt.get('timeout'),
        outdir=_DIR_RISCO,
    )



def etapa_isl_modelo(features_df: pd.DataFrame, cfg: dict, plot: bool) -> pd.DataFrame:
    """
    Classifica o Índice de Segurança Lateral (ISL) antes de entrar na curva.
    Target: ``isl_class`` — 3 classes ordinais (baixo / medio / alto).
    """
    from curvant.models import treinar_modelo_isl
    from curvant.driving.isl import resumo_isl

    resumo_isl(features_df.dropna(subset=['isl_max']))

    ml = cfg['ml']
    return treinar_modelo_isl(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        pca_n_components=ml.get('pca_n_components'),
        isl_max_cap_percentil=ml.get('isl_max_cap_percentil'),
        outdir=_DIR_ISL,
    )


def etapa_baseline_fisico(features_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Baseline físico (Monte Carlo / fórmula do ISL) avaliado no split por rota."""
    from curvant.models import avaliar_baseline_fisico

    ml = cfg['ml']
    return avaliar_baseline_fisico(
        features_df,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        isl_max_cap_percentil=ml.get('isl_max_cap_percentil'),
        outdir=_DIR_ISL,
    )


def etapa_regressao_ts(df_analysis: pd.DataFrame, features_df: pd.DataFrame, cfg: dict, plot: bool) -> None:
    """Previsão da velocidade crítica (e ISL derivado) sobre a série temporal pré-curva."""
    from curvant.models import treinar_regressao_ts
    treinar_regressao_ts(df_analysis, features_df, cfg, plot=plot, outdir=_DIR_VELOCIDADE)


def etapa_pytorch(features_df: pd.DataFrame, cfg: dict, plot: bool = True) -> None:
    """Treina o MLP multi-tarefa PyTorch com split por id_route."""
    from curvant.models import treinar_multitask_mlp

    pt_cfg = cfg.get('multitarefa', {})

    treinar_multitask_mlp(
        features_df,
        epochs=pt_cfg.get('epochs', 100),
        batch_size=pt_cfg.get('batch_size', 32),
        lr=pt_cfg.get('lr', 1e-3),
        dropout=pt_cfg.get('dropout', 0.5),
        weight_decay=pt_cfg.get('weight_decay', 0.01),
        test_size=cfg['ml']['test_size'],
        random_state=cfg['ml']['random_state'],
        outdir=_DIR_MULTITAREFA,
        plot=plot,
    )


def etapa_importancia_features(features_df: pd.DataFrame, cfg: dict, top_n: int = 40) -> None:
    """Treina XGBoost, plota importância das features (gain) e faz ablação das mc_p_*."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier
    from curvant.models import _split_por_rota, colunas_features

    ml = cfg['ml']
    target = 'manobra_combinado_curva'
    if target not in features_df.columns:
        print("  [importancia] target não encontrado, pulando.")
        return

    MC_COLS = ['mc_p_baixo', 'mc_p_medio', 'mc_p_alto']

    feature_cols = colunas_features(features_df)
    X_train, X_test, y_train, y_test, _ = _split_por_rota(
        features_df, target=target,
        test_size=ml['test_size'], random_state=ml['random_state'],
    )

    mc_presentes = [c for c in MC_COLS if c in feature_cols]
    idx_full   = list(range(len(feature_cols)))
    idx_sem_mc = [i for i, c in enumerate(feature_cols) if c not in MC_COLS]

    def _treinar_avaliar(X_tr, X_te, y_tr, y_te, idx):
        clf = XGBClassifier(n_estimators=200, random_state=ml['random_state'], eval_metric='logloss')
        clf.fit(X_tr[:, idx], y_tr)
        pred  = clf.predict(X_te[:, idx])
        proba = clf.predict_proba(X_te[:, idx])[:, 1]
        return clf, {
            'acc': accuracy_score(y_te, pred),
            'f1':  f1_score(y_te, pred, zero_division=0),
            'auc': roc_auc_score(y_te, proba),
        }

    clf_full, m_full   = _treinar_avaliar(X_train, X_test, y_train, y_test, idx_full)
    _,        m_sem_mc = _treinar_avaliar(X_train, X_test, y_train, y_test, idx_sem_mc)

    print(f"\n  Ablação das features Monte Carlo ({', '.join(mc_presentes) or 'nenhuma encontrada'}):")
    print(f"  {'Métrica':<8}  {'Com MC':>8}  {'Sem MC':>8}  {'Delta':>8}")
    print(f"  {'-'*40}")
    for metrica in ('acc', 'f1', 'auc'):
        v_full  = m_full[metrica]
        v_semmc = m_sem_mc[metrica]
        delta   = v_full - v_semmc
        sinal   = '+' if delta >= 0 else ''
        print(f"  {metrica:<8}  {v_full:>8.4f}  {v_semmc:>8.4f}  {sinal}{delta:>7.4f}")

    importances = clf_full.feature_importances_
    indices     = importances.argsort()[::-1][:top_n]
    top_feats   = [feature_cols[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    colors = ['tab:orange' if f in MC_COLS else 'tab:blue' for f in top_feats[::-1]]
    ax.barh(range(len(top_feats)), top_vals[::-1], color=colors)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], fontsize=8)
    ax.set_xlabel('Importância (gain)')
    ax.set_title(f'XGBoost — Top {top_n} features ({target})  [laranja = Monte Carlo]')
    fig.tight_layout()
    os.makedirs(_DIR_RISCO, exist_ok=True)
    caminho = os.path.join(_DIR_RISCO, 'importancia_features.pdf')
    fig.savefig(caminho, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Importância das features salva em {caminho}")
    print(f"  Top 10 features:")
    for i in range(min(10, len(top_feats))):
        mc_flag = ' [MC]' if top_feats[i] in MC_COLS else ''
        print(f"    {i+1:2d}. {top_feats[i]:<40s} {top_vals[i]:.4f}{mc_flag}")
