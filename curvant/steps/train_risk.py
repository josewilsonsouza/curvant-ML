"""
Funções:
    classicos            - modelos clássicos (LogReg, SVM, RF, XGBoost, MLP...) no alvo combinado
    otimizado            - o mesmo com tuning Optuna (análise complementar)
    criterios_separados  - um modelo por critério de risco (frenagem tardia, zigue-zague)
    importancia_features - importância XGBoost (gain) por feature (análise complementar)
"""

import os

_DIR_RISCO = 'results/tab/risk'


def _filtrar_indefinidas(features_df, coluna: str = 'manobra_indefinida_curva'):
    """Remove curvas na faixa de histerese do limiar, onde o rótulo é decidido por ruído."""
    if coluna not in features_df.columns:
        return features_df
    filtrado = features_df[features_df[coluna] == 0]
    n_desc = len(features_df) - len(filtrado)
    if n_desc:
        print(f"  Histerese: {n_desc} curvas indefinidas descartadas ({len(filtrado)} restantes)")
    return filtrado


def _tabela_criterios_risco(features_df, outdir: str) -> None:
    """Tabela da distribuição dos critérios de risco por curva (alvo manobra)."""
    crit_cols = {
        'manobra_frenagem_curva':   'Frenagem tardia',
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
        column_format='lccc',
    )


def classicos(features_df, cfg: dict, plot: bool):
    """Modelos clássicos para o alvo de risco (manobra), validação por rota."""
    from curvant.models import aplicar_modelos_ml

    features_df = _filtrar_indefinidas(features_df)
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
        f1_average='binary',   # mesma métrica dos critérios: F1 da classe positiva (Risco),
                               # para o combinado ser comparável com Kamm/lateral/zigue-zague
        outdir=_DIR_RISCO,
    )


def otimizado(features_df, cfg: dict, plot: bool):
    """Modelos de risco com Optuna (XGB + RF + LogReg) e split por rota."""
    from curvant.models import aplicar_modelos_ml_otimizados

    features_df = _filtrar_indefinidas(features_df)
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


def criterios_separados(features_df, cfg: dict, plot: bool = True) -> None:
    """Treina todos os modelos clássicos para cada critério de risco individualmente."""
    from curvant.models import aplicar_modelos_ml

    ml = cfg['ml']
    pca = ml.get('pca_n_components')

    criterios = {
        'manobra_frenagem_curva':   ('Frenagem tardia', 'frenagem'),
        'manobra_ziguezague_curva': ('Zigue-zague', 'zigzag'),
    }
    # Cada critério descarta as curvas indefinidas da SUA faixa de histerese;
    # o zigue-zague não tem limiar contínuo, então não filtra nada.
    indef_por_criterio = {
        'manobra_frenagem_curva': 'manobra_frenagem_indef_curva',
    }

    for target, (label, slug) in criterios.items():
        if target not in features_df.columns:
            continue
        df_crit = features_df
        if target in indef_por_criterio:
            df_crit = _filtrar_indefinidas(features_df, indef_por_criterio[target])
        n_pos = int(df_crit[target].sum())
        n_tot = len(df_crit)
        print(f"\n  [{label}]  positivos: {n_pos}/{n_tot} ({100*n_pos/n_tot:.1f}%)")
        aplicar_modelos_ml(
            df_crit,
            plot_cm=plot,
            random_state=ml['random_state'],
            test_size=ml['test_size'],
            cv_folds=ml['cv_folds'],
            pca_n_components=pca,
            target=target,
            f1_average='binary',
            outdir=os.path.join(_DIR_RISCO, slug),
        )


def importancia_features(features_df, cfg: dict, top_n: int = 40) -> None:
    """Treina XGBoost e plota a importância das features por ganho."""
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
    features_df = _filtrar_indefinidas(features_df)

    feature_cols = colunas_features(features_df)
    X_train, X_test, y_train, y_test, _ = _split_por_rota(
        features_df, target=target,
        test_size=ml['test_size'], random_state=ml['random_state'],
    )

    clf = XGBClassifier(n_estimators=200, random_state=ml['random_state'], eval_metric='logloss')
    clf.fit(X_train, y_train)
    pred  = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print()
    print('  XGBoost no conjunto de teste:')
    print(f"  acc {accuracy_score(y_test, pred):.4f} | "
          f"f1 {f1_score(y_test, pred, zero_division=0):.4f} | "
          f"auc {roc_auc_score(y_test, proba):.4f}")

    importances = clf.feature_importances_
    indices     = importances.argsort()[::-1][:top_n]
    top_feats   = [feature_cols[i] for i in indices]
    top_vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    ax.barh(range(len(top_feats)), top_vals[::-1], color='tab:blue')
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], fontsize=8)
    ax.set_xlabel('Importância (gain)')
    ax.set_title(f'XGBoost - Top {top_n} features ({target})')
    fig.tight_layout()
    os.makedirs(_DIR_RISCO, exist_ok=True)
    caminho = os.path.join(_DIR_RISCO, 'importancia_features.pdf')
    fig.savefig(caminho, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Importância das features salva em {caminho}")
    print(f"  Top 10 features:")
    for i in range(min(10, len(top_feats))):
        print(f"    {i+1:2d}. {top_feats[i]:<40s} {top_vals[i]:.4f}")
