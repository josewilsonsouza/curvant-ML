"""
Alvo de correção tardia: o motorista freou forte depois de já estar na curva.

    classicos            modelos clássicos, é o que a CLI chama
    otimizado            o mesmo com tuning Optuna, fora da CLI
    importancia_features importância XGBoost por ganho, fora da CLI
"""

import os

_DIR = 'results/tab/correcao'


def _filtrar_indefinidas(features_df, coluna: str = 'correcao_indefinida_curva'):
    """Remove curvas na faixa de histerese do limiar, onde o rótulo é decidido por ruído."""
    if coluna not in features_df.columns:
        return features_df
    filtrado = features_df[features_df[coluna] == 0]
    n_desc = len(features_df) - len(filtrado)
    if n_desc:
        print(f"  Histerese: {n_desc} curvas indefinidas descartadas ({len(filtrado)} restantes)")
    return filtrado


def classicos(features_df, cfg: dict, plot: bool):
    """Modelos clássicos para o alvo de correção tardia, validação por rota."""
    from curvant.models import aplicar_modelos_ml

    features_df = _filtrar_indefinidas(features_df)
    n_pos = int(features_df['correcao_tardia_curva'].sum())
    print(f"  Positivos: {n_pos}/{len(features_df)} ({100*n_pos/len(features_df):.1f}%)")

    ml  = cfg['ml']
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
        target='correcao_tardia_curva',
        f1_average='binary',       # F1 da classe positiva (corrigiu)
        labels=['Antecipou', 'Corrigiu'],
        outdir=_DIR,
    )


def otimizado(features_df, cfg: dict, plot: bool):
    """O mesmo alvo com Optuna (XGB + RF + LogReg) e split por rota."""
    from curvant.models import aplicar_modelos_ml_otimizados

    features_df = _filtrar_indefinidas(features_df)
    ml  = cfg['ml']
    opt = cfg.get('optuna', {})
    return aplicar_modelos_ml_otimizados(
        features_df,
        plot_cm=plot,
        random_state=ml['random_state'],
        test_size=ml['test_size'],
        cv_folds=ml['cv_folds'],
        target='correcao_tardia_curva',
        n_trials_xgb=opt.get('n_trials_xgb', opt.get('n_trials', 50)),
        n_trials_rf=opt.get('n_trials_rf', opt.get('n_trials', 30)),
        n_trials_lr=opt.get('n_trials_lr', 20),
        timeout=opt.get('timeout'),
        outdir=_DIR,
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
    target = 'correcao_tardia_curva'
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
    os.makedirs(_DIR, exist_ok=True)
    caminho = os.path.join(_DIR, 'importancia_features.pdf')
    fig.savefig(caminho, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Importância das features salva em {caminho}")
    print("  Top 10 features:")
    for i in range(min(10, len(top_feats))):
        print(f"    {i+1:2d}. {top_feats[i]:<40s} {top_vals[i]:.4f}")
