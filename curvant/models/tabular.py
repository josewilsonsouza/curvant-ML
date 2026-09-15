import os
import re

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import (
    GroupKFold, GridSearchCV, KFold, StratifiedGroupKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier, XGBRegressor

from curvant.constants import G as _G, MU as _MU
from curvant.driving.features import colunas_features, checar_leakage

# Utilitários internos

def _base_route(id_route: str) -> str:
    """Remove sufixo _p<N> gerado pelo splittar_por_gaps, retornando o ID original da gravação."""
    return re.sub(r'_p\d+$', '', id_route)

def _split_por_rota(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split por id_route (sem amostras da mesma rota em treino e teste).
    Retorna (X_train, X_test, y_train, y_test, groups_train).
    groups_train é usado pelo GroupKFold.
    """
    id_routes  = df['id_route'].astype(str)
    base_rotas = sorted({_base_route(r) for r in id_routes})
    train_base, test_base = train_test_split(
        base_rotas, test_size=test_size, random_state=random_state
    )
    train_base_set = set(train_base)
    test_base_set  = set(test_base)

    feature_cols = colunas_features(df)
    checar_leakage(feature_cols, target)
    cols = feature_cols + [target, 'id_route']

    df_train = (
        df[id_routes.map(_base_route).isin(train_base_set)][cols]
        .dropna(subset=[target])
    )
    df_test = (
        df[id_routes.map(_base_route).isin(test_base_set)][cols]
        .dropna(subset=[target])
    )

    # NaN em features de raio são preenchidas com a mediana
    # do conjunto de treino, semanticamente "raio típico" em vez de 0 (curvatura
    # infinita) que o StandardScaler interpretaria como outlier extremo negativo.
    # Demais NaN (features opcionais ausentes) ficam como 0.
    _raio_cols = [c for c in feature_cols if 'raio' in c]
    _medians   = df_train[_raio_cols].median()
    df_train_filled = df_train[feature_cols].copy()
    df_test_filled  = df_test[feature_cols].copy()
    for col in _raio_cols:
        df_train_filled[col] = df_train_filled[col].fillna(_medians[col])
        df_test_filled[col]  = df_test_filled[col].fillna(_medians[col])
    X_train = df_train_filled.fillna(0.0).values
    y_train = df_train[target].values
    # GroupKFold groups on the original recording ID (strips _p<N>) so all
    # sub-trajectories from the same file stay in the same fold
    groups_train = np.array([_base_route(r) for r in df_train['id_route'].astype(str)])
    X_test  = df_test_filled.fillna(0.0).values
    y_test  = df_test[target].values

    return X_train, X_test, y_train, y_test, groups_train


def _split_rota_generico(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split por base-route com feature_cols explícito (não deriva de NAO_FEATURES).

    Usado por ISL e regressão, cujos alvos (ex.: _isl_y, curve_*) exigem
    controle fino sobre quais colunas são features. Garante que nenhuma rota
    apareça em treino e teste simultaneamente.

    NaN em colunas de raio são preenchidos pela mediana de treino; demais NaN -> 0.
    Retorna (X_train, X_test, y_train, y_test, groups_train).
    """
    checar_leakage(feature_cols, target)
    id_routes  = df['id_route'].astype(str)
    base_rotas = sorted({_base_route(r) for r in id_routes})
    train_base, test_base = train_test_split(
        base_rotas, test_size=test_size, random_state=random_state
    )
    base_map   = id_routes.map(_base_route)
    df_train   = df[base_map.isin(set(train_base))].dropna(subset=[target])
    df_test    = df[base_map.isin(set(test_base))].dropna(subset=[target])

    raio_cols = [c for c in feature_cols if 'raio' in c]
    medians   = df_train[raio_cols].median() if raio_cols else None

    def _prep(d: pd.DataFrame) -> np.ndarray:
        d = d[feature_cols].copy()
        for col in raio_cols:
            d[col] = d[col].fillna(medians[col])
        return d.fillna(0.0).values

    X_train = _prep(df_train)
    X_test  = _prep(df_test)
    y_train = df_train[target].values
    y_test  = df_test[target].values
    groups_train = np.array([_base_route(r) for r in df_train['id_route'].astype(str)])

    return X_train, X_test, y_train, y_test, groups_train


def _preparar_xy(df: pd.DataFrame, target: str = 'correcao_tardia_curva') -> tuple[np.ndarray, np.ndarray]:
    """Extrai features e target como arrays numpy brutos (sem pré-processamento)."""
    feature_cols = colunas_features(df)
    checar_leakage(feature_cols, target)
    df_clean = df[feature_cols + [target]].dropna()
    return df_clean[feature_cols].values, df_clean[target].values


def _construir_pipeline(clf, random_state: int = 42, use_smote: bool = True, pca_n_components=None,
                        smote_k: int = 5):
    """
    Constrói imblearn Pipeline: (SMOTE ->) StandardScaler (-> PCA) -> classificador.

    ImbPipeline garante que o SMOTE é re-executado apenas no fold de treino
    durante cross_validate, amostras sintéticas nunca cruzam para o fold de
    validação, evitando data leakage.
    """
    steps = []
    if use_smote:
        steps.append(('smote', SMOTE(k_neighbors=smote_k, random_state=random_state)))
    steps.append(('scaler', StandardScaler()))
    if pca_n_components is not None:
        steps.append(('pca', PCA(n_components=pca_n_components, random_state=random_state)))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def _preproc_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    use_smote: bool = True,
    random_state: int = 42,
    pca_n_components=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica SMOTE (opcional) -> StandardScaler -> PCA (opcional) corretamente:
      - SMOTE apenas em X_train / y_train
      - Scaler e PCA: fit em X_train, transform em X_test (sem vazar estatísticas do teste)

    Retorna (X_train_pp, X_test_pp, y_train_pp).
    """
    if use_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    if pca_n_components is not None:
        pca = PCA(n_components=pca_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        print(f'  PCA: {pca.n_components_} componentes, variância explicada: {pca.explained_variance_ratio_.sum():.1%}')

    return X_train, X_test, y_train


# Modelos clássicos

def aplicar_modelos_ml(
    df: pd.DataFrame,
    plot_cm: bool = True,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
    target: str = 'correcao_tardia_curva',
    f1_average: str = 'weighted',
    outdir: str = 'results',
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Treina e avalia modelos clássicos de ML. Serve alvos binários (risco) e multiclasse
    (isl_class), controlados por f1_average ('binary'/'weighted' vs 'macro') e labels.

    Metodologia correta:
      1. Split estratificado nos dados BRUTOS (antes de SMOTE / scaler / PCA)
      2. StratifiedKFold sobre X_train bruto
      3. Dentro de cada fold: SMOTE -> Scaler (-> PCA) -> fit  (via ImbPipeline)
        , amostras sintéticas nunca cruzam para o fold de validação
      4. Avaliação no teste com Acurácia, F1, Precisão, Recall

    pca_n_components : None | int (nº de componentes) | float 0–1 (variância explicada)
    labels           : nomes das classes para a matriz de confusão (default: Negativo/Positivo)
    """
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state,
    )
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # k_neighbors do SMOTE: limitado pela classe MENOS frequente (vale para binário e
    # multiclasse; a contagem por soma só funcionaria com rótulos 0/1).
    _, _counts = np.unique(y_train, return_counts=True)
    n_minority_train = int(_counts.min())
    n_minority_per_fold = max(1, n_minority_train * (cv_folds - 1) // cv_folds)
    smote_k = min(5, n_minority_per_fold - 1)

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'SVM':                 svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão':   DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória':  RandomForestClassifier(random_state=random_state),
        'XGBoost':             XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            eval_metric='logloss', random_state=random_state,
        ),
        'Rede Neural (MLP)':   MLPClassifier(
            activation='relu', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    linhas = []
    for nome, clf in modelos.items():
        pipe = _construir_pipeline(clf, random_state=random_state, pca_n_components=pca_n_components,
                                   use_smote=True, smote_k=smote_k)

        f1_scorer = 'f1' if f1_average == 'binary' else f'f1_{f1_average}'
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'acc': 'accuracy', 'f1': f1_scorer},
        )
        cv_acc = cv_res['test_acc'].mean()
        cv_f1  = cv_res['test_f1'].mean()

        # Avaliação final no conjunto de teste (nunca visto pelo pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average=f1_average)
        prec = precision_score(y_test, y_pred, average=f1_average, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=f1_average)

        print(
            f'{nome:30s}  CV Acc: {cv_acc:.3f}  CV F1: {cv_f1:.3f}  |  '
            f'Teste Acc: {acc:.3f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}'
        )

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=(labels or ['Negativo', 'Positivo']),
                        yticklabels=(labels or ['Negativo', 'Positivo']))
            plt.title(nome)
            plt.ylabel('Real')
            plt.xlabel('Prevista')
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f'cm_{nome}.pdf'), bbox_inches='tight')
            plt.close()

        linhas.append({
            'Classificador':    nome,
            'CV Acc (média)':   cv_acc,
            'CV F1 (média)':    cv_f1,
            'Acc (teste)':      acc,
            'F1 (teste)':       f1,
            'Precisão (teste)': prec,
            'Recall (teste)':   rec,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs(outdir, exist_ok=True)
    df_res.to_latex(
        os.path.join(outdir, 'resultados.tex'),
        float_format='%.3f',
        index=False,
        caption='Resultados dos modelos clássicos de ML, validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label='tab:ml_resultados',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# Optuna, Tuning de XGBoost e RandomForest

def _optuna_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    task: str = 'classify',
    cv_folds: int = 5,
    n_trials: int = 50,
    timeout: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Tuna XGBoost com Optuna usando GroupKFold.
    task='classify' -> XGBClassifier + f1_weighted
    task='regress'  -> XGBRegressor  + r2
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 9),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state':     random_state,
        }
        if task == 'classify':
            clf     = XGBClassifier(eval_metric='logloss', **params)
            scoring = 'f1_weighted'
        else:
            clf     = XGBRegressor(eval_metric='rmse', **params)
            scoring = 'r2'
        pipe   = _construir_pipeline(clf, random_state=random_state, use_smote=(task == 'classify'))
        cv     = (StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                  if task == 'classify' else GroupKFold(n_splits=cv_folds))
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params


def _optuna_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    task: str = 'classify',
    cv_folds: int = 5,
    n_trials: int = 30,
    timeout: int | None = None,
    random_state: int = 42,
) -> dict:
    """
    Tuna RandomForest com Optuna usando GroupKFold.
    task='classify' -> RandomForestClassifier + f1_weighted
    task='regress'  -> RandomForestRegressor  + r2
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_categorical('max_depth', [None, 10, 20, 30]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state':      random_state,
        }
        if task == 'classify':
            clf     = RandomForestClassifier(**params)
            scoring = 'f1_weighted'
        else:
            clf     = RandomForestRegressor(**params)
            scoring = 'r2'
        pipe   = _construir_pipeline(clf, random_state=random_state, use_smote=(task == 'classify'))
        cv     = (StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                  if task == 'classify' else GroupKFold(n_splits=cv_folds))
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params


def _optuna_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    cv_folds: int = 5,
    n_trials: int = 20,
    timeout: int | None = None,
    random_state: int = 42,
) -> dict:
    """Tuna LogisticRegression (C) com Optuna usando GroupKFold."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e3, log=True)
        clf = LogisticRegression(C=C, random_state=random_state, max_iter=2000)
        pipe = _construir_pipeline(clf, random_state=random_state, use_smote=True)
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups,
                                scoring='f1_weighted', n_jobs=-1)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params


def aplicar_modelos_ml_otimizados(
    df: pd.DataFrame,
    plot_cm: bool = True,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    n_trials_xgb: int = 50,
    n_trials_rf: int = 30,
    n_trials_lr: int = 20,
    timeout: int | None = None,
    target: str = 'correcao_tardia_curva',
    f1_average: str = 'weighted',
    outdir: str = 'results',
) -> pd.DataFrame:
    """
    Igual a aplicar_modelos_ml mas com Optuna para XGBoost e RandomForest.
    Usa split por id_route e StratifiedGroupKFold.
    """
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state,
    )
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    timeout_str = f", timeout={timeout}s" if timeout else ""
    print(f"  Tuning XGBoost com Optuna ({n_trials_xgb} trials{timeout_str})...")
    best_xgb = _optuna_xgb(X_train, y_train, groups_train, cv_folds=cv_folds,
                            n_trials=n_trials_xgb, timeout=timeout, random_state=random_state)
    print(f"  Melhores params XGB: {best_xgb}")

    print(f"  Tuning RandomForest com Optuna ({n_trials_rf} trials{timeout_str})...")
    best_rf = _optuna_rf(X_train, y_train, groups_train, cv_folds=cv_folds,
                         n_trials=n_trials_rf, timeout=timeout, random_state=random_state)
    print(f"  Melhores params RF: {best_rf}")

    print(f"  Tuning LogisticRegression com Optuna ({n_trials_lr} trials{timeout_str})...")
    best_lr = _optuna_lr(X_train, y_train, groups_train, cv_folds=cv_folds,
                         n_trials=n_trials_lr, timeout=timeout, random_state=random_state)
    print(f"  Melhores params LR: {best_lr}")

    modelos = {
        'Regressão Logística': LogisticRegression(**best_lr, random_state=random_state, max_iter=2000),
        'SVM':                 svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão':   DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória':  RandomForestClassifier(**best_rf),
        'XGBoost':             XGBClassifier(eval_metric='logloss', **best_xgb),
        'Rede Neural (MLP)':   MLPClassifier(
            activation='relu', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    n_minority_train = int(y_train.sum()) if y_train.mean() <= 0.5 else int((1 - y_train).sum())
    n_minority_per_fold = max(1, n_minority_train * (cv_folds - 1) // cv_folds)
    smote_k = min(5, n_minority_per_fold - 1)

    linhas = []
    for nome, clf in modelos.items():
        pipe   = _construir_pipeline(clf, random_state=random_state, use_smote=True, smote_k=smote_k)
        f1_scorer = 'f1' if f1_average == 'binary' else f'f1_{f1_average}'
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'acc': 'accuracy', 'f1': f1_scorer},
        )
        cv_acc = cv_res['test_acc'].mean()
        cv_f1  = cv_res['test_f1'].mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average=f1_average)
        prec = precision_score(y_test, y_pred, average=f1_average, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=f1_average)

        print(
            f'{nome:30s}  CV Acc: {cv_acc:.3f}  CV F1: {cv_f1:.3f}  |  '
            f'Teste Acc: {acc:.3f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}'
        )

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negativo', 'Positivo'],
                        yticklabels=['Negativo', 'Positivo'])
            plt.title(nome)
            plt.ylabel('Real')
            plt.xlabel('Prevista')
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f'cm_{nome}_otim.pdf'), bbox_inches='tight')
            plt.close()

        linhas.append({
            'Classificador': nome, 'CV Acc (média)': cv_acc, 'CV F1 (média)': cv_f1,
            'Acc (teste)': acc, 'F1 (teste)': f1, 'Precisão (teste)': prec, 'Recall (teste)': rec,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs(outdir, exist_ok=True)
    df_res.to_latex(
        os.path.join(outdir, 'resultados_otimizado.tex'),
        float_format='%.3f',
        index=False,
        caption='Resultados dos modelos com tuning Optuna (XGBoost e RandomForest), validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label='tab:ml_resultados_opt',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# Regressão genérica

def plotar_scatter_regressao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    nome_modelo: str,
    outdir: str = 'results',
) -> None:
    """
    Scatter plot predicted vs actual para regressão.
    Inclui linha de identidade (y=x) e R² anotado.
    Salva em outdir/scatter_<target>_<nome_modelo>.pdf
    """
    from sklearn.metrics import r2_score
    os.makedirs(outdir, exist_ok=True)

    r2 = r2_score(y_true, y_pred)
    lim = (min(y_true.min(), y_pred.min()) * 0.95,
           max(y_true.max(), y_pred.max()) * 1.05)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color='steelblue', edgecolors='none')
    ax.plot(lim, lim, color='tomato', linewidth=1.2, linestyle='--', label='y = x')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'Real, {target}')
    ax.set_ylabel('Previsto')
    ax.set_title(f'{nome_modelo}\nR² = {r2:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f'scatter_{target}_{nome_modelo.replace(" ", "_")}.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Scatter salvo em {path}")




# Regressão tabular

def treinar_regressao(
    df: pd.DataFrame,
    target: str = 'v_critica',
    plot: bool = True,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    outdir: str = 'results',
    unidade: str = 'km/h',
    classes_derivadas=None,
) -> pd.DataFrame:
    """
    Modelos clássicos de regressão sobre as features agregadas por curva.

    É a contraparte tabular do que temporais.treinar_regressao_ts faz sobre a série bruta:
    mesmo alvo, mesmo split por rota, mas a entrada é o vetor de features da curva em vez
    da janela de 50 passos. Serve para responder se a série bruta ganha das features.

    Sem SMOTE (alvo contínuo) e com GroupKFold em vez de StratifiedGroupKFold, pelo mesmo
    motivo. Reporta uma referência trivial (prever sempre a média do treino), que é o piso
    abaixo do qual um modelo não tem serventia.

    classes_derivadas: função que corta o valor contínuo em faixas. Quando dada, cada modelo
    também reporta o F1-macro da faixa obtida cortando a predição, o que permite comparar a
    regressão com um classificador treinado direto nas faixas sem perder a distância até o
    limiar durante o treino.
    """
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state,
    )
    cv = GroupKFold(n_splits=cv_folds)

    print(f"  Split - treino: {len(y_train)} (média={y_train.mean():.3f})  "
          f"teste: {len(y_test)} (média={y_test.mean():.3f})")

    # Baseline sem aprendizado: prever sempre a média do treino.
    pred_media = np.full_like(y_test, y_train.mean(), dtype=float)
    print(f"  [BASELINE média do treino]  R²: {r2_score(y_test, pred_media):7.4f}  "
          f"MAE: {mean_absolute_error(y_test, pred_media):6.3f} {unidade}")

    modelos = {
        'Ridge':              Ridge(random_state=random_state),
        'Floresta Aleatória': RandomForestRegressor(random_state=random_state, n_jobs=-1),
        'XGBoost':            XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=random_state, verbosity=0,
        ),
        'Rede Neural (MLP)':  MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    linhas = []
    for nome, reg in modelos.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', reg)])

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'},
        )
        cv_r2  = cv_res['test_r2'].mean()
        cv_mae = -cv_res['test_mae'].mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f'{nome:22s}  CV R²: {cv_r2:7.4f}  CV MAE: {cv_mae:6.3f}  |  '
              f'Teste R²: {r2:7.4f}  MAE: {mae:6.3f} {unidade}')

        f1_faixa = acc_faixa = None
        if classes_derivadas is not None:
            cls_pred = classes_derivadas(y_pred)
            cls_true = classes_derivadas(y_test)
            f1_faixa  = f1_score(cls_true, cls_pred, average='macro')
            acc_faixa = accuracy_score(cls_true, cls_pred)
            print(f'{"":22s}  -> faixa cortada da predição  F1-macro: {f1_faixa:.4f}  '
                  f'Acc: {acc_faixa:.4f}')

        if plot:
            plotar_scatter_regressao(y_test, y_pred, target, nome, outdir=outdir)

        linhas.append({
            'Modelo':        nome,
            'CV R²':         cv_r2,
            f'CV MAE ({unidade})':  cv_mae,
            'R² (teste)':    r2,
            f'MAE ({unidade})':     mae,
            **({'F1-macro da faixa': f1_faixa, 'Acc da faixa': acc_faixa}
               if classes_derivadas is not None else {}),
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs(outdir, exist_ok=True)
    df_res.to_latex(
        os.path.join(outdir, 'resultados.tex'),
        float_format='%.3f',
        index=False,
        caption=f'Regressão tabular para {target.replace("_", chr(92) + "_")}, '
                f'validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label=f'tab:reg_{target}',
        position='h',
    )
    return df_res
