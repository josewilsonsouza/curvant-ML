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
from curvant.driving.features import colunas_features


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
    base_rotas = list({_base_route(r) for r in id_routes})
    train_base, test_base = train_test_split(
        base_rotas, test_size=test_size, random_state=random_state
    )
    train_base_set = set(train_base)
    test_base_set  = set(test_base)

    feature_cols = colunas_features(df)
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
    # do conjunto de treino — semanticamente "raio típico" em vez de 0 (curvatura
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
    id_routes  = df['id_route'].astype(str)
    base_rotas = list({_base_route(r) for r in id_routes})
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


def _preparar_xy(df: pd.DataFrame, target: str = 'manobra') -> tuple[np.ndarray, np.ndarray]:
    """Extrai features e target como arrays numpy brutos (sem pré-processamento)."""
    feature_cols = colunas_features(df)
    df_clean = df[feature_cols + [target]].dropna()
    return df_clean[feature_cols].values, df_clean[target].values


def _construir_pipeline(clf, random_state: int = 42, use_smote: bool = True, pca_n_components=None):
    """
    Constrói imblearn Pipeline: (SMOTE ->) StandardScaler (-> PCA) -> classificador.

    ImbPipeline garante que o SMOTE é re-executado apenas no fold de treino
    durante cross_validate — amostras sintéticas nunca cruzam para o fold de
    validação, evitando data leakage.
    """
    steps = []
    if use_smote:
        steps.append(('smote', SMOTE(random_state=random_state)))
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
        print(f'  PCA: {pca.n_components_} componentes — variância explicada: {pca.explained_variance_ratio_.sum():.1%}')

    return X_train, X_test, y_train


# Modelos clássicos

def aplicar_modelos_ml(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
    target: str = 'manobra',
    f1_average: str = 'weighted',
    outdir: str = 'results',
) -> pd.DataFrame:
    """
    Treina e avalia modelos clássicos de ML.

    Metodologia correta:
      1. Split estratificado nos dados BRUTOS (antes de SMOTE / scaler / PCA)
      2. StratifiedKFold sobre X_train bruto
      3. Dentro de cada fold: SMOTE -> Scaler (-> PCA) -> fit  (via ImbPipeline)
         — amostras sintéticas nunca cruzam para o fold de validação
      4. Avaliação no teste com Acurácia, F1 weighted, Precisão, Recall

    pca_n_components : None | int (nº de componentes) | float 0–1 (variância explicada)
    """
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state,
    )
    cv = GroupKFold(n_splits=cv_folds)

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
        pipe = _construir_pipeline(clf, random_state=random_state, pca_n_components=pca_n_components, use_smote=True)

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'acc': 'accuracy', 'f1': f'f1_{f1_average}'},
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
                        xticklabels=['Segura', 'Risco'],
                        yticklabels=['Segura', 'Risco'])
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
        caption='Resultados dos modelos clássicos de ML — validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label='tab:ml_resultados',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# Optuna — Tuning de XGBoost e RandomForest

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
        cv     = GroupKFold(n_splits=cv_folds)
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
        cv     = GroupKFold(n_splits=cv_folds)
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
        cv = GroupKFold(n_splits=cv_folds)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups,
                                scoring='f1_weighted', n_jobs=-1)
        return scores['test_score'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params


def aplicar_modelos_ml_otimizados(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    n_trials_xgb: int = 50,
    n_trials_rf: int = 30,
    n_trials_lr: int = 20,
    timeout: int | None = None,
    target: str = 'manobra',
    f1_average: str = 'weighted',
    outdir: str = 'results',
) -> pd.DataFrame:
    """
    Igual a aplicar_modelos_ml mas com Optuna para XGBoost e RandomForest.
    Usa split por id_route e GroupKFold.
    """
    X_train, X_test, y_train, y_test, groups_train = _split_por_rota(
        df, target=target, test_size=test_size, random_state=random_state,
    )
    cv = GroupKFold(n_splits=cv_folds)

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

    linhas = []
    for nome, clf in modelos.items():
        pipe   = _construir_pipeline(clf, random_state=random_state, use_smote=True)
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'acc': 'accuracy', 'f1': f'f1_{f1_average}'},
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
                        xticklabels=['Segura', 'Risco'],
                        yticklabels=['Segura', 'Risco'])
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
        caption='Resultados dos modelos com tuning Optuna (XGBoost e RandomForest) — validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label='tab:ml_resultados_opt',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# ISL - Índice de Segurança Lateral

_ISL_ENCODE = {'baixo': 0, 'medio': 1, 'alto': 2}
_ISL_LABELS = ['Baixo', 'Médio', 'Alto']


def treinar_modelo_isl(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
    isl_max_cap_percentil: int | None = 99,
    outdir: str = 'results',
) -> pd.DataFrame:
    """
    Treina modelos clássicos para prever a classe ISL (3 classes) antes da curva.

    Target: ``isl_class`` codificado como ordinal  0=baixo / 1=medio / 2=alto.
    isl_max_cap_percentil : remove outliers de B-spline/GPS (isl_max acima do percentil).
    """
    df_valid = df.dropna(subset=['isl_class', 'isl_max']).copy()

    if isl_max_cap_percentil is not None:
        cap = float(np.percentile(df_valid['isl_max'], isl_max_cap_percentil))
        n_antes = len(df_valid)
        df_valid = df_valid[df_valid['isl_max'] <= cap].copy()
        print(f"  ISL — cap p{isl_max_cap_percentil}: {cap:.3f}  ({n_antes - len(df_valid)} amostras removidas)")

    df_valid['_isl_y'] = df_valid['isl_class'].map(_ISL_ENCODE)
    df_valid = df_valid.dropna(subset=['_isl_y'])
    df_valid['_isl_y'] = df_valid['_isl_y'].astype(int)

    counts = df_valid['_isl_y'].value_counts().sort_index()
    label_str = ' | '.join(f'{_ISL_LABELS[k]}: {v}' for k, v in counts.items())
    print(f"  ISL — amostras válidas: {len(df_valid)}  ({label_str})")

    feature_cols = [c for c in colunas_features(df_valid) if c != '_isl_y']
    X_train, X_test, y_train, y_test, groups_train = _split_rota_generico(
        df_valid, feature_cols, target='_isl_y',
        test_size=test_size, random_state=random_state,
    )

    # StratifiedGroupKFold: respeita as fronteiras de gravação (sem leakage por rota)
    # e mantém a proporção das 3 classes em cada fold.
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'SVM':                 svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão':   DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória':  RandomForestClassifier(random_state=random_state),
        'XGBoost':             XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            eval_metric='mlogloss', random_state=random_state,
        ),
        'Rede Neural (MLP)':   MLPClassifier(
            activation='relu', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    linhas = []
    for nome, clf in modelos.items():
        pipe = _construir_pipeline(clf, random_state=random_state, pca_n_components=pca_n_components, use_smote=True)

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'acc': 'accuracy', 'f1': 'f1_macro'},
        )
        cv_acc = cv_res['test_acc'].mean()
        cv_f1  = cv_res['test_f1'].mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro')

        print(
            f'{nome:30s}  CV Acc: {cv_acc:.3f}  CV F1: {cv_f1:.3f}  |  '
            f'Teste Acc: {acc:.3f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}'
        )

        if plot_cm:
            classes_presentes = sorted(np.unique(np.concatenate([y_test, y_pred])))
            labels = [_ISL_LABELS[c] for c in classes_presentes]
            cm = confusion_matrix(y_test, y_pred, labels=classes_presentes)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=labels, yticklabels=labels)
            plt.title(f'ISL — {nome}')
            plt.ylabel('Real')
            plt.xlabel('Prevista')
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f'cm_{nome}.pdf'), bbox_inches='tight')
            plt.close()

        linhas.append({
            'Classificador':    nome,
            'CV Acc (média)':   cv_acc,
            'CV F1 macro (md)': cv_f1,
            'Acc (teste)':      acc,
            'F1 macro (teste)': f1,
            'Precisão (teste)': prec,
            'Recall (teste)':   rec,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs(outdir, exist_ok=True)
    df_res.to_latex(
        os.path.join(outdir, 'resultados.tex'),
        float_format='%.3f',
        index=False,
        caption='Resultados da classificação do Índice de Segurança Lateral (ISL) — 3 classes (baixo/médio/alto), F1 macro, validação cruzada por rota.',
        label='tab:isl_resultados',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# Baseline físico (sem aprendizado

def avaliar_baseline_fisico(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.3,
    isl_max_cap_percentil: int | None = 99,
    outdir: str = 'results',
) -> pd.DataFrame:
    """
    Baseline físico (sem treino), avaliado em conjunto de teste por rota com os
    mesmos parâmetros de split dos modelos ML, para quantificar quanto o ML agrega
    além da física pura.

    - isl_class : predição = argmax(mc_p_baixo, mc_p_medio, mc_p_alto)  [Monte Carlo]
    - isl_max   : ISL_phys = (v_pred_kinematica/3.6)² / (f4_raio_min · g · μ)
    """
    mc_cols = ['mc_p_baixo', 'mc_p_medio', 'mc_p_alto']

    def _test_mask(d: pd.DataFrame) -> pd.Series:
        base  = d['id_route'].astype(str).map(_base_route)
        bases = list(set(base))
        _, test_base = train_test_split(bases, test_size=test_size, random_state=random_state)
        return base.isin(set(test_base))

    linhas = []

    # isl_class via argmax da simulação de Monte Carlo
    if 'isl_class' in df.columns and all(c in df.columns for c in mc_cols):
        d = df.dropna(subset=['isl_class', 'isl_max'] + mc_cols).copy()
        if isl_max_cap_percentil is not None and len(d):
            cap = float(np.percentile(d['isl_max'], isl_max_cap_percentil))
            d = d[d['isl_max'] <= cap]
        d = d[d[mc_cols].sum(axis=1) > 0]   # ignora linhas sem MC (raio ausente)
        if len(d):
            dt     = d[_test_mask(d)]
            y_true = dt['isl_class'].map(_ISL_ENCODE).astype(int).values
            y_pred = dt[mc_cols].values.argmax(axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
            print(f"  [baseline físico] isl_class (argmax MC)  Acc: {acc:.3f}  F1-macro: {f1:.3f}  (n_teste={len(dt)})")
            linhas.append({'Alvo': 'isl_class', 'Método': 'argmax(MC)',
                           'Métrica 1': f'Acc={acc:.3f}', 'Métrica 2': f'F1m={f1:.3f}'})

    # isl_max via fórmula física com velocidade prevista pela cinemática da janela
    need = ['isl_max', 'v_pred_kinematica', 'f4_raio_min']
    if all(c in df.columns for c in need):
        d = df.dropna(subset=need).copy()
        d = d[d['f4_raio_min'] > 0]
        if isl_max_cap_percentil is not None and len(d):
            cap = float(np.percentile(d['isl_max'], isl_max_cap_percentil))
            d = d[d['isl_max'] <= cap]
        if len(d):
            dt       = d[_test_mask(d)]
            isl_phys = (dt['v_pred_kinematica'].values / 3.6) ** 2 / (dt['f4_raio_min'].values * _G * _MU)
            y_true   = dt['isl_max'].values
            mae  = mean_absolute_error(y_true, isl_phys)
            rmse = float(np.sqrt(mean_squared_error(y_true, isl_phys)))
            r2   = r2_score(y_true, isl_phys)
            print(f"  [baseline físico] isl_max (fórmula)  MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}  (n_teste={len(dt)})")
            linhas.append({'Alvo': 'isl_max', 'Método': 'v_pred²/(R·g·μ)',
                           'Métrica 1': f'MAE={mae:.3f}', 'Métrica 2': f'R²={r2:.3f}'})

    df_res = pd.DataFrame(linhas)
    if not df_res.empty:
        os.makedirs(outdir, exist_ok=True)
        df_res.to_latex(
            os.path.join(outdir, 'baseline.tex'),
            index=False,
            caption='Baseline físico (sem aprendizado) — argmax da simulação de Monte Carlo '
                    'para isl\\_class e fórmula do ISL para isl\\_max, avaliados no conjunto de '
                    'teste por rota (mesmo split dos modelos ML).',
            label='tab:baseline_fisico',
            position='h',
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
    ax.set_xlabel(f'Real — {target}')
    ax.set_ylabel('Previsto')
    ax.set_title(f'{nome_modelo}\nR² = {r2:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f'scatter_{target}_{nome_modelo.replace(" ", "_")}.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Scatter salvo em {path}")


def treinar_regressao(
    df: pd.DataFrame,
    target: str,
    plot: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
    cap_percentil: int | None = 99,
    outdir: str = 'results',
) -> pd.DataFrame:
    """
    Treina modelos de regressão para prever qualquer target contínuo.

    Pipeline: StandardScaler (-> PCA) -> regressor. Sem SMOTE.
    Métricas: CV MAE, CV R² | MAE, RMSE, R² no teste.

    cap_percentil : remove amostras acima deste percentil do target (None = sem filtro).
    """
    df_valid = df.dropna(subset=[target]).copy()

    if cap_percentil is not None:
        cap = float(np.percentile(df_valid[target], cap_percentil))
        n_antes = len(df_valid)
        df_valid = df_valid[df_valid[target] <= cap].copy()
        print(f"  [{target}] cap p{cap_percentil}: {cap:.3f}  ({n_antes - len(df_valid)} removidas)")

    print(
        f"  [{target}]  n={len(df_valid)}"
        f"  média={df_valid[target].mean():.3f}"
        f"  mediana={df_valid[target].median():.3f}"
        f"  std={df_valid[target].std():.3f}"
    )

    feature_cols = [c for c in colunas_features(df_valid) if c != target]
    X_train, X_test, y_train, y_test, groups_train = _split_rota_generico(
        df_valid, feature_cols, target=target,
        test_size=test_size, random_state=random_state,
    )

    # GroupKFold: folds respeitam as fronteiras de gravação (sem leakage por rota)
    cv = GroupKFold(n_splits=cv_folds)

    steps_base = [('scaler', StandardScaler())]
    if pca_n_components is not None:
        steps_base.append(('pca', PCA(n_components=pca_n_components, random_state=random_state)))

    modelos = {
        'Ridge':              Ridge(),
        'SVR':                svm.SVR(kernel='rbf'),
        'Floresta Aleatória': RandomForestRegressor(random_state=random_state),
        'XGBoost':            XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            eval_metric='rmse', random_state=random_state,
        ),
    }

    linhas = []
    for nome, reg in modelos.items():
        pipe = Pipeline(steps_base + [('reg', reg)])

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv, groups=groups_train,
            scoring={'mae': 'neg_mean_absolute_error', 'r2': 'r2'},
        )
        cv_mae = -cv_res['test_mae'].mean()
        cv_r2  =  cv_res['test_r2'].mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = r2_score(y_test, y_pred)

        print(
            f'{nome:22s}  CV MAE: {cv_mae:.3f}  CV R²: {cv_r2:.3f}  |  '
            f'MAE: {mae:.3f}  RMSE: {rmse:.3f}  R²: {r2:.3f}'
        )

        if plot:
            plotar_scatter_regressao(y_test, y_pred, target=target, nome_modelo=nome, outdir=outdir)

        linhas.append({
            'Modelo': nome, 'CV MAE': cv_mae, 'CV R²': cv_r2,
            'MAE (teste)': mae, 'RMSE (teste)': rmse, 'R² (teste)': r2,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs(outdir, exist_ok=True)
    safe = target.replace('/', '_')
    df_res.to_latex(
        os.path.join(outdir, f'regressao_{safe}.tex'),
        float_format='%.3f',
        index=False,
        caption=f'Resultados da regressão para o alvo \\texttt{{{target}}} — MAE, RMSE e $R^2$ em validação cruzada por rota e conjunto de teste.',
        label=f'tab:regressao_{safe}',
        position='h',
        column_format='lccccc',
    )
    return df_res
