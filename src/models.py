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
    GroupKFold, GridSearchCV, KFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier, XGBRegressor

_COLS_EXCLUIR = [
    # identificadores
    'id_route', 'id_trecho_curvo', 'time_inicio', 'time_fim',
    # targets de caracterização por curva
    'manobra', 'manobra_combinado_curva',
    'manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva',
    # targets ISL — cinemático (v²/Rg μ) e baseado no sensor (|accel_y|/g μ)
    # ambos calculados dentro da curva: leakage se usados como features
    'isl_value', 'isl_mean', 'isl_max', 'isl_class', 'isl_alto',
    'isl_sensor_max', 'isl_sensor_mean', 'isl_sensor_class',
    # targets de aceleração dentro da curva
    'curve_accel_y_max', 'curve_accel_y_mean',
    'curve_abs_accel_max', 'curve_abs_accel_mean',
    # targets de velocidade
    'v_excess', 'manobra_velocidade', 'v_safe_dnit', 'v_entry_ratio',
    # geometria bruta da curva atual (leakage; usar f4_* em Modo 1)
    'curve_raio_min', 'curve_raio_mean', 'curve_dnit_num',
]


# ── Utilitários internos ──────────────────────────────────────────────────────

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

    feature_cols = [c for c in df.columns if c not in _COLS_EXCLUIR]
    cols = feature_cols + [target, 'id_route']

    df_train = (
        df[id_routes.map(_base_route).isin(train_base_set)][cols]
        .dropna(subset=[target])
    )
    df_test = (
        df[id_routes.map(_base_route).isin(test_base_set)][cols]
        .dropna(subset=[target])
    )

    # NaN em features de raio (f4_raio_* em Modo 2) são preenchidas com a mediana
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


def _preparar_xy(df: pd.DataFrame, target: str = 'manobra') -> tuple[np.ndarray, np.ndarray]:
    """Extrai features e target como arrays numpy brutos (sem pré-processamento)."""
    feature_cols = [c for c in df.columns if c not in _COLS_EXCLUIR]
    df_clean = df[feature_cols + [target]].dropna()
    return df_clean[feature_cols].values, df_clean[target].values


def _construir_pipeline(clf, random_state: int = 42, use_smote: bool = True, pca_n_components=None):
    """
    Constrói imblearn Pipeline: (SMOTE →) StandardScaler (→ PCA) → classificador.

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
    Aplica SMOTE (opcional) → StandardScaler → PCA (opcional) corretamente:
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


# ── Modelos clássicos ─────────────────────────────────────────────────────────

def aplicar_modelos_ml(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
    target: str = 'manobra',
    f1_average: str = 'weighted',
) -> pd.DataFrame:
    """
    Treina e avalia modelos clássicos de ML.

    Metodologia correta:
      1. Split estratificado nos dados BRUTOS (antes de SMOTE / scaler / PCA)
      2. StratifiedKFold sobre X_train bruto
      3. Dentro de cada fold: SMOTE → Scaler (→ PCA) → fit  (via ImbPipeline)
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
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/matriz_confusao_{nome}.pdf', bbox_inches='tight')
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
    os.makedirs('results', exist_ok=True)
    df_res.to_latex(
        'results/ml_resultados.tex',
        float_format='%.3f',
        index=False,
        caption='Resultados dos modelos clássicos de ML — validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label='tab:ml_resultados',
        position='h',
        column_format='lcccccc',
    )
    return df_res


def treinar_mlp_sklearn(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.3,
    pca_n_components=None,
    hidden_layer_sizes=None,
    activation: str = 'relu',
    solver: str = 'adam',
    alpha: float = 0.001,
    max_iter: int = 2000,
) -> MLPClassifier:
    """
    Treina MLPClassifier com GridSearchCV.
    SMOTE, StandardScaler e PCA aplicados apenas no conjunto de treino.
    GridSearchCV usa f1_weighted (mais adequado que accuracy para dados desbalanceados).
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [(128, 64)]

    X, y = _preparar_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    X_train_pp, X_test_pp, y_train_pp = _preproc_train_test(
        X_train, X_test, y_train,
        use_smote=False, random_state=random_state, pca_n_components=pca_n_components,
    )

    # Normaliza hidden_layer_sizes: aceita listas de listas ou lista simples
    if hidden_layer_sizes and not isinstance(hidden_layer_sizes[0], (list, tuple)):
        hidden_layer_sizes = [hidden_layer_sizes]
    param_grid = {
        'hidden_layer_sizes': [tuple(h) for h in hidden_layer_sizes],
        'activation': [activation],
        'solver': [solver],
        'alpha': [alpha],
        'learning_rate': ['constant'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        MLPClassifier(max_iter=2000, random_state=random_state),
        param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, return_train_score=True,
    )
    grid.fit(X_train_pp, y_train_pp)
    best = grid.best_estimator_
    print(f'  Melhores parâmetros: {grid.best_params_}')

    y_pred    = best.predict(X_test_pp)
    train_acc = best.score(X_train_pp, y_train_pp)
    test_acc  = accuracy_score(y_test, y_pred)
    test_f1   = f1_score(y_test, y_pred, average='weighted')
    test_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    test_rec  = recall_score(y_test, y_pred, average='weighted')

    print(
        f'  Treino Acc: {train_acc:.4f} | '
        f'Teste Acc: {test_acc:.4f}  F1: {test_f1:.4f}  Prec: {test_prec:.4f}  Rec: {test_rec:.4f}'
    )
    if train_acc - test_acc > 0.1:
        print('  Aviso: possível overfitting detectado.')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Segura', 'Risco'], yticklabels=['Segura', 'Risco'])
    plt.title('Matriz de Confusão — MLP (sklearn)')
    plt.ylabel('Real')
    plt.xlabel('Prevista')
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/matriz_confusao_MLP_sklearn.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(best.loss_curve_, color='blue', label='Perda no Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Curva de Perda do MLP')
    plt.legend()
    plt.grid()
    plt.savefig('results/curva_perda_MLP_sklearn.pdf', bbox_inches='tight')
    plt.close()

    return best


# ── Redes Neurais (Keras) ─────────────────────────────────────────────────────

def preparar_dados_keras(
    df: pd.DataFrame,
    target: str = 'manobra',
    test_size: float = 0.3,
    val_size: float = 0.2,
    random_state: int = 42,
    pca_n_components=None,
) -> tuple:
    """
    Prepara dados para MLP Keras sem leakage de SMOTE na validação.

    Ordem correta:
      1. Split treino / teste estratificado nos dados brutos
      2. Split treino / val estratificado nos dados brutos (antes do SMOTE)
      3. SMOTE apenas em X_train
      4. Scaler fit em X_train → transform X_val e X_test

    Retorna (X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat).
    """
    from tensorflow.keras.utils import to_categorical

    X, y = _preparar_xy(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    if pca_n_components is not None:
        pca = PCA(n_components=pca_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_val   = pca.transform(X_val)
        X_test  = pca.transform(X_test)

    return (
        X_train, X_val, X_test,
        to_categorical(y_train), to_categorical(y_val), to_categorical(y_test),
    )


def construir_mlp_keras(num_features: int):
    """Constrói um MLP com Keras (3 camadas Dense + Dropout)."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    modelo = Sequential([
        Input(shape=(num_features,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax'),
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


def preparar_dados_gru(
    df: pd.DataFrame,
    janela_tempo: int = 1,
    target: str = 'manobra',
    test_size: float = 0.3,
    random_state: int = 42,
    pca_n_components=None,
) -> tuple:
    """
    Prepara dados 3D (samples, timesteps, features) para GRU.
    Split estratificado → Scaler (→ PCA) apenas no treino (sem SMOTE — dados sequenciais).
    Retorna (X_train_3d, X_test_3d, y_train_cat, y_test_cat, num_features).
    """
    from tensorflow.keras.utils import to_categorical

    X, y = _preparar_xy(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_test, y_train = _preproc_train_test(
        X_train, X_test, y_train,
        use_smote=False, random_state=random_state, pca_n_components=pca_n_components,
    )

    num_features = X_train.shape[1]
    n_train = X_train.shape[0] // janela_tempo
    n_test  = X_test.shape[0]  // janela_tempo

    X_train_3d  = X_train[:n_train * janela_tempo].reshape(n_train, janela_tempo, num_features)
    X_test_3d   = X_test[:n_test * janela_tempo].reshape(n_test, janela_tempo, num_features)
    y_train_cat = to_categorical(y_train[:n_train])
    y_test_cat  = to_categorical(y_test[:n_test])

    return X_train_3d, X_test_3d, y_train_cat, y_test_cat, num_features


def construir_gru(janela_tempo: int, num_features: int):
    """Constrói um modelo GRU com Keras."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense

    modelo = Sequential([
        GRU(3, input_shape=(janela_tempo, num_features), return_sequences=True),
        GRU(2),
        Dense(2, activation='softmax'),
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


# ── LSTM ──────────────────────────────────────────────────────────────────────

def preparar_dados_lstm(
    df: pd.DataFrame,
    janela_tempo: int = 1,
    target: str = 'manobra',
    test_size: float = 0.3,
    random_state: int = 42,
    pca_n_components=None,
) -> tuple:
    """
    Prepara dados 3D para LSTM. Idêntico ao GRU.
    Split estratificado → Scaler (→ PCA) apenas no treino, sem SMOTE.
    Retorna (X_train_3d, X_test_3d, y_train_cat, y_test_cat, num_features).
    """
    from tensorflow.keras.utils import to_categorical

    X, y = _preparar_xy(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_test, y_train = _preproc_train_test(
        X_train, X_test, y_train,
        use_smote=False, random_state=random_state, pca_n_components=pca_n_components,
    )

    num_features = X_train.shape[1]
    n_train = X_train.shape[0] // janela_tempo
    n_test  = X_test.shape[0]  // janela_tempo

    X_train_3d  = X_train[:n_train * janela_tempo].reshape(n_train, janela_tempo, num_features)
    X_test_3d   = X_test[:n_test * janela_tempo].reshape(n_test, janela_tempo, num_features)
    y_train_cat = to_categorical(y_train[:n_train])
    y_test_cat  = to_categorical(y_test[:n_test])

    return X_train_3d, X_test_3d, y_train_cat, y_test_cat, num_features


def construir_lstm(janela_tempo: int, num_features: int):
    """
    Constrói um modelo LSTM com Keras.

    Arquitetura:
      LSTM(64, return_sequences=True) → Dropout(0.3)
      LSTM(32)                        → Dropout(0.3)
      Dense(2, softmax)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    modelo = Sequential([
        LSTM(64, input_shape=(janela_tempo, num_features), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(2, activation='softmax'),
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


# ── Optuna — Tuning de XGBoost e RandomForest ────────────────────────────────

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
    task='classify' → XGBClassifier + f1_weighted
    task='regress'  → XGBRegressor  + r2
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
        pipe   = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
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
    task='classify' → RandomForestClassifier + f1_weighted
    task='regress'  → RandomForestRegressor  + r2
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
        pipe   = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        cv     = GroupKFold(n_splits=cv_folds)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, groups=groups, scoring=scoring, n_jobs=-1)
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
    timeout: int | None = None,
    target: str = 'manobra',
    f1_average: str = 'weighted',
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

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
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
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/matriz_confusao_{nome}_opt.pdf', bbox_inches='tight')
            plt.close()

        linhas.append({
            'Classificador': nome, 'CV Acc (média)': cv_acc, 'CV F1 (média)': cv_f1,
            'Acc (teste)': acc, 'F1 (teste)': f1, 'Precisão (teste)': prec, 'Recall (teste)': rec,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs('results', exist_ok=True)
    df_res.to_latex(
        'results/ml_resultados_opt.tex',
        float_format='%.3f',
        index=False,
        caption='Resultados dos modelos com tuning Optuna (XGBoost e RandomForest) — validação cruzada por rota (GroupKFold) e conjunto de teste.',
        label='tab:ml_resultados_opt',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# ── ISL — Índice de Segurança Lateral ────────────────────────────────────────

_COLS_EXCLUIR_ISL = _COLS_EXCLUIR  # isl_mean/max/class/alto já estão em _COLS_EXCLUIR

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

    feature_cols = [c for c in df_valid.columns if c not in _COLS_EXCLUIR_ISL and c != '_isl_y']
    df_clean = df_valid[feature_cols + ['_isl_y']].dropna()
    X = df_clean[feature_cols].values
    y = df_clean['_isl_y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

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
            pipe, X_train, y_train, cv=cv,
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
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/isl_cm_{nome}.pdf', bbox_inches='tight')
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
    os.makedirs('results', exist_ok=True)
    df_res.to_latex(
        'results/isl_resultados.tex',
        float_format='%.3f',
        index=False,
        caption='Resultados da classificação do Índice de Segurança Lateral (ISL) — 3 classes (baixo/médio/alto), F1 macro, validação cruzada por rota.',
        label='tab:isl_resultados',
        position='h',
        column_format='lcccccc',
    )
    return df_res


# ── Regressão genérica ────────────────────────────────────────────────────────

def treinar_regressao(
    df: pd.DataFrame,
    target: str,
    plot: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
    cap_percentil: int | None = 99,
) -> pd.DataFrame:
    """
    Treina modelos de regressão para prever qualquer target contínuo.

    Pipeline: StandardScaler (→ PCA) → regressor. Sem SMOTE.
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

    feature_cols = [c for c in df_valid.columns if c not in _COLS_EXCLUIR_ISL]
    df_clean = df_valid[feature_cols + [target]].dropna()
    X = df_clean[feature_cols].values
    y = df_clean[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

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
            pipe, X_train, y_train, cv=cv,
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
            from graphics.visualization import plotar_scatter_regressao
            plotar_scatter_regressao(y_test, y_pred, target=target, nome_modelo=nome)

        linhas.append({
            'Modelo': nome, 'CV MAE': cv_mae, 'CV R²': cv_r2,
            'MAE (teste)': mae, 'RMSE (teste)': rmse, 'R² (teste)': r2,
        })

    df_res = pd.DataFrame(linhas)
    os.makedirs('results', exist_ok=True)
    safe = target.replace('/', '_')
    df_res.to_latex(
        f'results/regressao_{safe}.tex',
        float_format='%.3f',
        index=False,
        caption=f'Resultados da regressão para o alvo \\texttt{{{target}}} — MAE, RMSE e $R^2$ em validação cruzada por rota e conjunto de teste.',
        label=f'tab:regressao_{safe}',
        position='h',
        column_format='lccccc',
    )
    return df_res


