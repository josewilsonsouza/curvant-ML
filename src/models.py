import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

_COLS_EXCLUIR = [
    'manobra', 'id_route', 'id_trecho_curvo', 'time_inicio', 'time_fim',
    'manobra_accel_perigo', 'manobra_dir_perigosa', 'manobra_zigue_zague',
    # colunas ISL — excluídas das features dos modelos gerais; isl_alto é target do modelo ISL
    'isl_mean', 'isl_max', 'isl_class', 'isl_alto',
]


# ── Utilitários internos ──────────────────────────────────────────────────────

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
    X, y = _preparar_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'SVM':                 svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão':   DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória':  RandomForestClassifier(random_state=random_state),
        'Rede Neural (MLP)':   MLPClassifier(
            activation='relu', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    linhas = []
    for nome, clf in modelos.items():
        pipe = _construir_pipeline(clf, random_state=random_state, pca_n_components=pca_n_components)

        # cross_validate sobre X_train bruto — SMOTE contido em cada fold pelo ImbPipeline
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={'acc': 'accuracy', 'f1': 'f1_weighted'},
        )
        cv_acc = cv_res['test_acc'].mean()
        cv_f1  = cv_res['test_f1'].mean()

        # Avaliação final no conjunto de teste (nunca visto pelo pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted')

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
            plt.show()

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
    df_res.to_latex('results/ml_resultados.tex', float_format='%.3f', index=False)
    return df_res


def treinar_mlp_sklearn(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.3,
    pca_n_components=None,
) -> MLPClassifier:
    """
    Treina MLPClassifier com GridSearchCV.
    SMOTE, StandardScaler e PCA aplicados apenas no conjunto de treino.
    GridSearchCV usa f1_weighted (mais adequado que accuracy para dados desbalanceados).
    """
    X, y = _preparar_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    X_train_pp, X_test_pp, y_train_pp = _preproc_train_test(
        X_train, X_test, y_train,
        use_smote=True, random_state=random_state, pca_n_components=pca_n_components,
    )

    param_grid = {
        'hidden_layer_sizes': [(128, 64)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
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
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(best.loss_curve_, color='blue', label='Perda no Treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.title('Curva de Perda do MLP')
    plt.legend()
    plt.grid()
    plt.show()

    return best


# ── Redes Neurais (Keras) ─────────────────────────────────────────────────────

def preparar_dados_keras(
    df: pd.DataFrame,
    target: str = 'manobra',
    test_size: float = 0.3,
    random_state: int = 42,
    pca_n_components=None,
) -> tuple:
    """
    Prepara dados para MLP Keras.
    Split estratificado → SMOTE + Scaler (→ PCA) apenas no treino.
    Retorna (X_train, X_test, y_train_cat, y_test_cat).
    """
    from tensorflow.keras.utils import to_categorical

    X, y = _preparar_xy(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train, X_test, y_train = _preproc_train_test(
        X_train, X_test, y_train,
        use_smote=True, random_state=random_state, pca_n_components=pca_n_components,
    )
    return X_train, X_test, to_categorical(y_train), to_categorical(y_test)


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


# ── ISL — Índice de Segurança Lateral ────────────────────────────────────────

# Colunas excluídas das features do modelo ISL (isl_alto é o target, não feature)
_COLS_EXCLUIR_ISL = _COLS_EXCLUIR  # isl_mean/max/class/alto já estão em _COLS_EXCLUIR


def treinar_modelo_isl(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
    pca_n_components=None,
) -> pd.DataFrame:
    """
    Treina modelos clássicos para prever o ISL (Índice de Segurança Lateral)
    antes de o veículo entrar na curva.

    Target: ``isl_alto`` (1 = ISL ≥ 0.8 — alto risco lateral, 0 = baixo/médio)
    Features: janela pré-curva — estatísticas de vehicle_speed, engine_rpm,
              accel_x, accel_y (mean/std/median/max/min/slope/cv) +
              contexto do trajeto.

    Metodologia idêntica a ``aplicar_modelos_ml``:
      split estratificado → SMOTE+Scaler por fold (ImbPipeline) → CV → teste final.

    Retorna DataFrame com métricas por classificador e salva em
    ``results/isl_resultados.tex``.
    """
    df_valid = df.dropna(subset=['isl_alto']).copy()
    n_alto  = int(df_valid['isl_alto'].sum())
    n_baixo = int((df_valid['isl_alto'] == 0).sum())
    print(f"  ISL — amostras válidas: {len(df_valid)}  (alto: {n_alto} | baixo/médio: {n_baixo})")

    feature_cols = [c for c in df_valid.columns if c not in _COLS_EXCLUIR_ISL]
    df_clean = df_valid[feature_cols + ['isl_alto']].dropna()
    X = df_clean[feature_cols].values
    y = df_clean['isl_alto'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state, max_iter=1000),
        'SVM':                 svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão':   DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória':  RandomForestClassifier(random_state=random_state),
        'Rede Neural (MLP)':   MLPClassifier(
            activation='relu', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    linhas = []
    for nome, clf in modelos.items():
        pipe = _construir_pipeline(clf, random_state=random_state, pca_n_components=pca_n_components)

        cv_res = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={'acc': 'accuracy', 'f1': 'f1_weighted'},
        )
        cv_acc = cv_res['test_acc'].mean()
        cv_f1  = cv_res['test_f1'].mean()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted')

        print(
            f'{nome:30s}  CV Acc: {cv_acc:.3f}  CV F1: {cv_f1:.3f}  |  '
            f'Teste Acc: {acc:.3f}  F1: {f1:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}'
        )

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=['Baixo/Médio', 'Alto'],
                        yticklabels=['Baixo/Médio', 'Alto'])
            plt.title(f'ISL — {nome}')
            plt.ylabel('Real')
            plt.xlabel('Prevista')
            os.makedirs('results', exist_ok=True)
            plt.savefig(f'results/isl_cm_{nome}.pdf', bbox_inches='tight')
            plt.show()

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
    df_res.to_latex('results/isl_resultados.tex', float_format='%.3f', index=False)
    return df_res
