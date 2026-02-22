import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

_COLS_EXCLUIR = [
    'manobra', 'id_route', 'id_trecho_curvo', 'time_inicio', 'time_fim',
    'manobra_accel_perigo', 'manobra_dir_perigosa', 'manobra_zigue_zague',
]


def _preparar_xy(
    df: pd.DataFrame,
    target: str = 'manobra',
    random_state: int = 42,
    use_smote: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Retorna (X_scaled, y, scaler) após SMOTE e StandardScaler."""
    feature_cols = [c for c in df.columns if c not in _COLS_EXCLUIR]
    X = df[feature_cols].values
    y = df[target].values

    if use_smote:
        X, y = SMOTE(random_state=random_state).fit_resample(X, y)

    scaler = StandardScaler()
    return scaler.fit_transform(X), y, scaler


# ── Modelos clássicos ─────────────────────────────────────────────────────────

def aplicar_modelos_ml(
    df: pd.DataFrame,
    plot_cm: bool = False,
    random_state: int = 42,
    test_size: float = 0.3,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Treina e avalia modelos clássicos de ML com validação cruzada k-fold.
    Retorna DataFrame pivotado com acurácia por bloco e classificador.
    """
    X_scaled, y, _ = _preparar_xy(df, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    modelos = {
        'Regressão Logística': LogisticRegression(random_state=random_state),
        'SVM': svm.SVC(kernel='linear', random_state=random_state),
        'Árvore de Decisão': DecisionTreeClassifier(random_state=random_state),
        'Floresta Aleatória': RandomForestClassifier(random_state=random_state),
        'Rede Neural (MLP)': MLPClassifier(
            activation='relu', learning_rate='constant', solver='adam',
            hidden_layer_sizes=(128, 64), max_iter=2000, random_state=random_state,
        ),
    }

    resultados = []
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X_scaled, y, cv=cv_folds)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        print(f'{nome} — Acurácia: {accuracy_score(y_test, y_pred):.4f}')

        for fold, score in enumerate(scores, start=1):
            resultados.append({'Bloco': f'Bloco {fold}', 'Classificador': nome, 'Acurácia': score})

        if plot_cm:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Segura', 'Risco'],
                        yticklabels=['Segura', 'Risco'])
            plt.title(nome)
            plt.ylabel('Real')
            plt.xlabel('Prevista')
            plt.savefig(f'matriz_confusao_{nome}.pdf', bbox_inches='tight')
            plt.show()

    df_pivot = (
        pd.DataFrame(resultados)
        .pivot(index='Bloco', columns='Classificador', values='Acurácia')
    )
    df_pivot.to_latex('ml_resultados.tex', float_format='%.2f')
    return df_pivot.reset_index()


def treinar_mlp_sklearn(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.3,
) -> MLPClassifier:
    """
    Treina um MLPClassifier (sklearn) com GridSearchCV e plota matriz de confusão
    e curva de perda. Reporta possível overfitting.
    """
    X_scaled, y, _ = _preparar_xy(df, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    param_grid = {
        'hidden_layer_sizes': [(128, 64)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate': ['constant'],
    }
    grid = GridSearchCV(
        MLPClassifier(max_iter=2000, random_state=random_state),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f'Melhores parâmetros: {grid.best_params_}')

    y_pred = best.predict(X_test)
    train_acc = best.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Acurácia — Treino: {train_acc:.4f} | Teste: {test_acc:.4f}')
    if train_acc - test_acc > 0.1:
        print('Aviso: possível overfitting detectado.')

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
) -> tuple:
    """Prepara dados com SMOTE + StandardScaler + to_categorical para modelos Keras."""
    from tensorflow.keras.utils import to_categorical

    X_scaled, y, _ = _preparar_xy(df, target=target, random_state=random_state)
    y_cat = to_categorical(y)
    return train_test_split(X_scaled, y_cat, test_size=test_size, random_state=random_state)


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
) -> tuple:
    """Prepara dados no formato 3D (samples, timesteps, features) para GRU."""
    from tensorflow.keras.utils import to_categorical

    X_scaled, y, _ = _preparar_xy(df, target=target, random_state=random_state, use_smote=False)
    num_features = X_scaled.shape[1]
    num_samples = X_scaled.shape[0] // janela_tempo

    X_3d = X_scaled[:num_samples * janela_tempo].reshape(num_samples, janela_tempo, num_features)
    y_cat = to_categorical(y[:num_samples])

    return *train_test_split(X_3d, y_cat, test_size=test_size, random_state=random_state), num_features


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
) -> tuple:
    """Prepara dados no formato 3D (samples, timesteps, features) para LSTM.
    Idêntico ao GRU — reutilizável para qualquer modelo recorrente."""
    from tensorflow.keras.utils import to_categorical

    X_scaled, y, _ = _preparar_xy(df, target=target, random_state=random_state, use_smote=False)
    num_features = X_scaled.shape[1]
    num_samples = X_scaled.shape[0] // janela_tempo

    X_3d = X_scaled[:num_samples * janela_tempo].reshape(num_samples, janela_tempo, num_features)
    y_cat = to_categorical(y[:num_samples])

    return *train_test_split(X_3d, y_cat, test_size=test_size, random_state=random_state), num_features


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
