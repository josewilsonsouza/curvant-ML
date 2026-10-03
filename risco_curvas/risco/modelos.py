"""Classificadores avaliados e referências de comparação.

Os modelos tabulares recebem estatísticas da janela de observação; a LSTM
recebe a sequência. As referências usam só o que está disponível no fim da
observação: a classe mais frequente do treino e uma árvore rasa sobre a
aceleração lateral prevista ``v²/R``, com a velocidade no fim da observação e o
raio da curva à frente.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import Config

TABULARES = ("RF", "XGB", "LGB")
SEQUENCIAIS = ("LSTM",)
REFERENCIAS = ("Classe majoritária", "Física v²/R")


def tabular(nome: str, cfg: Config, semente: int) -> Pipeline:
    """Pipeline com imputação pela mediana, padronização e o classificador pedido."""
    m = cfg["modelos"]
    if nome == "RF":
        modelo = RandomForestClassifier(random_state=semente, n_jobs=-1, **m["rf"])
    elif nome == "XGB":
        modelo = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=semente, n_jobs=-1, **m["xgb"])
    elif nome == "LGB":
        modelo = LGBMClassifier(objective="multiclass", random_state=semente, verbose=-1, n_jobs=-1, **m["lgb"])
    else:
        raise ValueError(nome)
    return Pipeline([("imputacao", SimpleImputer(strategy="median")), ("escala", StandardScaler()), ("modelo", modelo)])


def entradas_fisicas(v_fim_kmh: np.ndarray, raio_m: np.ndarray) -> np.ndarray:
    """Aceleração lateral prevista, velocidade no fim da observação e raio da curva."""
    v = np.nan_to_num(np.asarray(v_fim_kmh, dtype=float) / 3.6)
    raio = np.asarray(raio_m, dtype=float)
    return np.c_[v**2 / raio, raio, v]


def referencia(nome: str, cfg: Config, semente: int):
    if nome == "Classe majoritária":
        return DummyClassifier(strategy="most_frequent")
    if nome == "Física v²/R":
        return DecisionTreeClassifier(random_state=semente, **cfg["modelos"]["baseline_fisico"])
    raise ValueError(nome)


class ClassificadorLSTM:
    """Duas camadas LSTM com parada antecipada pela perda de validação.

    Ausentes são preenchidos pela mediana de cada canal no treino, e os canais
    são padronizados com média e desvio do treino.
    """

    def __init__(self, cfg: Config, semente: int):
        self.p = cfg["modelos"]["lstm"]
        self.semente = semente

    def _preparar(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32).copy()
        faltando = ~np.isfinite(X)
        X[faltando] = np.take(self.mediana_, np.nonzero(faltando)[-1])
        return (X - self.media_) / self.desvio_

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "ClassificadorLSTM":
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam

        canais = X.reshape(-1, X.shape[-1])
        self.mediana_ = np.nan_to_num(np.nanmedian(canais, axis=0)).astype(np.float32)
        preenchido = np.where(np.isfinite(canais), canais, self.mediana_)
        self.media_ = preenchido.mean(axis=0).astype(np.float32)
        desvio = preenchido.std(axis=0)
        self.desvio_ = np.where(desvio > 0, desvio, 1.0).astype(np.float32)

        tf.keras.utils.set_random_seed(self.semente)
        p = self.p
        self.rede_ = Sequential([
            Input(shape=X.shape[1:]),
            LSTM(p["unidades"], return_sequences=True),
            Dropout(p["dropout"]),
            LSTM(p["unidades"]),
            Dense(3, activation="softmax"),
        ])
        self.rede_.compile(optimizer=Adam(p["taxa_aprendizado"]), loss="sparse_categorical_crossentropy")
        self.rede_.fit(
            self._preparar(X), y,
            validation_data=(self._preparar(X_val), y_val),
            epochs=p["epocas"], batch_size=p["lote"], verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=p["paciencia"], restore_best_weights=True)],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rede_.predict(self._preparar(X), verbose=0).argmax(axis=1)
