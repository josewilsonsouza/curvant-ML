"""
CurvantML — Deep learning multi-tarefa em PyTorch.
Substitui os modelos Keras (MLP, GRU, LSTM).

Arquitetura principal: MultiTaskMLP com encoder compartilhado e cinco heads:
  - isl_value       (regressão, MSELoss)
  - isl_class       (3 classes, CrossEntropyLoss)
  - manobra_accel_curva, manobra_lateral_curva, manobra_ziguezague_curva (binário, BCELoss)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.models import _COLS_EXCLUIR, _base_route

_ISL_ENCODE = {'baixo': 0, 'medio': 1, 'alto': 2}

_TARGETS_BINARIOS = [
    'manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva',
]
_TARGETS_TODOS = ['isl_value', 'isl_class'] + _TARGETS_BINARIOS


class CurvantDataset(Dataset):
    """Dataset multi-tarefa para o MLP. X é float32; targets são float32 ou int64."""

    def __init__(self, X: np.ndarray, targets: dict):
        self.X       = torch.FloatTensor(X)
        self.targets = {}
        for k, v in targets.items():
            if k == 'isl_class':
                self.targets[k] = torch.LongTensor(v.astype(np.int64))
            else:
                self.targets[k] = torch.FloatTensor(v.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}


class MultiTaskMLP(nn.Module):
    """
    MLP multi-tarefa: encoder compartilhado (256→128) + cinco heads independentes.
    Entrada: vetor de features F1-F5 (normalizado externamente).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.3),
        )
        self.head_isl_value = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.head_isl_class = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        self.head_accel   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_lateral = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_zz      = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        return {
            'isl_value':                self.head_isl_value(z).squeeze(-1),
            'isl_class':                self.head_isl_class(z),
            'manobra_accel_curva':      self.head_accel(z).squeeze(-1),
            'manobra_lateral_curva':    self.head_lateral(z).squeeze(-1),
            'manobra_ziguezague_curva': self.head_zz(z).squeeze(-1),
        }


def _preparar_targets(df: pd.DataFrame) -> dict:
    """Extrai e codifica os alvos disponíveis no DataFrame."""
    targets = {}
    if 'isl_value' in df.columns:
        targets['isl_value'] = df['isl_value'].fillna(0.0).values
    if 'isl_class' in df.columns:
        targets['isl_class'] = df['isl_class'].map(_ISL_ENCODE).fillna(0).values
    for col in _TARGETS_BINARIOS:
        if col in df.columns:
            targets[col] = df[col].fillna(0).values
    return targets


def treinar_multitask_mlp(
    df: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    test_size: float = 0.3,
    random_state: int = 42,
    lambdas: dict = None,
) -> MultiTaskMLP:
    """
    Treina o MultiTaskMLP com split por id_route.

    lambdas: pesos por loss. Se None, todos 1.0.
    Retorna o modelo treinado.
    """
    if lambdas is None:
        lambdas = {k: 1.0 for k in _TARGETS_TODOS}

    feature_cols = [c for c in df.columns if c not in _COLS_EXCLUIR]
    id_routes    = df['id_route'].astype(str)
    base_rotas   = list({_base_route(r) for r in id_routes})
    train_base, test_base = train_test_split(base_rotas, test_size=test_size, random_state=random_state)
    train_base_set = set(train_base)
    test_base_set  = set(test_base)

    target_cols_presentes = [c for c in _TARGETS_TODOS if c in df.columns]
    all_cols = feature_cols + target_cols_presentes + ['id_route']

    df_train = df[id_routes.map(_base_route).isin(train_base_set)][all_cols]
    df_test  = df[id_routes.map(_base_route).isin(test_base_set)][all_cols]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].fillna(0.0).values.astype(np.float32))
    X_test  = scaler.transform(df_test[feature_cols].fillna(0.0).values.astype(np.float32))

    train_targets = _preparar_targets(df_train)
    test_targets  = _preparar_targets(df_test)

    train_loader = DataLoader(
        CurvantDataset(X_train, train_targets),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )

    model     = MultiTaskMLP(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fns = {
        'isl_value':                nn.MSELoss(),
        'isl_class':                nn.CrossEntropyLoss(),
        'manobra_accel_curva':      nn.BCELoss(),
        'manobra_lateral_curva':    nn.BCELoss(),
        'manobra_ziguezague_curva': nn.BCELoss(),
    }

    historico_loss = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, targets_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss_sum = None
            for key, fn in loss_fns.items():
                if key not in targets_batch:
                    continue
                y   = targets_batch[key]
                lam = lambdas.get(key, 1.0)
                l   = lam * fn(preds[key], y)
                loss_sum = l if loss_sum is None else loss_sum + l
            if loss_sum is not None:
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()
        loss_medio = epoch_loss / len(train_loader)
        historico_loss.append(loss_medio)
        if (epoch + 1) % 20 == 0:
            print(f"    Época {epoch + 1}/{epochs} — Loss médio: {loss_medio:.4f}")

    from graphics.visualization import plotar_loss_pytorch
    plotar_loss_pytorch(historico_loss, nome='multitask_mlp')

    model.eval()
    with torch.no_grad():
        preds_test = model(torch.FloatTensor(X_test))

    print("\n  MultiTaskMLP — Métricas (teste, split por rota):")
    for key in target_cols_presentes:
        if key not in preds_test or key not in test_targets:
            continue
        p = preds_test[key].numpy()
        y = test_targets[key]
        if key == 'isl_value':
            print(f"    isl_value   R²: {r2_score(y, p):.4f}")
        elif key == 'isl_class':
            pred_labels = np.argmax(p, axis=1)
            print(f"    isl_class   F1-macro: {f1_score(y.astype(int), pred_labels, average='macro', zero_division=0):.4f}")
        else:
            pred_labels = (p > 0.5).astype(int)
            print(f"    {key:<30s} F1: {f1_score(y.astype(int), pred_labels, average='weighted', zero_division=0):.4f}")

    return model
