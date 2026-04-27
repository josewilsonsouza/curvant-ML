"""
CurvantML — Deep learning multi-tarefa em PyTorch.

MultiTaskMLP: encoder compartilhado (256→128) + quatro heads independentes:
  - isl_class                (3 classes, CrossEntropyLoss)
  - manobra_accel_curva      (binário, BCELoss)
  - manobra_lateral_curva    (binário, BCELoss)
  - manobra_ziguezague_curva (binário, BCELoss)

Regressão de série temporal (curve_accel_y_max) tratada separadamente
em modelos_ts.py, que opera sobre dados brutos da janela pré-curva.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.models import _COLS_EXCLUIR, _base_route

_ISL_ENCODE = {'baixo': 0, 'medio': 1, 'alto': 2}
_ISL_LABELS  = {0: 'baixo', 1: 'medio', 2: 'alto'}

_TARGETS_BINARIOS = [
    'manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva',
]
_TARGETS_TODOS = ['isl_class'] + _TARGETS_BINARIOS


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
    MLP multi-tarefa: encoder compartilhado (256→128) + quatro heads independentes.
    Entrada: vetor de features F1-F5 (normalizado externamente).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.3),
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
            'isl_class':                self.head_isl_class(z),
            'manobra_accel_curva':      self.head_accel(z).squeeze(-1),
            'manobra_lateral_curva':    self.head_lateral(z).squeeze(-1),
            'manobra_ziguezague_curva': self.head_zz(z).squeeze(-1),
        }


def _preparar_targets(df: pd.DataFrame) -> dict:
    """Extrai e codifica os alvos disponíveis no DataFrame."""
    targets = {}
    if 'isl_class' in df.columns:
        targets['isl_class'] = df['isl_class'].map(_ISL_ENCODE).fillna(0).values
    for col in _TARGETS_BINARIOS:
        if col in df.columns:
            targets[col] = df[col].fillna(0).values
    return targets


_LAMBDAS_PADRAO = {
    'isl_class':                1.0,
    'manobra_accel_curva':      1.0,
    'manobra_lateral_curva':    1.0,
    'manobra_ziguezague_curva': 1.0,
}


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

    lambdas: pesos por loss. Se None, usa _LAMBDAS_PADRAO (todos 1.0).
    Scheduler: ReduceLROnPlateau(patience=50, factor=0.5).
    Retorna o modelo treinado.
    """
    if lambdas is None:
        lambdas = _LAMBDAS_PADRAO.copy()

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50,
    )

    loss_fns = {
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
        scheduler.step(loss_medio)
        if (epoch + 1) % 20 == 0:
            lr_atual = optimizer.param_groups[0]['lr']
            print(f"    Época {epoch + 1}/{epochs} — Loss: {loss_medio:.4f}  LR: {lr_atual:.2e}")

    from graphics.visualization import plotar_loss_pytorch
    plotar_loss_pytorch(historico_loss, nome='multitask_mlp')

    model.eval()
    with torch.no_grad():
        preds_test = model(torch.FloatTensor(X_test))

    os.makedirs('results', exist_ok=True)
    print("\n  MultiTaskMLP — Métricas (teste, split por rota):")
    for key in target_cols_presentes:
        if key not in preds_test or key not in test_targets:
            continue
        p = preds_test[key].numpy()
        y = test_targets[key]

        if key == 'isl_class':
            pred_labels = np.argmax(p, axis=1)
            f1 = f1_score(y.astype(int), pred_labels, average='macro', zero_division=0)
            print(f"    isl_class   F1-macro: {f1:.4f}")
            cm = confusion_matrix(y.astype(int), pred_labels)
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap='Blues')
            ticks = [_ISL_LABELS[i] for i in range(3)]
            ax.set_xticks(range(3)); ax.set_xticklabels(ticks)
            ax.set_yticks(range(3)); ax.set_yticklabels(ticks)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9)
            ax.set_xlabel('Prevista'); ax.set_ylabel('Real')
            ax.set_title(f'PyTorch — isl_class  F1={f1:.3f}')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig('results/pytorch_cm_isl_class.pdf', bbox_inches='tight')
            plt.close(fig)

        else:
            pred_labels = (p > 0.5).astype(int)
            f1 = f1_score(y.astype(int), pred_labels, average='weighted', zero_division=0)
            print(f"    {key:<30s} F1: {f1:.4f}")
            cm = confusion_matrix(y.astype(int), pred_labels)
            fig, ax = plt.subplots(figsize=(3, 3))
            im = ax.imshow(cm, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=10)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['Segura', 'Risco'])
            ax.set_yticks([0, 1]); ax.set_yticklabels(['Segura', 'Risco'])
            ax.set_xlabel('Prevista'); ax.set_ylabel('Real')
            short = key.replace('manobra_', '').replace('_curva', '')
            ax.set_title(f'PyTorch — {short}  F1={f1:.3f}')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(f'results/pytorch_cm_{short}.pdf', bbox_inches='tight')
            plt.close(fig)

    return model
