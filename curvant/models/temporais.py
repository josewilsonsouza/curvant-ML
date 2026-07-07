import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier as _XGBClassifier
    from xgboost import XGBRegressor as _XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from curvant.constants import G as _G_TS, MU as _MU_TS, ISL_BAIXO, ISL_ALTO
from curvant.models.tabular import _base_route

_SENSORS_PADRAO = ['vehicle_speed', 'accel_x', 'accel_y', 'engine_rpm']

_ISL_ENCODE = {'baixo': 0, 'medio': 1, 'alto': 2}

# Targets de classificação e número de classes
_TARGETS_BINARIOS_TS  = {
    'manobra_combinado_curva', 'manobra_accel_curva',
    'manobra_lateral_curva', 'manobra_ziguezague_curva',
}
_TARGETS_MULTICLASS_TS = {'isl_class': 3}
_TARGETS_CLASSIF_TS    = _TARGETS_BINARIOS_TS | set(_TARGETS_MULTICLASS_TS)


def _v_para_isl_class(v_kmh: np.ndarray, raios: np.ndarray) -> np.ndarray:
    """Converte velocidade (km/h) + raio (m) → classe ISL (0=baixo,1=medio,2=alto)."""
    isl = (v_kmh / 3.6) ** 2 / (raios * _G_TS * _MU_TS)
    return np.where(isl < ISL_BAIXO, 0, np.where(isl < ISL_ALTO, 1, 2)).astype(int)


def _baseline_persistencia_vcritica(y_test, raios_test, preditores: dict, outdir: str = 'results') -> None:
    """
    Baselines de persistência para v_critica (sem aprendizado), avaliados no MESMO
    conjunto de teste e com o MESMO raio (f4_raio_min) dos modelos.

    preditores: dict {nome: array_v_previsto_km_h}. Tipicamente:
      - 'v_pred_kinematica'   : extrapolação cinemática com desaceleração de conforto
      - 'vel_aprox_media'     : média da velocidade na janela pré-curva (persistência pura)

    Quantifica quanto os modelos agregam além de "a velocidade na curva ≈ a de
    aproximação". Salva o resultado em outdir/baseline.tex.
    """
    print(f"\n  [BASELINE persistência]  v_critica (sem treino)")
    linhas = []
    for nome, pred in preditores.items():
        r2  = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        linha = f"  {nome:22s}  R²: {r2:7.4f}  MAE: {mae:6.3f} km/h"
        registro = {'Baseline': nome, 'R²': round(r2, 4), 'MAE (km/h)': round(mae, 3)}
        if raios_test is not None:
            cls_pred = _v_para_isl_class(pred, raios_test)
            cls_true = _v_para_isl_class(y_test, raios_test)
            f1  = f1_score(cls_true, cls_pred, average='macro', zero_division=0)
            acc = accuracy_score(cls_true, cls_pred)
            linha += f"  | isl_class F1: {f1:.4f}  Acc: {acc:.4f}"
            registro['isl_class F1'] = round(f1, 4)
            registro['isl_class Acc'] = round(acc, 4)
        print(linha)
        linhas.append(registro)

    if linhas:
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(linhas).to_latex(
            os.path.join(outdir, 'baseline.tex'),
            index=False,
            caption='Baseline de persistência para v\\_critica (sem aprendizado), no mesmo '
                    'conjunto de teste por rota dos modelos.',
            label='tab:baseline_vcritica',
            position='h',
        )


def _detectar_task(target: str, task_cfg: str) -> tuple[str, int]:
    """Retorna (task, n_classes): task='regression'|'classification', n_classes=1|2|3."""
    if task_cfg == 'classification' or target in _TARGETS_CLASSIF_TS:
        if target in _TARGETS_MULTICLASS_TS:
            return 'classification', _TARGETS_MULTICLASS_TS[target]
        return 'classification', 2
    return 'regression', 1


# Extração de sequências

def extrair_sequencias_precurva(
    df_analysis: pd.DataFrame,
    features_df: pd.DataFrame,
    sensors: list[str],
    n_timesteps: int,
    target: str = 'v_critica',
    scalares_extras: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Para cada curva em features_df extrai a série temporal da janela pré-curva
    de df_analysis (usando time_inicio/time_fim) e reamostrada para n_timesteps.

    Canais temporais (axis=2):
        0 … n_sensors-1  : sensores OBD (vehicle_speed, accel_x, accel_y, engine_rpm …)
        n_sensors         : distancia_restante - distância até a curva em cada timestep
                            (d_max_in_window - distancia_acumulada, normalizada para [0,1]).
                            Canal decrescente: zero no instante de entrada na curva.

    scalares_extras: colunas de features_df adicionadas como canais constantes
    (mesmo valor em todos os timesteps), APÓS os canais temporais.

    Retorna:
        X          - (n, n_timesteps, n_sensors + 1 [+ n_extras]) float32
        y          - (n,) float32
        id_route   - (n,) str
        n_temporal - número de canais temporais (n_sensors + 1)
    """
    sensors_disponiveis = [s for s in sensors if s in df_analysis.columns]
    extras = [c for c in (scalares_extras or []) if c in features_df.columns]
    x_new  = np.linspace(0, 1, n_timesteps)

    has_dist = 'distancia_acumulada' in df_analysis.columns
    dist_cols = sensors_disponiveis + ['time_sec'] + (['distancia_acumulada'] if has_dist else [])

    grupos = {
        rota: sub[dist_cols].reset_index(drop=True)
        for rota, sub in df_analysis.groupby('id_route')
    }

    n_temporal = len(sensors_disponiveis) + (1 if has_dist else 0)
    sequences, targets, rotas = [], [], []

    for _, row in features_df.iterrows():
        if pd.isna(row.get(target)):
            continue
        t0 = row.get('time_inicio')
        t1 = row.get('time_fim')
        if pd.isna(t0) or pd.isna(t1):
            continue

        sub = grupos.get(str(row['id_route']))
        if sub is None:
            continue

        mask   = (sub['time_sec'] >= t0) & (sub['time_sec'] <= t1)
        janela_df = sub.loc[mask, sensors_disponiveis].copy()
        janela_df = janela_df.ffill().bfill().fillna(0.0)
        janela = janela_df.values.astype(np.float32)

        if len(janela) < 2:
            continue

        x_old     = np.linspace(0, 1, len(janela))
        resampled = np.stack(
            [np.interp(x_new, x_old, janela[:, j]) for j in range(janela.shape[1])],
            axis=1,
        )  # (n_timesteps, n_sensors)

        # distancia_restante: d_max - d(t), decreasing toward zero at curve entry
        if has_dist:
            d_raw = sub.loc[mask, 'distancia_acumulada'].values.astype(np.float64)
            d_interp = np.interp(x_new, x_old, d_raw)
            d_restante = d_interp[-1] - d_interp          # decreasing to ~0
            d_range = d_restante.max() - d_restante.min()
            if d_range > 0:
                d_restante = d_restante / d_range          # normalize [0, 1]
            resampled = np.concatenate(
                [resampled, d_restante[:, np.newaxis].astype(np.float32)],
                axis=1,
            )

        for col in extras:
            val = row.get(col)
            val = 0.0 if pd.isna(val) else float(val)
            resampled = np.concatenate(
                [resampled, np.full((n_timesteps, 1), val, dtype=np.float32)],
                axis=1,
            )

        sequences.append(resampled)
        targets.append(float(row[target]))
        rotas.append(str(row['id_route']))

    if not sequences:
        raise ValueError("Nenhuma sequência válida extraída. Verifique time_inicio/time_fim em features_df.")

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), np.array(rotas), n_temporal


# Modelos neurais

class GRURegressor(nn.Module):
    def __init__(self, n_sensors: int, n_scalars: int = 0, hidden_size: int = 64, n_layers: int = 1,
                 dropout: float = 0.3, n_out: int = 1):
        super().__init__()
        self.n_sensors = n_sensors
        self.gru = nn.GRU(
            n_sensors, hidden_size, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_scalars, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq   = x[:, :, :self.n_sensors]
        scalars = x[:, 0, self.n_sensors:]
        _, h    = self.gru(x_seq)
        z       = torch.cat([h[-1], scalars], dim=1)
        return self.head(z).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, n_sensors: int, n_scalars: int = 0, hidden_size: int = 64, n_layers: int = 1,
                 dropout: float = 0.3, n_out: int = 1):
        super().__init__()
        self.n_sensors = n_sensors
        self.lstm = nn.LSTM(
            n_sensors, hidden_size, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_scalars, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq   = x[:, :, :self.n_sensors]
        scalars = x[:, 0, self.n_sensors:]
        _, (h, _) = self.lstm(x_seq)
        z       = torch.cat([h[-1], scalars], dim=1)
        return self.head(z).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, n_sensors: int, n_scalars: int = 0, channels: list[int] = None,
                 kernel_size: int = 3, dropout: float = 0.3, n_out: int = 1):
        super().__init__()
        self.n_sensors = n_sensors
        if channels is None:
            channels = [32, 64]
        layers, in_ch = [], n_sensors
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.ReLU(), nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv    = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(in_ch * 4 + n_scalars, 32), nn.ReLU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq   = x[:, :, :self.n_sensors].permute(0, 2, 1)
        scalars = x[:, 0, self.n_sensors:]
        z       = self.pool(self.conv(x_seq)).flatten(1)
        z       = self.dropout(z)
        z       = torch.cat([z, scalars], dim=1)
        return self.head(z).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, n_sensors: int, n_timesteps: int, n_scalars: int = 0,
                 hidden_size: int = 128, dropout: float = 0.3, n_out: int = 1):
        super().__init__()
        self.n_sensors = n_sensors
        flat_in = n_sensors * n_timesteps + n_scalars
        self.fc1  = nn.Linear(flat_in, hidden_size)
        self.fc2  = nn.Linear(hidden_size, 64)
        self.fc3  = nn.Linear(64, n_out)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq   = x[:, :, :self.n_sensors].flatten(1)
        scalars = x[:, 0, self.n_sensors:]
        z = torch.cat([x_seq, scalars], dim=1)
        z = self.drop(self.act(self.fc1(z)))
        z = self.drop(self.act(self.fc2(z)))
        return self.fc3(z).squeeze(-1)


_MODELOS_TS   = {'gru': GRURegressor, 'lstm': LSTMRegressor, 'cnn1d': CNN1DRegressor, 'mlp': MLPRegressor}
_CLASSICOS_TS = {'linear', 'rf', 'xgboost'}
_TODOS_TS     = set(_MODELOS_TS) | _CLASSICOS_TS


def _flatten_para_classico(X: np.ndarray, n_temporal: int) -> np.ndarray:
    """
    Converte (n, T, C) em features tabulares para modelos clássicos:
      - canais temporais [0:n_temporal]: mean, std, max, min, slope → n_temporal × 5
      - canais escalares [n_temporal:] : valor único (constante no tempo) → n_scalar
    Total: n_temporal * 5 + n_scalar  (ex.: 5×5 + 14 = 39 features)
    """
    n, T, C = X.shape
    temporal = X[:, :, :n_temporal]

    t   = np.linspace(0, 1, T)
    t_c = t - t.mean()
    slopes = (temporal * t_c[None, :, None]).sum(axis=1) / (t_c ** 2).sum()

    parts = [
        temporal.mean(axis=1),
        temporal.std(axis=1),
        temporal.max(axis=1),
        temporal.min(axis=1),
        slopes,
    ]
    if C > n_temporal:
        parts.append(X[:, 0, n_temporal:])  # escalares constantes - basta o 1º timestep

    return np.concatenate(parts, axis=1).astype(np.float32)


# Clássicos

def _build_classico(nome: str, rnd: int, task: str, y_train: np.ndarray | None = None):
    if task == 'classification':
        if nome == 'linear':
            return LogisticRegression(max_iter=1000, random_state=rnd, class_weight='balanced')
        if nome == 'rf':
            return RandomForestClassifier(n_estimators=200, random_state=rnd, n_jobs=-1, class_weight='balanced')
        if nome == 'xgboost':
            if not _HAS_XGB:
                raise ImportError("xgboost não instalado.")
            scale = 1.0
            if y_train is not None:
                n_pos = (y_train == 1).sum()
                n_neg = (y_train == 0).sum()
                scale = float(n_neg / n_pos) if n_pos > 0 else 1.0
            return _XGBClassifier(n_estimators=200, random_state=rnd, verbosity=0,
                                  eval_metric='logloss', scale_pos_weight=scale)
    else:
        if nome == 'linear':
            return Ridge()
        if nome == 'rf':
            return RandomForestRegressor(n_estimators=200, random_state=rnd, n_jobs=-1)
        if nome == 'xgboost':
            if not _HAS_XGB:
                raise ImportError("xgboost não instalado.")
            return _XGBRegressor(n_estimators=200, random_state=rnd, verbosity=0)
    raise ValueError(f"Modelo clássico desconhecido: '{nome}'")


def _treinar_um_classico(
    nome: str,
    X_train_flat: np.ndarray,
    X_test_flat: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target: str,
    plot: bool,
    rnd: int,
    task: str = 'regression',
    n_classes: int = 1,
    raios_test: np.ndarray | None = None,
    outdir: str = 'results',
) -> dict:
    label = {'linear': 'Ridge' if task == 'regression' else 'LogisticReg',
             'rf': 'RandomForest', 'xgboost': 'XGBoost'}[nome]
    clf   = _build_classico(nome, rnd, task, y_train=y_train.astype(int) if task == 'classification' else None)

    if task == 'classification':
        clf.fit(X_train_flat, y_train.astype(int))
        y_pred = clf.predict(X_test_flat)
        avg    = 'macro' if n_classes > 2 else 'binary'
        f1     = f1_score(y_test.astype(int), y_pred, average=avg, zero_division=0)
        acc    = accuracy_score(y_test.astype(int), y_pred)
        print(f"  {label:<15s} F1: {f1:.4f}  Acc: {acc:.4f}")

        if plot:
            os.makedirs(outdir, exist_ok=True)
            cm  = confusion_matrix(y_test.astype(int), y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap='Blues')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9)
            ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
            ax.set_title(f'TS {label} - {target}\nF1={f1:.3f}')
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'cm_{nome}_{target}.pdf'), bbox_inches='tight')
            plt.close(fig)

        return {'Modelo': label, 'F1': f1, 'Acc': acc}

    else:
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        r2     = r2_score(y_test, y_pred)
        mae    = mean_absolute_error(y_test, y_pred)
        print(f"  {label:<15s} R²: {r2:.4f}  MAE: {mae:.4f} km/h")

        cls_pred_isl = cls_true_isl = None
        if target == 'v_critica' and raios_test is not None:
            cls_pred_isl = _v_para_isl_class(y_pred, raios_test)
            cls_true_isl = _v_para_isl_class(y_test, raios_test)
            f1_isl  = f1_score(cls_true_isl, cls_pred_isl, average='macro', zero_division=0)
            acc_isl = accuracy_score(cls_true_isl, cls_pred_isl)
            print(f"    -> isl_class via v_critica  F1-macro: {f1_isl:.4f}  Acc: {acc_isl:.4f}")

        if plot:
            os.makedirs(outdir, exist_ok=True)
            unit = 'km/h' if target == 'v_critica' else 'm/s²'
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, alpha=0.4, s=15)
            lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lim, lim, 'r--', lw=1)
            ax.set_xlabel(f'{target} real ({unit})')
            ax.set_ylabel(f'{target} previsto ({unit})')
            ax.set_title(f'{label} - R²={r2:.3f}  MAE={mae:.3f}')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'scatter_{nome}.pdf'), bbox_inches='tight')
            plt.close(fig)

            if cls_pred_isl is not None:
                labels_isl = ['baixo', 'medio', 'alto']
                cm = confusion_matrix(cls_true_isl, cls_pred_isl, labels=[0, 1, 2])
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(cm, cmap='Blues')
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9)
                ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels_isl)
                ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labels_isl)
                ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
                ax.set_title(f'{label} - isl_class via v_critica\nF1={f1_isl:.3f}')
                fig.colorbar(im, ax=ax); fig.tight_layout()
                fig.savefig(os.path.join(outdir, f'cm_{nome}_isl.pdf'), bbox_inches='tight')
                plt.close(fig)

        return {'Modelo': label, 'R²': r2, 'MAE (km/h)': mae}


# Neural

def _treinar_um_neural(
    model_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target: str,
    plot: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    n_layers: int,
    dropout: float,
    channels: list,
    kernel: int,
    n_ts: int,
    task: str = 'regression',
    n_classes: int = 1,
    patience: int = 40,
    val_size: float = 0.15,
    n_seq_sensor: int = 0,
    weight_decay: float = 0.0,
    raios_test: np.ndarray | None = None,
    outdir: str = 'results',
) -> nn.Module:
    n_total   = X_train.shape[2]
    n_sens    = n_seq_sensor if n_seq_sensor > 0 else n_total
    n_scalars = n_total - n_sens
    n_out     = 1 if (task == 'regression' or n_classes == 2) else n_classes

    # Split de validação (dentro do treino) para early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42,
    )

    if task == 'regression':
        y_scaler    = StandardScaler()
        y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
        y_tensor_tr  = torch.FloatTensor(y_tr_scaled)
        y_tensor_val = torch.FloatTensor(y_val_scaled)
    elif n_classes == 2:
        y_tensor_tr  = torch.FloatTensor(y_tr.astype(np.float32))
        y_tensor_val = torch.FloatTensor(y_val.astype(np.float32))
    else:
        y_tensor_tr  = torch.LongTensor(y_tr.astype(np.int64))
        y_tensor_val = torch.LongTensor(y_val.astype(np.int64))

    X_tensor_val = torch.FloatTensor(X_val)
    effective_batch = batch_size
    if len(X_tr) < batch_size:
        effective_batch = max(4, len(X_tr) // 2)
        print(f"    [AVISO] {len(X_tr)} amostras de treino < batch_size={batch_size}; "
              f"usando batch_size={effective_batch}")
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), y_tensor_tr),
        batch_size=effective_batch, shuffle=True, drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(
            f"DataLoader vazio após ajuste ({len(X_tr)} amostras). "
            "Para --ts, janela_distancia deve ser >= 50 m para garantir sequências válidas."
        )

    if model_type in ('gru', 'lstm'):
        model = _MODELOS_TS[model_type](n_sens, n_scalars=n_scalars, hidden_size=hidden,
                                        n_layers=n_layers, dropout=dropout, n_out=n_out)
    elif model_type == 'mlp':
        model = MLPRegressor(n_sens, n_timesteps=n_ts, n_scalars=n_scalars,
                             hidden_size=hidden, dropout=dropout, n_out=n_out)
    else:
        model = _MODELOS_TS[model_type](n_sens, n_scalars=n_scalars, channels=channels,
                                        kernel_size=kernel, dropout=dropout, n_out=n_out)

    if task == 'regression':
        loss_fn = nn.MSELoss()
    elif n_classes == 2:
        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        pos_w = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        classes, counts = np.unique(y_tr.astype(int), return_counts=True)
        w = np.zeros(n_classes, dtype=np.float32)
        for cls, cnt in zip(classes, counts):
            w[cls] = 1.0 / cnt
        w /= w.sum()
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(w))

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    loss_label = 'MSE' if task == 'regression' else 'Loss'

    hist_train, hist_val = [], []
    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        hist_train.append(ep_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_tensor_val), y_tensor_val).item()
        hist_val.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop - época {epoch + 1}  val: {val_loss:.4f}  "
                      f"(melhor: {best_val:.4f})")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Época {epoch + 1}/{epochs} - train: {ep_loss:.4f}  val: {val_loss:.4f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        raw_out = model(torch.FloatTensor(X_test))

    if plot:
        os.makedirs(outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(hist_train, label='treino', alpha=0.8)
        ax.plot(hist_val,   label='val',    alpha=0.8)
        ax.axvline(len(hist_train) - no_improve, color='r', linestyle='--', linewidth=0.8, label='best')
        ax.set_xlabel('Época'); ax.set_ylabel(loss_label)
        ax.set_title(f'Loss - {model_type.upper()} {task} {target}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'loss_{model_type}_{target}.pdf'), bbox_inches='tight')
        plt.close(fig)

    if task == 'regression':
        y_pred = y_scaler.inverse_transform(raw_out.numpy().reshape(-1, 1)).ravel()
        r2     = r2_score(y_test, y_pred)
        mae    = mean_absolute_error(y_test, y_pred)
        unit = 'km/h' if target == 'v_critica' else 'm/s²'
        print(f"  {model_type.upper()} - R²: {r2:.4f}  MAE: {mae:.4f} {unit}")

        cls_pred_isl = cls_true_isl = None
        if target == 'v_critica' and raios_test is not None:
            cls_pred_isl = _v_para_isl_class(y_pred, raios_test)
            cls_true_isl = _v_para_isl_class(y_test, raios_test)
            f1_isl  = f1_score(cls_true_isl, cls_pred_isl, average='macro', zero_division=0)
            acc_isl = accuracy_score(cls_true_isl, cls_pred_isl)
            print(f"    -> isl_class via v_critica  F1-macro: {f1_isl:.4f}  Acc: {acc_isl:.4f}")

        if plot:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, alpha=0.4, s=15)
            lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lim, lim, 'r--', lw=1)
            ax.set_xlabel(f'{target} real ({unit})')
            ax.set_ylabel(f'{target} previsto ({unit})')
            ax.set_title(f'{model_type.upper()} - R²={r2:.3f}  MAE={mae:.3f}')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'scatter_{model_type}_{target}.pdf'), bbox_inches='tight')
            plt.close(fig)

            if cls_pred_isl is not None:
                labels_isl = ['baixo', 'medio', 'alto']
                cm = confusion_matrix(cls_true_isl, cls_pred_isl, labels=[0, 1, 2])
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(cm, cmap='Blues')
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9)
                ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels_isl)
                ax.set_yticks([0, 1, 2]); ax.set_yticklabels(labels_isl)
                ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
                ax.set_title(f'{model_type.upper()} - isl_class via v_critica\nF1={f1_isl:.3f}')
                fig.colorbar(im, ax=ax); fig.tight_layout()
                fig.savefig(os.path.join(outdir, f'cm_{model_type}_isl.pdf'), bbox_inches='tight')
                plt.close(fig)

    else:
        avg = 'macro' if n_classes > 2 else 'binary'
        if n_classes == 2:
            y_pred = (torch.sigmoid(raw_out).numpy() > 0.5).astype(int)
        else:
            y_pred = raw_out.numpy().argmax(axis=1)

        f1  = f1_score(y_test.astype(int), y_pred, average=avg, zero_division=0)
        acc = accuracy_score(y_test.astype(int), y_pred)
        print(f"  {model_type.upper()} - F1: {f1:.4f}  Acc: {acc:.4f}")

        if plot:
            cm  = confusion_matrix(y_test.astype(int), y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap='Blues')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9)
            ax.set_xlabel('Previsto'); ax.set_ylabel('Real')
            ax.set_title(f'TS {model_type.upper()} - {target}\nF1={f1:.3f}')
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'cm_{model_type}_{target}.pdf'), bbox_inches='tight')
            plt.close(fig)

    return model


# Orquestração

def treinar_regressao_ts(
    df_analysis: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: dict,
    feat_cfg: dict,
    plot: bool = True,
    outdir: str = 'results',
) -> None:
    """
    Treina modelos de série temporal (regressão ou classificação) sobre
    dados brutos da janela pré-curva.

    Hiperparâmetros em config.yaml > temporais; canais de entrada (sensors e
    scalares_extras) em features.yaml > flags.velocity.
    """
    ts_cfg    = cfg.get('temporais', {})
    vel_cfg   = feat_cfg.get('flags', {}).get('velocity', {})
    raw_model = ts_cfg.get('model', 'gru')
    modelos   = [raw_model.lower()] if isinstance(raw_model, str) else [m.lower() for m in raw_model]

    invalidos = [m for m in modelos if m not in _TODOS_TS]
    if invalidos:
        raise ValueError(f"Modelos desconhecidos: {invalidos}. Válidos: {sorted(_TODOS_TS)}")

    target       = ts_cfg.get('target', 'v_critica')
    task_cfg     = ts_cfg.get('task', 'auto')
    task, n_classes = _detectar_task(target, task_cfg)

    sensors    = vel_cfg.get('sensors', _SENSORS_PADRAO)
    n_ts       = int(ts_cfg.get('n_timesteps', 50))
    epochs     = int(ts_cfg.get('epochs', 100))
    batch_size = int(ts_cfg.get('batch_size', 32))
    lr         = float(ts_cfg.get('lr', 1e-3))
    hidden     = int(ts_cfg.get('hidden_size', 64))
    n_layers   = int(ts_cfg.get('n_layers', 1))
    dropout    = float(ts_cfg.get('dropout', 0.3))
    channels   = ts_cfg.get('channels', [32, 64])
    kernel     = int(ts_cfg.get('kernel_size', 3))
    test_size  = float(cfg.get('ml', {}).get('test_size', 0.3))
    rnd        = int(cfg.get('ml', {}).get('random_state', 42))
    scalares_cfg = vel_cfg.get('scalares_extras') or []
    patience     = int(ts_cfg.get('early_stopping_patience', 40))
    val_size_nn  = float(ts_cfg.get('val_size', 0.15))
    weight_decay = float(ts_cfg.get('weight_decay', 0.01))

    print(f"  Tarefa: {task.upper()} | Modelos: {modelos} | target: {target}")
    print(f"  Timesteps: {n_ts} | Sensores: {sensors}")

    # Codifica isl_class (string → int) antes da extração
    if target == 'isl_class' and not pd.api.types.is_numeric_dtype(features_df[target]):
        features_df = features_df.copy()
        features_df[target] = features_df[target].map(_ISL_ENCODE)

    # Canal autoregressivo: target da curva anterior (por rota base)
    prev_col = f'prev_{target}'
    if prev_col not in features_df.columns:
        ft = features_df.copy().sort_values(['id_route', 'time_inicio'])
        ft[prev_col] = ft.groupby(ft['id_route'].apply(_base_route))[target].shift(1)
        features_df  = ft

    scalares_todos = list(dict.fromkeys(scalares_cfg + [prev_col]))
    disponiveis    = [c for c in scalares_todos if c in features_df.columns]
    print(f"  Canais escalares extras: {disponiveis}")

    X, y, rotas, n_seq_sensor = extrair_sequencias_precurva(
        df_analysis, features_df, sensors, n_ts, target,
        scalares_extras=disponiveis,
    )
    n_sensors = X.shape[2]
    print(f"  {len(X)} sequências  |  canais: {n_sensors} "
          f"({n_seq_sensor} temporais + {len(disponiveis)} escalares)")

    # Cap de percentil (só para regressão - remove valores clipados pelo preprocessing)
    if task == 'regression':
        cap_pct = ts_cfg.get('target_cap_percentil')
        if cap_pct is not None:
            cap_val  = np.percentile(y, cap_pct)
            mask_cap = y <= cap_val
            X, y, rotas = X[mask_cap], y[mask_cap], rotas[mask_cap]
            print(f"  Cap {cap_pct}º percentil: <= {cap_val:.3f} -> {len(y)} sequências mantidas")

    # Split estratificado pela mediana/moda do target por rota
    route_y: dict[str, list] = {}
    for r, yi in zip(rotas, y):
        route_y.setdefault(_base_route(r), []).append(yi)

    base_rotas    = list(route_y.keys())
    if len(base_rotas) < 2:
        raise ValueError(f"Número insuficiente de rotas ({len(base_rotas)}) para realizar o split de treino/teste. Verifique as configurações de features ou filtros de dataset.")

    route_stats   = np.array([np.median(route_y[r]) for r in base_rotas])
    n_bins        = min(3, len(base_rotas) // 2)
    
    if n_bins >= 1:
        bins          = np.quantile(route_stats, np.linspace(0, 1, n_bins + 1))
        strata        = np.digitize(route_stats, bins[1:-1])
        _, counts     = np.unique(strata, return_counts=True)
        if any(c < 2 for c in counts):
            strata = None
    else:
        strata = None

    train_base, _ = train_test_split(base_rotas, test_size=test_size, random_state=rnd, stratify=strata)
    train_mask    = np.array([_base_route(r) in set(train_base) for r in rotas])

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    if task == 'regression':
        print(f"  Split - treino: {len(y_train)} (média={y_train.mean():.3f})  "
              f"teste: {len(y_test)} (média={y_test.mean():.3f})")
    else:
        vals, cnts = np.unique(y_test.astype(int), return_counts=True)
        dist = ' | '.join(f'cls{v}:{c}' for v, c in zip(vals, cnts))
        print(f"  Split - treino: {len(y_train)}  teste: {len(y_test)} ({dist})")

    # Extrai raios do teste antes da normalização (canal escalar constante)
    raios_test = None
    if target == 'v_critica' and 'f4_raio_min' in disponiveis:
        raio_idx   = n_seq_sensor + disponiveis.index('f4_raio_min')
        raios_test = X_test[:, 0, raio_idx].copy()   # valores originais, não normalizados

    # Baselines de persistência: v_critica ≈ v_pred_kinematica e ≈ vel. média de
    # aproximação (mesmo split/raio dos modelos). Extrair antes da normalização.
    if target == 'v_critica':
        preditores = {}
        if 'v_pred_kinematica' in disponiveis:
            vp_idx = n_seq_sensor + disponiveis.index('v_pred_kinematica')
            preditores['v_pred_kinematica'] = X_test[:, 0, vp_idx].copy()
        if 'vehicle_speed' in sensors:
            vs_idx = sensors.index('vehicle_speed')
            preditores['vel_aprox_media'] = X_test[:, :, vs_idx].mean(axis=1).copy()
        if preditores:
            _baseline_persistencia_vcritica(y_test, raios_test, preditores, outdir=outdir)

    # Normaliza sensores (fit apenas no treino) - compartilhado por todos os modelos
    for j in range(n_sensors):
        sc = StandardScaler()
        X_train[:, :, j] = sc.fit_transform(X_train[:, :, j])
        X_test[:, :, j]  = sc.transform(X_test[:, :, j])

    X_train_flat = _flatten_para_classico(X_train, n_seq_sensor)
    X_test_flat  = _flatten_para_classico(X_test,  n_seq_sensor)
    print(f"  Features clássicos: {X_train_flat.shape[1]} "
          f"({n_seq_sensor} sensores × 5 stats + {X_train_flat.shape[1] - n_seq_sensor * 5} escalares)")

    for model_type in modelos:
        print(f"\n  [{model_type.upper()}]")
        if model_type in _CLASSICOS_TS:
            _treinar_um_classico(
                model_type, X_train_flat, X_test_flat, y_train, y_test,
                target=target, plot=plot, rnd=rnd, task=task, n_classes=n_classes,
                raios_test=raios_test, outdir=outdir,
            )
        else:
            _treinar_um_neural(
                model_type, X_train, X_test, y_train, y_test,
                target=target, plot=plot,
                epochs=epochs, batch_size=batch_size, lr=lr,
                hidden=hidden, n_layers=n_layers, dropout=dropout,
                channels=channels, kernel=kernel, n_ts=n_ts,
                task=task, n_classes=n_classes,
                patience=patience, val_size=val_size_nn,
                n_seq_sensor=n_seq_sensor, weight_decay=weight_decay,
                raios_test=raios_test, outdir=outdir,
            )
