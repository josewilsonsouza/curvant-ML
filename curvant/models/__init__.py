"""
Pacote de modelos do CurvantML.

Reúne os três módulos de modelagem e re-exporta a API pública, para que os
imports antigos no estilo ``from src.models import <nome>`` continuem funcionando
após a reorganização em pacote.

  tabular.py      — modelos sobre a matriz de features (clássicos, Optuna, ISL,
                    regressão, baseline) + splits por rota
  temporais.py    — modelos de série temporal (Ridge/RF/XGBoost + LSTM/GRU/CNN/MLP)
  multitarefa.py  — MLP multi-tarefa em PyTorch (vários alvos de uma vez)

A lista do que é/não é feature fica em curvant/driving/features.py (NAO_FEATURES,
colunas_features), perto de onde as features são criadas.
"""

from curvant.driving.features import colunas_features
from .tabular import (
    aplicar_modelos_ml,
    aplicar_modelos_ml_otimizados,
    treinar_modelo_isl,
    avaliar_baseline_fisico,
    _split_por_rota,
    _split_rota_generico,
    _base_route,
)
from .temporais import treinar_regressao_ts
from .multitarefa import treinar_multitask_mlp
