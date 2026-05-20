# Design: MultiTaskMLP — Arquitetura Logística Melhorada

**Data:** 2026-05-19
**Status:** Aprovado

## Contexto

Testes empíricos mostram que Regressão Logística supera o MultiTaskMLP atual (encoder 256→128, 2 camadas, Dropout 0.3). A hipótese é que o problema tem fronteira de decisão aproximadamente linear e a arquitetura atual superparametriza — muitos graus de liberdade para um dataset tabular pequeno.

## Objetivo

Substituir a arquitetura atual por uma versão mais enxuta inspirada em regressão logística, mantendo o benefício do multi-task learning. Adicionar `manobra_combinado_curva` como 5º head para permitir comparação direta com a LR clássica.

## Arquitetura

### Antes (removida)
```
Input → BN → Linear(256) → ReLU → Dropout(0.3) → Linear(128) → ReLU → Dropout(0.3)
     → head: Linear(128→64) → ReLU → Linear(64→out)   [×4]
```

### Depois (nova)
```
Input → BN → Linear(64) → ReLU → Dropout(0.5) → Linear(64→out)   [×5 heads]
```

Encoder compartilhado de 1 camada (64 unidades). Cada head é uma única camada linear aplicada diretamente à saída do encoder — sem camada intermediária nos heads. Toda a capacidade não-linear concentra-se no encoder, facilitando a regularização via `weight_decay`.

## Heads

| Head | Loss | Tipo | Status |
|---|---|---|---|
| `isl_class` | CrossEntropyLoss | 3 classes | existente |
| `manobra_accel_curva` | BCELoss | binário | existente |
| `manobra_lateral_curva` | BCELoss | binário | existente |
| `manobra_ziguezague_curva` | BCELoss | binário | existente |
| `manobra_combinado_curva` | BCELoss | binário | **novo** |

## Regularização

- `weight_decay: 0.01` no Adam — equivalente ao penalizador L2 da LR (C=100)
- `dropout: 0.5` no encoder (era 0.3)
- Combinação dropout + weight_decay previne overfitting em dataset tabular pequeno

## Alterações por arquivo

### `src/models_pytorch.py`
- `MultiTaskMLP.__init__`: remover encoder 256→128 e sub-heads 64; adicionar encoder Linear(64) + Dropout(0.5); simplificar cada head para `nn.Linear(64, out)`
- `_TARGETS_BINARIOS`: adicionar `'manobra_combinado_curva'`
- `treinar_multitask_mlp`: adicionar parâmetro `weight_decay: float = 0.01`; passar ao `Adam`; gerar CM e F1 para o novo head

### `config.yaml`
- `dropout`: `0.3` → `0.5`
- `weight_decay: 0.01` (novo campo)

### `src/pipeline.py`
- Ler `weight_decay` de `config['neural_networks']['multitask_mlp']` e passar para `treinar_multitask_mlp`

## O que não muda

- Nome da classe `MultiTaskMLP` e da função `treinar_multitask_mlp`
- Split por rota base (`_split_por_rota`), DataLoader com `drop_last=True`
- Scheduler `ReduceLROnPlateau`
- Geração de plots de loss e confusion matrices
- Parâmetros `epochs`, `batch_size`, `lr` no config

## Critério de sucesso

F1 do head `manobra_combinado_curva` igual ou superior ao F1 da Regressão Logística clássica no mesmo split de teste.

## Não fazer commits