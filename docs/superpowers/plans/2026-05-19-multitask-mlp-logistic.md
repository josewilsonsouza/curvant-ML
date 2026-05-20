# MultiTaskMLP Logistic Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir o encoder 256→128 do MultiTaskMLP por um encoder de 1 camada (64 unidades) com Dropout(0.5) e weight_decay L2, e adicionar `manobra_combinado_curva` como 5º head.

**Architecture:** Encoder compartilhado `Input → BN → Linear(64) → ReLU → Dropout(0.5)` seguido de 5 heads lineares diretos (sem sub-camada intermediária). Regularização via `weight_decay=0.01` no Adam.

**Tech Stack:** PyTorch, pandas, scikit-learn, config.yaml (carregar_config)

---

## Files

| Arquivo | Ação | O que muda |
|---|---|---|
| `config.yaml` | Modify | `dropout: 0.3→0.5`, adicionar `weight_decay: 0.01` |
| `src/models_pytorch.py` | Modify | Nova arquitetura, novo head, `weight_decay` param |
| `src/pipeline.py` | Modify | Passar `weight_decay` para `treinar_multitask_mlp` |

---

### Task 1: Atualizar config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Alterar dropout e adicionar weight_decay**

Localizar a seção `multitask_mlp` em `config.yaml` (em torno da linha 109) e aplicar as mudanças:

```yaml
  multitask_mlp:
    epochs: 400
    batch_size: 32
    lr: 0.005
    dropout: 0.5        # era 0.3
    weight_decay: 0.01  # novo — L2 equivalente ao C da regressão logística
```

---

### Task 2: Simplificar arquitetura do MultiTaskMLP

**Files:**
- Modify: `src/models_pytorch.py:56-83`

- [ ] **Step 1: Substituir a classe MultiTaskMLP**

Substituir o bloco completo da classe `MultiTaskMLP` (linhas 56–83) pelo código abaixo. O encoder passa de 2 camadas (256→128) para 1 camada (64 unidades). Os heads passam de `Linear→ReLU→Linear` para um único `Linear`. O `Sigmoid` sai dos heads e vai para o `forward` via `torch.sigmoid`, para consistência.

```python
class MultiTaskMLP(nn.Module):
    """
    MLP multi-tarefa: encoder compartilhado (64 unidades) + cinco heads lineares.
    Arquitetura próxima de regressão logística multi-task com leve capacidade não-linear.
    """

    def __init__(self, n_features: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_isl_class = nn.Linear(64, 3)
        self.head_accel     = nn.Linear(64, 1)
        self.head_lateral   = nn.Linear(64, 1)
        self.head_zz        = nn.Linear(64, 1)
        self.head_combinado = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        return {
            'isl_class':                self.head_isl_class(z),
            'manobra_accel_curva':      torch.sigmoid(self.head_accel(z)).squeeze(-1),
            'manobra_lateral_curva':    torch.sigmoid(self.head_lateral(z)).squeeze(-1),
            'manobra_ziguezague_curva': torch.sigmoid(self.head_zz(z)).squeeze(-1),
            'manobra_combinado_curva':  torch.sigmoid(self.head_combinado(z)).squeeze(-1),
        }
```

---

### Task 3: Adicionar manobra_combinado_curva aos targets e losses

**Files:**
- Modify: `src/models_pytorch.py:31-34` (constantes)
- Modify: `src/models_pytorch.py:97-102` (`_LAMBDAS_PADRAO`)
- Modify: `src/models_pytorch.py:156-161` (`loss_fns` dentro de `treinar_multitask_mlp`)

- [ ] **Step 1: Adicionar target à lista `_TARGETS_BINARIOS`**

Localizar (linha ~31) e substituir:

```python
_TARGETS_BINARIOS = [
    'manobra_accel_curva', 'manobra_lateral_curva', 'manobra_ziguezague_curva',
    'manobra_combinado_curva',
]
```

- [ ] **Step 2: Adicionar entrada em `_LAMBDAS_PADRAO`**

Localizar (linha ~97) e substituir:

```python
_LAMBDAS_PADRAO = {
    'isl_class':                1.0,
    'manobra_accel_curva':      1.0,
    'manobra_lateral_curva':    1.0,
    'manobra_ziguezague_curva': 1.0,
    'manobra_combinado_curva':  1.0,
}
```

- [ ] **Step 3: Adicionar entrada em `loss_fns` dentro de `treinar_multitask_mlp`**

Localizar o dict `loss_fns` (linha ~156) e substituir:

```python
    loss_fns = {
        'isl_class':                nn.CrossEntropyLoss(),
        'manobra_accel_curva':      nn.BCELoss(),
        'manobra_lateral_curva':    nn.BCELoss(),
        'manobra_ziguezague_curva': nn.BCELoss(),
        'manobra_combinado_curva':  nn.BCELoss(),
    }
```

---

### Task 4: Adicionar weight_decay à função de treinamento

**Files:**
- Modify: `src/models_pytorch.py:105-114` (assinatura de `treinar_multitask_mlp`)
- Modify: `src/models_pytorch.py:151-154` (instanciação do optimizer)

- [ ] **Step 1: Adicionar parâmetro `weight_decay` à assinatura**

Localizar a assinatura de `treinar_multitask_mlp` (linha ~105) e substituir:

```python
def treinar_multitask_mlp(
    df: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.5,
    weight_decay: float = 0.01,
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
```

- [ ] **Step 2: Passar weight_decay ao Adam**

Localizar a linha do `torch.optim.Adam` (linha ~151) e substituir:

```python
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

---

### Task 5: Atualizar pipeline.py para passar weight_decay

**Files:**
- Modify: `src/pipeline.py:250-258`

- [ ] **Step 1: Passar weight_decay na chamada de treinar_multitask_mlp**

Localizar o bloco `treinar_multitask_mlp(` em `etapa_pytorch` (linha ~250) e substituir:

```python
    treinar_multitask_mlp(
        features_df,
        epochs=pt_cfg.get('epochs', 100),
        batch_size=pt_cfg.get('batch_size', 32),
        lr=pt_cfg.get('lr', 1e-3),
        dropout=pt_cfg.get('dropout', 0.5),
        weight_decay=pt_cfg.get('weight_decay', 0.01),
        test_size=cfg['ml']['test_size'],
        random_state=cfg['ml']['random_state'],
    )
```

---

### Task 6: Verificação end-to-end

**Files:** nenhum

- [ ] **Step 1: Executar o pipeline com --pytorch**

```bash
python scripts/run.py --pytorch
```

Saída esperada no terminal:
- Épocas impressas a cada 20 iterações sem erros de shape ou CUDA
- Seção `MultiTaskMLP — Métricas (teste, split por rota):` com 5 linhas (isl_class + 4 binárias incluindo `manobra_combinado_curva`)
- Arquivos gerados em `results/`: `pytorch_cm_combinado.pdf`, `pytorch_cm_accel.pdf`, etc.

Se aparecer erro `RuntimeError: Expected more than 1 value per channel`, o `drop_last=True` no DataLoader já previne isso — verificar se o batch_size é menor que o tamanho do split de treino.

- [ ] **Step 2: Verificar F1 de manobra_combinado_curva**

Comparar o F1 impresso para `manobra_combinado_curva` com o F1 da Regressão Logística clássica (impresso na seção de modelos clássicos ao rodar `python scripts/run.py`). Critério de sucesso: F1 ≥ LR clássica.
