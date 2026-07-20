# Seleção explícita de dataset: plano de implementação

> Spec: docs/superpowers/specs/2026-07-20-selecao-dataset-design.md

**Objetivo:** escolher o dataset por `config.yaml` (`data.dataset`) ou flag `--data`,
com auto-detecção como fallback e cache intermediário separado por dataset.

**Arquitetura:** tudo concentrado em `curvant/cli.py` (resolução do dataset, caches e
flags), mais a nova seção `data:` no `config.yaml` e docs no `CLAUDE.md`.

## Restrições globais

- Sem suíte de testes no projeto; verificação por chamadas diretas às funções e pela CLI.
- Nunca rodar `git add/commit/push`.
- Sem travessão nem traços decorativos em código e docs.
- Python >= 3.10 (pode usar `str | None`).

### Tarefa 1: resolução do dataset e cache por nome em `curvant/cli.py`

**Arquivos:** modificar `curvant/cli.py`.

**Produz:** `_nome_base(valor) -> str`, `_resolver_dataset(cfg, arg_data) -> (caminho, eh_limpo, nome)`,
`_input_bruto(cfg) -> str`, `_carregar_features(..., arg_data=None)`.

- [x] Substituir as constantes `_CACHE_ANALYSIS`/`_CACHE_FEATURES` e a lista dupla de
  candidatos por:

```python
_RAW_CANDIDATOS = [
    'data/eletro_rjdf_serra_rjmgba_janeiro_agda.parquet',
    'data/eletro_rjdf_serra_rjmgba_janeiro.parquet',
    'data/eletro_rjdf_serra.parquet',
]


def _nome_base(valor: str) -> str:
    """Caminho ou nome -> nome base do dataset (sem pasta, sem .parquet, sem _clean)."""
    nome = os.path.basename(valor)
    if nome.endswith('.parquet'):
        nome = nome[: -len('.parquet')]
    if nome.endswith('_clean'):
        nome = nome[: -len('_clean')]
    return nome


def _resolver_dataset(cfg: dict, arg_data: str | None) -> tuple[str, bool, str]:
    """Qual arquivo usar: --data > config data.dataset > auto-detecção.

    Devolve (caminho, eh_limpo, nome_base), preferindo o _clean quando existir.
    """
    escolhido = arg_data or (cfg.get('data') or {}).get('dataset')
    if escolhido:
        nome = _nome_base(str(escolhido))
        for caminho, limpo in ((f'data/{nome}_clean.parquet', True), (f'data/{nome}.parquet', False)):
            if os.path.exists(caminho):
                return caminho, limpo, nome
        disponiveis = sorted({
            _nome_base(f) for f in os.listdir('data')
            if f.endswith('.parquet') and not f.startswith('.')
        })
        raise FileNotFoundError(
            f"Dataset '{nome}' não encontrado em data/. Disponíveis: {', '.join(disponiveis)}"
        )
    for bruto in _RAW_CANDIDATOS:
        caminho = f'data/{_nome_base(bruto)}_clean.parquet'
        if os.path.exists(caminho):
            return caminho, True, _nome_base(bruto)
    for bruto in _RAW_CANDIDATOS:
        if os.path.exists(bruto):
            return bruto, False, _nome_base(bruto)
    raise FileNotFoundError("Nenhum arquivo de dados encontrado em data/")


def _input_bruto(cfg: dict) -> str:
    escolhido = (cfg.get('data') or {}).get('dataset')
    if escolhido:
        caminho = f'data/{_nome_base(str(escolhido))}.parquet'
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo bruto do dataset não encontrado: {caminho}")
        return caminho
    caminho = next((p for p in _RAW_CANDIDATOS if os.path.exists(p)), None)
    if caminho is None:
        raise FileNotFoundError(
            "Nenhum arquivo bruto encontrado em data/. Esperado um de: "
            + ', '.join(_RAW_CANDIDATOS)
        )
    return caminho
```

- [x] Em `_carregar_features`, trocar a assinatura para
  `(cfg, feat_cfg, rebuild, mostrar_risco=False, arg_data=None)` e o bloco `candidates`
  por:

```python
    data_path, is_clean, nome = _resolver_dataset(cfg, arg_data)
    cache_analysis = f'data/.cache_df_analysis_{nome}.parquet'
    cache_features = f'data/.cache_features_df_{nome}.parquet'
```

  usando `cache_analysis`/`cache_features` no lugar das constantes no resto da função e
  citando o `nome` no print do cache.

- [x] Em `_add_alvo`, nova flag:

```python
    p.add_argument('--data', type=str, default=None,
                   help='Dataset a usar (nome ou caminho em data/); sobrescreve config data.dataset')
```

- [x] Em `main`: no ramo do preprocess, `in_path = args.input or _input_bruto(carregar_config())`;
  no ramo de treino, passar `arg_data=args.data` para `_carregar_features`.

### Tarefa 2: seção `data:` no `config.yaml`

- [x] Inserir logo após o bloco `huggingface:`:

```yaml
# Dataset do pipeline: nome base ou caminho em data/ (com ou sem _clean/.parquet).
# null = auto-detecta o primeiro disponível, preferindo o _clean. A flag --data sobrescreve.
data:
  dataset: null
```

### Tarefa 3: apagar caches órfãos

- [x] Remover de `data/`: `.cache_df_analysis.parquet`, `.cache_features_df.parquet`,
  `.cache_features_df_modo1.parquet` e `.cache_features_df_{modo}.parquet`.

### Tarefa 4: docs no `CLAUDE.md`

- [x] Seção Dados: explicar a ordem `--data` > `data.dataset` > auto-detecção, o cache
  por dataset (`.cache_df_analysis_<nome>.parquet`) e manter a lista como fallback.
- [x] Bloco de utilidades dos Comandos: exemplo com `--data`.
- [x] Tabela de parâmetros: linha `config | data | dataset | null | dataset usado; null = auto`.

### Tarefa 5: verificação

- [x] `python -c` chamando `_resolver_dataset` com: config vazio (auto: deve dar
  `agda_clean`), `arg_data='eletro_rjdf_serra'` (deve dar `serra_clean`),
  `arg_data='data/eletro_rjdf_serra_rjmgba_janeiro_agda_clean.parquet'` (deve normalizar
  para o mesmo nome base) e `arg_data='nao_existe'` (deve listar os disponíveis).
- [x] `cvt tab risk --help` mostra `--data`; `cvt preprocess` com `data.dataset` setado
  usa o bruto certo (checar só a resolução via `_input_bruto`, sem rodar a limpeza).
