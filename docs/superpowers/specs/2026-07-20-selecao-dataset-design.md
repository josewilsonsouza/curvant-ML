# Seleção explícita de dataset (config + CLI)

Data: 2026-07-20. Status: aprovado.

## Problema

O `cvt` escolhe o arquivo de dados por listas fixas no código (`_RAW_CANDIDATOS` e
`candidates` em `curvant/cli.py`). Para usar um conjunto novo é preciso editar o código.
Além disso o cache intermediário é um par único e global
(`.cache_df_analysis.parquet` / `.cache_features_df.parquet`), validado só por data de
modificação: ao trocar para um dataset mais antigo, o cache do dataset anterior parece
válido e o pipeline usaria dados errados.

## Decisão

Seleção por config com sobrescrita pontual na CLI, e cache separado por dataset.
Prioridade: flag `--data` > campo `data.dataset` do `config.yaml` > auto-detecção
atual (fallback, comportamento de hoje).

## Desenho

### Resolução do dataset

Função nova `_resolver_dataset(cfg, arg_data)` em `curvant/cli.py`.

O valor aceito (na flag ou no config) pode ser um caminho (`data/x.parquet`) ou só o
nome (`x`). Tudo é normalizado para o nome base: sem pasta, sem `.parquet`, sem sufixo
`_clean`. A partir do nome base vale a regra atual: usa `data/<nome>_clean.parquet` se
existir, senão o bruto `data/<nome>.parquet` com o aviso de dados não limpos. Se nenhum
dos dois existir, erro claro listando os parquet disponíveis em `data/`.

Com `data.dataset: null` e sem `--data`, mantém a auto-detecção pelas listas de
candidatos, que continuam no código só como fallback.

### cvt preprocess

Sem `--input`, limpa o bruto do dataset selecionado (`data/<nome>.parquet`) em vez de
usar `_RAW_CANDIDATOS`. O `--input` explícito continua valendo como hoje.

### Cache por dataset

Os intermediários passam a `.cache_df_analysis_<nome>.parquet` e
`.cache_features_df_<nome>.parquet`. Validação por data de modificação e `--rebuild`
continuam iguais, mas cada dataset tem seu par, então trocar e voltar não reprocessa
nem mistura. Os caches antigos em `data/` são apagados na implementação:
`.cache_df_analysis.parquet`, `.cache_features_df.parquet`,
`.cache_features_df_modo1.parquet` e `.cache_features_df_{modo}.parquet` (os dois
últimos são restos de uma versão velha do código).

### Config e docs

Nova seção no `config.yaml`, logo após o bloco `huggingface:`:

```yaml
data:
  dataset: null   # null = auto-detecta; ou nome/caminho, ex: eletro_rjdf_serra_rjmgba_janeiro_agda
```

Flag `--data` nos subcomandos `tab` e `seq`. O `preprocess` não ganha `--data`: ele já
tem `--input`, que segue como sobrescrita direta. CLAUDE.md atualizado na seção Dados e
na tabela de parâmetros.

## Fora do escopo

Nada muda nos algoritmos (`driving/`, `models/`), nas features nem nos alvos. A
whitelist de features e o `features.yaml` não são tocados.
