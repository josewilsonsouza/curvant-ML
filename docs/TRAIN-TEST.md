# Treino, validação e teste

Existem **duas partições físicas** (treino e teste) e a **validação é obtida reparticionando o
treino**. Não existe um "conjunto de validação" guardado à parte.

## Nível 1: treino e teste, separados por rota

O primeiro corte acontece antes de qualquer coisa, sobre os dados crus
([`_split_por_rota`](../curvant/models/tabular.py)):

- **Treino ~70%**, **teste ~30%** (`config.yaml > ml.test_size`).
- O corte é **por rota**, não por curva.

Esse detalhe é o mais importante da metodologia. Todas as curvas de uma mesma gravação vão
inteiras para o treino **ou** inteiras para o teste, nunca divididas. Se metade das curvas de uma
rota fosse para o treino e metade para o teste, o modelo veria condições quase idênticas dos dois
lados (mesmo motorista, mesmo trecho, mesmo carro) e a métrica ficaria inflada.

O sufixo `_p<N>`, que marca sub trajetos de uma gravação cortada em gaps, é removido antes do
corte, então pedaços da mesma gravação também ficam do mesmo lado.

O **teste é intocado**: separado antes do SMOTE, do scaler e do PCA, e usado uma única vez, no
fim, para medir o desempenho.

## Nível 2: validação, dentro do treino

O treino não é usado inteiro de uma vez. Como se obtém a validação depende da representação,
porque as duas precisam de coisas diferentes.

### Tabular (`cvt tab`): validação cruzada rotativa

O treino é dividido em **5 dobras** (`config.yaml > ml.cv_folds`) com `StratifiedGroupKFold`. Em
cada rodada, 4 dobras treinam e 1 valida, girando 5 vezes. A métrica reportada é a média.

- **`Group`**: as dobras também respeitam a rota, então nenhuma rota aparece ao mesmo tempo no
  treino e na validação de um fold. É o mesmo princípio do nível 1, agora dentro do treino.
- **`Stratified`**: cada dobra mantém a proporção das classes, para a validação não cair numa
  fatia sem casos da classe minoritária.

Na regressão tabular (`cvt tab velocity`) a estratificação não faz sentido, porque o alvo é
contínuo, então usa-se `GroupKFold` puro.

### Sequência (`cvt seq`): hold-out fixo

As redes neurais treinam por épocas e precisam de um conjunto de validação **fixo** para
acompanhar a curva de perda e decidir a hora de parar (early stopping). Uma validação cruzada
rotativa não serve aqui, porque o critério de parada precisa de uma referência estável entre
épocas.

Então uma fração fixa do treino é separada (`config.yaml > temporais.val_size`, 30% por padrão) e
usada só para monitorar a perda. O treino para quando ela deixa de melhorar por
`early_stopping_patience` épocas.

No modo multitarefa esse corte é **estratificado pela faixa de ISL**, para a validação não ficar
pobre em alguma faixa. A parada antecipada olha o **MSE da velocidade**, que é o alvo principal,
o que protege a velocidade de regredir por causa da cabeça de classe.

> [!NOTE]
> Esse split de validação da sequência usa um sorteio simples, sem agrupar por rota. Para o
> early stopping o impacto é pequeno, porque ele só decide quando parar e não é a métrica
> reportada, mas vale saber que ali a validação pode conter curvas da mesma rota do treino. O
> **teste** continua separado por rota e limpo.

## SMOTE: só no treino, nunca na avaliação

O desbalanceamento entre Segura e Risco é tratado com SMOTE, que cria amostras sintéticas da
classe minoritária. Ele roda **apenas na porção de treino de cada fold**, nunca na validação nem
no teste.

Isso é garantido pela construção: o SMOTE é um passo interno de um `Pipeline` do imblearn
(`SMOTE -> StandardScaler -> (PCA) -> classificador`), e o `cross_validate` só chama o `fit` do
pipeline na partição de treino de cada fold. A validação é apenas transformada e avaliada.

Duas razões para nunca balancear a avaliação:

- Você estaria **medindo o desempenho em dados inventados** pelo próprio SMOTE, não em curvas
  reais.
- Haveria **vazamento**: um ponto sintético gerado a partir de vizinhos que caíram no treino
  apareceria quase igual na validação, dando ao modelo uma vantagem que ele não teria na prática.

A regra geral é que reamostragem é técnica de **treino**, para o modelo aprender a classe rara.
Teste e validação têm que refletir a distribuição real, desbalanceada, que o modelo vai encontrar
de verdade. Por isso as métricas usam F1 (weighted ou macro) em vez de acurácia pura, que
premiaria um modelo que só acerta a classe majoritária.

Nos modelos de sequência, quando há classificação, o balanceamento é feito por **peso de classe**
na função de perda (`class_weight`, `scale_pos_weight`, entropia cruzada ponderada) em vez de
SMOTE. Sintetizar sequências temporais interpolando vizinhos distorceria a série; reponderar a
perda é mais seguro.

## Resumo

| Papel | O que é | Como é criado | SMOTE? |
|---|---|---|---|
| **Treino** | ~70% das rotas | corte por rota (nível 1) | sim, só aqui |
| **Validação** | fatia do treino | tabular: 5 dobras rotativas; sequência: hold-out fixo | não |
| **Teste** | ~30% das rotas, intocado | corte por rota (nível 1) | não |
