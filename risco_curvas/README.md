# Risco em curvas com OBD-II e relógio inteligente

Predição antecipada do nível de risco (baixo, médio, alto) de uma curva a partir do que o veículo e o motorista fazem nos metros anteriores a ela. Cada viagem foi registrada por dois meios ao mesmo tempo: um adaptador OBD-II, com dados do veículo e GPS do celular, e o aplicativo Sensor Logger, com o GPS do celular e os sensores de um relógio pareado. A análise compara três cenários de dispositivo (OBD, relógio e os dois juntos), quatro horizontes de antecedência (1; 1,5; 2 e 3 s) e quatro modelos (RF, XGB, LGB e LSTM), sempre contra duas referências: a classe majoritária e uma regra física baseada em `v²/R`.

## Organização

```
config.yaml          parâmetros de todas as etapas
risco/
  fontes/            leitura dos registros OBD e das gravações do Sensor Logger
  pareamento.py      junção das duas fontes por viagem, validada pelo GPS
  limpeza.py         valores fora da faixa física, saltos e picos isolados
  trajetoria.py      segmentação nas lacunas de gravação e detecção de curvas
  amostras.py        janelas de observação e de risco, cálculo de ae
  rotulos.py         níveis de risco por KMeans em sqrt(ae)
  variaveis.py       variáveis por cenário e matrizes de entrada
  modelos.py         classificadores e referências
  avaliacao.py       validação cruzada agrupada por viagem
  relatorio.py       tabelas e gráficos consolidados
dados/
  brutos/<conjunto>/obd/              registros OBD (CSV)
  brutos/<conjunto>/sensor_logger/    gravações do celular e do relógio (ZIP ou JSON)
  original/          bases processadas recebidas, usadas como estão
  corrigida/         bases reconstruídas a partir dos brutos por este projeto
resultados/<base>/   métricas, predições e gráficos de cada base
```

As bases `original` e `corrigida` passam pela mesma geração de amostras, rotulagem e avaliação. A diferença entre os resultados das duas mede, portanto, o efeito da reconstrução dos dados, com o método fixo.

## Execução

```bash
cd risco_curvas
python -m risco tudo                                 # todas as etapas, as duas bases
python -m risco base                                 # só a reconstrução da base corrigida
python -m risco amostras --base corrigida --horizontes 1 2
python -m risco avaliar --base original --modelos RF XGB LGB
python -m risco relatorio
```

Dependências: pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, tensorflow, matplotlib e pyyaml.

## Construção da base corrigida

**Pareamento.** Uma gravação do Sensor Logger só é ligada a uma viagem OBD quando, com o veículo em movimento, a mediana da distância entre a posição do GPS do celular e a registrada pelo OBD fica abaixo de 10 m. Gravações que se sobrepõem no tempo mas pertencem a outro veículo ficam assim de fora. Os sensores são lidos no instante de cada leitura OBD: o GPS por interpolação linear entre leituras próximas, e os sensores de alta frequência pela média numa janela de 1 s centrada no instante. Os ângulos são combinados pelo seno e cosseno. O arquivo `dados/corrigida/pareamento.csv` registra, para cada viagem, as gravações candidatas, a distância de cada uma e o motivo de exclusão, quando houver.

**Limpeza.** Valores fora da faixa física de cada grandeza viram ausentes, assim como saltos instantâneos de velocidade e de posição. Os sensores físicos (acelerômetros, giroscópios, orientação e velocidade do GPS) passam pelo filtro de Hampel. Sinais comandados pelo motorista, como pedal e acelerador, não passam pelo filtro, porque neles um pulso curto é uma ação real. A contagem por regra e variável fica em `dados/corrigida/limpeza.csv`.

**Trajetória.** A viagem é dividida em segmentos sempre que duas leituras ficam a mais de 2 s uma da outra, e nenhuma grandeza é calculada através dessas lacunas. A detecção de curvas segue o critério geométrico da base original: curvatura das posições projetadas, suavização gaussiana, limiar no percentil 30 da viagem e trechos com sentido constante.

## Amostras e rótulos

Cada amostra liga uma janela de risco de 10 m dentro de uma curva a uma janela de observação de 20 m que termina o horizonte escolhido antes do risco. Só entram as curvas acima do percentil 30 de curvatura da viagem. Na grade de 2 m, um valor só é interpolado entre leituras reais a até 2 s uma da outra, e a amostra precisa ter leitura de todas as variáveis na observação. O risco é a excedência média de aceleração lateral `ae`, com limiar `(inclinação + μ)·g`, μ = 0,15. Janelas com aceleração lateral acima de 1 g são descartadas por não serem fisicamente alcançáveis. As contagens de descarte por motivo ficam ao lado de cada arquivo de amostras.

Os níveis vêm de um KMeans com k = 3 em `sqrt(ae)`. Em cada dobra, os cortes são ajustados só com o treino e aplicados ao teste.

## Variáveis

As listas por cenário partem da seleção validada em `dados/original/variaveis_por_cenario.csv`. Saem dessa seleção, conforme `config.yaml`:

- **cópias exatas de outra coluna**: velocidade, RPM e aceleração com dois nomes, e consumo instantâneo, que é proporcional ao CO₂;
- **sinais praticamente constantes**: os giroscópios do OBD e a relação de equivalência ar/combustível;
- **variáveis que identificam o local ou o momento da viagem**: altitude, pressão, nível de combustível, magnetômetro e rumo. Com elas, o modelo pode reconhecer o trecho de estrada em vez de aprender o comportamento que antecede o risco.

Os modelos tabulares recebem média, desvio, mínimo e máximo de cada variável na observação, mais o raio médio da curva à frente. A LSTM recebe a sequência, com o raio como canal constante.

## Avaliação

A validação cruzada usa 5 dobras agrupadas por viagem: o teste só contém viagens ausentes do treino. A LSTM usa 20% das viagens de treino para a parada antecipada.

A referência física é uma árvore de profundidade 3 sobre a aceleração lateral prevista `v²/R`, a velocidade no fim da observação e o raio da curva. Ela usa apenas informação disponível no momento da predição, a mesma a que os modelos têm acesso. O resultado de interesse é o ganho de cada modelo sobre essa referência.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados/<base>/<horizonte>/metricas_por_dobra.csv` | métricas de cada modelo, cenário e dobra |
| `resultados/<base>/<horizonte>/predicoes.parquet` | classe real e prevista por amostra |
| `resultados/<base>/<horizonte>/cortes_por_dobra.csv` | cortes de `ae` e contagem de classes no treino |
| `resultados/<base>/resumo.csv` | média e desvio entre dobras e ganho sobre a referência física |
| `resultados/<base>/amostras.csv` | amostras, viagens e classes por horizonte |
| `resultados/<base>/f1_alto_por_horizonte.png` | F1 da classe alto por horizonte e cenário |
| `resultados/comparacao_bases.csv` | as duas bases lado a lado |
