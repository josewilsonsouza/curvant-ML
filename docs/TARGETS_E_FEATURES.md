# Alvos e features

Prevemos o que vai acontecer dentro de uma curva usando apenas o que foi medido antes de entrar nela. O alvo é sempre medido dentro da curva; a janela de features termina `lead_gap` metros antes da entrada. Nada que o carro viu na curva pode virar feature.

## Os três alvos

| Subcomando | Coluna | O que responde | Tipo |
|---|---|---|---|
| `velocity` | `v_critica` | com que velocidade o motorista vai chegar no ponto mais exigente | km/h |
| `isl` | `isl_p95` | quanto isso vai consumir da aderência do pneu | número |
| `correcao` | `correcao_tardia_curva` | se ele vai precisar corrigir depois de já estar na curva | sim ou não |

Cada alvo roda nas duas representações (`cvt tab <alvo>` e `cvt seq <alvo>`), que é como se compara se a série bruta traz algo além das features resumidas. As saídas vão para `results/<repr>/<alvo>/`.

Em `cvt seq velocity` o modelo prevê a velocidade e a faixa de ISL na mesma passada, por uma segunda cabeça de classificação que usa a faixa medida como gabarito.

### Como o ISL é calculado

O ISL diz quanto da aderência disponível a curva consome. Em cada ponto:

$$\text{isl} = \frac{(v/3.6)^2}{\max(|R|,\,R_{\min}) \cdot g \cdot \mu}$$

O raio vem da B-spline ajustada ao GPS, e é ruidoso. Dois cuidados lidam com isso. O raio recebe um piso (`curve_detection.raio_min_isl`) antes de entrar na divisão, senão um raio espúrio de poucos metros faz o valor explodir. E a curva é resumida pelo percentil 95 dos seus pontos, não pelo máximo, que se deixa levar por um único ponto de ruído.

O rótulo resultante é estável: mexer no piso do raio ou somar à velocidade o ruído de medida conhecido troca a faixa de poucos por cento das curvas.

### Por que o ISL é regressão e não classificação

As faixas baixo, médio e alto vêm de dois cortes fixos sobre uma grandeza contínua, e cerca de um quarto das curvas cai bem em cima de um dos cortes. Classificar direto trataria um erro minúsculo na fronteira igual a um erro grosseiro, e o treino não receberia nenhuma informação sobre a distância até o limiar.

Regredir o `isl_p95` e cortar depois preserva essa informação e entrega a faixa do mesmo jeito. O acerto da faixa fica praticamente igual ao de um classificador treinado direto nela, então o ganho não é de acerto: é ter um erro contínuo interpretável em vez de só uma taxa de acerto de três caixas.

### Colunas calculadas sem subcomando

`isl_class` é a faixa medida, usada como gabarito da cabeça auxiliar em `cvt seq velocity`. `isl_max` alimenta as features de contexto. `correcao_indefinida_curva` marca as curvas na faixa morta do limiar de desaceleração, para descartá-las do treino.

## As features

Todas saem da janela de aproximação ou da geometria da rota, que é conhecida de antemão. A lista completa e ativa está em [features.yaml](../features.yaml); esta seção explica o que cada grupo significa.

**Velocidade na aproximação.** O nível com que o motorista chega (`vehicle_speed_min`), o quanto a velocidade oscilou (`vehicle_speed_cv`) e para onde ela estava indo (`vehicle_speed_slope`, e o mesmo só na metade final da janela em `vehicle_speed_slope_tarde`). O produtor calcula nove estatísticas por sensor, e a whitelist leva um representante de cada grupo correlacionado.

**Oscilação de direção.** `precurva_bearing_range` mede o quanto o rumo variou na aproximação, e `precurva_n_mudancas_dir` conta quantas vezes o motorista trocou de lado.

**O quanto o trecho antes já era sinuoso.** `precurva_raio_min`, `precurva_raio_mean` e `precurva_raio_last`, sendo o último o ponto mais perto da entrada.

**Geometria da curva que vem pela frente.** `curva_raio_min` e `curva_raio_mean`. Vale uma ressalva: esse raio não vem de mapa, sai da mesma B-spline que gera o rótulo de ISL, sobre os mesmos pontos. Feature e alvo compartilham o ruído do GPS, então parte do acerto nos alvos contínuos vem daí. Resolver isso exigiria geometria de mapa, que é justamente o que a suposição de rota conhecida já assume.

**O que o motorista já enfrentou na rota.** `n_curvas_antes`, `prev_isl_max`, `mean_isl_antes` e `prev_raio_min`. Todas usam apenas curvas já percorridas.

O acelerômetro não entra em nenhuma das listas. Ele é do celular usado na coleta, não da central do veículo, e o que registra é vibração da estrada e do motor. A medição está em [ANALISE_DADOS.md](ANALISE_DADOS.md).

## A sequência recebe outra entrada

A representação tabular usa a lista `flags.tab.features`. A de sequência não usa essa lista: ela trabalha com a série no tempo, e por isso tem duas entradas separadas em `flags.seq`.

Os canais que variam no tempo são `vehicle_speed` e `engine_rpm`, mais `distancia_restante`, que é o quanto falta para a entrada da curva em cada instante. Os escalares da curva entram repetidos em todos os instantes, como canais constantes, e incluem o alvo da curva anterior, que a própria rotina acrescenta.

## Como adicionar uma feature

Crie a coluna em [features.py](../curvant/driving/features.py) e liste o nome na flag que deve usá-la. A seleção é opt-in: o que não estiver listado é produzido e ignorado. `checar_leakage` recusa a execução se o alvo aparecer na lista de features.
