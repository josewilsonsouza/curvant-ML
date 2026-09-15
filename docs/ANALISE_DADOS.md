# Análise dos dados

Medições sobre o dataset `eletro_rjdf_serra_rjmgba_janeiro_agda_denso`, com 439.512 pontos, 109 rotas, 2.593 curvas e cerca de 101 horas de gravação. Elas separam o que é limitação dos dados do que seria limitação de modelo, e sustentam as escolhas de alvos e features do projeto. Figuras em `results/analise_dados/`.

O resumo é que os modelos vão bem naquilo que o GPS sustenta e mal naquilo que dependeria do acelerômetro, e há uma razão medível para isso.

## O achado principal: o acelerômetro não mede a dinâmica do veículo

Dentro de curvas, a aceleração lateral medida pelo sensor não tem relação com a aceleração centrípeta calculada pela física. Restringindo a situações onde o cálculo é confiável (raio entre 20 e 150 metros, velocidade acima de 30 km/h, 62.080 pontos), a correlação entre `|accel_y|` e `v²/R` é 0,055. Praticamente zero.

Três verificações independentes confirmam que isso não é um detalhe de calibração:

**O sinal não muda com o lado da curva.** Se o eixo y apontasse para a lateral do veículo, curvas para a esquerda e para a direita dariam leituras de sinais opostos. A diferença média entre os dois casos é de 0,004 m/s², ou seja, nenhuma.

**A magnitude não fecha.** A aceleração centrípeta mediana em curva é 2,2 m/s², enquanto o `|accel_y|` mediano é 0,32 m/s², sete vezes menor. Agrupando por faixa de aceleração real, o sensor sai de 0,42 para apenas 0,53 m/s² enquanto a física vai de 0,9 para 7,5.

**O eixo longitudinal também não bate.** Comparando `accel_x` com a aceleração obtida da variação de velocidade (dv/dt), em 388 mil pontos, a correlação é 0,001. Nenhuma das 86 rotas com dados suficientes tem qualquer eixo passando de 0,2.

Tentei recuperar o sinal por reorientação: para cada rota, a melhor combinação linear possível dos três eixos explica 0,3% da variação da aceleração centrípeta. Não é um problema de eixos trocados que uma rotação resolveria. O sensor está registrando vibração, não o movimento do corpo do veículo, o que é coerente com coleta por celular em posição livre.

Os picos intra-segundo do dataset `_denso` melhoram a situação, mas pouco: a correlação sobe de 0,055 para 0,181. Isso indica que parte do problema é a amostragem a 1 Hz, mas a maior parte é a orientação do sensor.

### Verificações que descartam erro de análise

Antes de aceitar essa conclusão, testei as explicações alternativas mais prováveis. Todas foram descartadas.

**As unidades estão corretas.** As colunas de gravidade do próprio aparelho têm norma mediana de 9,735 m/s², praticamente o valor de g, o que confirma que o acelerômetro está em m/s² e não em múltiplos de g. As colunas de origem já vinham rotuladas assim, e a velocidade em km/h. A comparação com `v²/R` está na mesma unidade dos dois lados.

**Não é a quantização da velocidade.** A velocidade do OBD anda em passos de 1 km/h, o que poderia estragar o cálculo de dv/dt. Refazendo com a velocidade contínua do GPS, a correlação com `accel_x` continua em 0,002.

**Não é defasagem de tempo.** Testei correlação cruzada deslocando o acelerômetro de 8 segundos para trás até 8 segundos para frente, em três rotas longas. Se houvesse erro de sincronismo, apareceria um pico em algum deslocamento. Não aparece: a correlação fica no mesmo patamar baixo em toda a faixa.

**Não é dado congelado.** Apenas 0,4% das leituras do acelerômetro repetem a anterior, contra 32,8% da velocidade do OBD (esperado, pela quantização). O sensor está variando de verdade.

**O método de comparação é válido, e isso é o argumento decisivo.** Como controle positivo, comparei a aceleração longitudinal obtida do GPS com a obtida do OBD, dois sensores independentes. Elas concordam com correlação de 0,712. Ou seja, a grandeza existe, é mensurável e o cálculo está certo. O acelerômetro é o único que não acompanha, com 0,002.

**O sensor mede algo real, só não é o movimento do veículo.** O desvio padrão de `accel_x` sobe de 0,52 m/s² com o carro parado para 1,12 m/s² acima de 80 km/h. O sensor responde à velocidade, o que é a assinatura de vibração da estrada e do motor, não da aceleração do corpo do veículo.

## O raio da spline produz geometria impossível

A detecção de curvas gera raios que não correspondem a estradas reais. Entre os pontos marcados como curva, 28% têm raio abaixo de 20 metros e 12,6% abaixo de 5 metros. Nenhuma rodovia tomada em velocidade tem raio de poucos metros, então isso é ruído de GPS amplificado pela derivada segunda da spline.

A consequência aparece na aceleração centrípeta: 9,2% dos pontos em curva excedem 5,9 m/s², que é o limite de aderência do asfalto seco, e 5,3% passam de 8 m/s². São valores fisicamente impossíveis, no mesmo cálculo que alimenta o ISL e a `v_critica`. No nível da curva inteira, 18,9% dos segmentos têm ao menos um ponto assim.

É por isso que existe o piso de raio (`curve_detection.raio_min` e `raio_min_isl`) antes de qualquer divisão por R.

## Quanto do acerto em v_critica vem do modelo

Regressões lineares sobre a mesma informação, para comparar com o que os modelos entregam:

| Entrada | R² |
|---|---|
| Só a velocidade da janela | 0,574 |
| Velocidade e raio mínimo da curva | 0,668 |
| Mais geometria e distância | 0,682 |

O aprendizado acrescenta um ganho real, mas modesto, sobre uma regressão de duas variáveis. É o argumento do teto de informação: a velocidade com que o motorista chega e a geometria da curva explicam quase tudo o que é explicável, e as duas representações convergem para o mesmo patamar.

## Qualidade geral dos dados

Os sensores universais não têm valores faltantes, e a taxa de amostragem é limpa: mediana de 1,00 s, com apenas 0,05% dos intervalos acima de 1,5 s.

A velocidade do OBD é quantizada em passos de 1 km/h e concorda com a do GPS com erro médio de 1,73 km/h. Isso dá uma medida direta do piso de incerteza do alvo: o MAE dos modelos fica em torno de três vezes o ruído da própria medida, o que delimita o quanto ainda há para ganhar.

## O que fica em aberto

Um único caminho pode levantar o teto de informação em vez de apenas limpar ruído: dar ao modelo um perfil de curvatura amostrado ao longo do trecho à frente, no lugar dos escalares de geometria que ele recebe hoje.
