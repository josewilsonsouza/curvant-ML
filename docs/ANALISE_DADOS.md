# Análise dos dados

Levantamento feito em 21/07/2026 sobre o dataset `eletro_rjdf_serra_rjmgba_janeiro_agda_denso`, usando os caches do pipeline (439.512 pontos, 109 rotas, 2.593 curvas, cerca de 101 horas de gravação). O objetivo era entender por que o desempenho estagnou e separar o que é limitação de modelo do que é limitação dos dados. Figuras em `results/analise_dados/`.

O resumo é que os modelos vão bem naquilo que o GPS sustenta e mal naquilo que depende do acelerômetro, e há uma razão medível para isso.

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

## Por que isso explica os resultados que temos

Dois dos três critérios de risco (Kamm e lateral) são calculados a partir do acelerômetro. O terceiro (zigue-zague) vem do GPS, via mudanças de rumo e raio. Separando as curvas rotuladas como risco:

| Origem do rótulo | Curvas | Fração do risco |
|---|---|---|
| Só acelerômetro (Kamm ou lateral) | 570 | 52% |
| Só GPS (zigue-zague) | 258 | 24% |
| Ambos | 259 | 24% |

Pouco mais da metade das curvas de risco é definida exclusivamente por um sensor que não tem relação medível com a dinâmica do veículo. Isso casa exatamente com o desempenho por critério já registrado em `results/tab/risk/`: o zigue-zague, que é o critério de GPS, chega a F1 0,561, enquanto Kamm fica em 0,384 e o lateral em 0,494. E casa com o contraste maior do projeto, em que os alvos derivados de GPS e velocidade (`v_critica` com R² 0,75, `isl_class` com F1 0,71) vão muito melhor do que o risco binário.

Não é que o modelo não aprende o risco. É que metade do rótulo de risco é ruído com aparência de rótulo, e nenhum modelo aprende a prever ruído.

Há um sinal adicional na mesma direção: a taxa de curvas de risco varia de 4% a 89% conforme a gravação, e a variação entre rotas responde por 24% da variação total do rótulo. Parte disso é real, porque rotas de serra são mais exigentes que rotas de planalto, mas com o sensor desalinhado boa parte vem de como o aparelho estava posicionado em cada coleta.

## Segundo problema: o raio da spline produz geometria impossível

A detecção de curvas gera raios que não correspondem a estradas reais. Entre os pontos marcados como curva, 28% têm raio abaixo de 20 metros e 12,6% abaixo de 5 metros. Nenhuma rodovia tomada em velocidade tem raio de poucos metros, então isso é ruído de GPS amplificado pela derivada segunda da spline.

A consequência aparece na aceleração centrípeta calculada: 9,2% dos pontos em curva excedem 5,9 m/s², que é o limite de aderência do asfalto seco, e 5,3% passam de 8 m/s². São valores fisicamente impossíveis, e é o mesmo cálculo que alimenta o critério de zigue-zague, o ISL e a `v_critica`. No nível da curva inteira, 18,9% dos segmentos têm ao menos um ponto assim, e 14,5% das curvas têm ISL acima de 1,5.

O piso de 20 metros que já existe (`raio_min_isl`) protege o rótulo de ISL, mas o `ctp_accel` usado no risco continua com piso de 5 metros e carrega esse ruído.

## Terceiro ponto: quanto do acerto em v_critica é o modelo

Vale saber o quanto o aprendizado acrescenta sobre contas simples. Regressões lineares sobre a mesma informação dão:

| Entrada | R² |
|---|---|
| Só a velocidade da janela | 0,574 |
| Velocidade e raio mínimo da curva | 0,668 |
| Mais geometria e distância | 0,682 |
| XGBoost tabular (whitelist completa) | 0,764 |
| LSTM sobre a sequência | 0,730 |

O modelo entrega um ganho real, mas modesto, sobre uma regressão de duas variáveis. Isso reforça o argumento do teto de informação que já discutimos: a velocidade com que o motorista chega e a geometria da curva explicam quase tudo o que é explicável, e as duas representações convergem para o mesmo patamar.

## Outros pontos menores

Os sensores universais não têm valores faltantes, e a taxa de amostragem é limpa (mediana de 1,00 s, apenas 0,05% dos intervalos acima de 1,5 s). A velocidade do OBD é quantizada em passos de 1 km/h e concorda bem com a do GPS (erro médio de 1,73 km/h), o que dá uma medida direta do piso de incerteza: o MAE de 6,3 km/h dos modelos está a cerca de três vezes o ruído da própria medida do alvo.

A whitelist tabular tinha redundância alta, já corrigida. As cinco estatísticas de nível da velocidade tinham correlação acima de 0,95 entre si, e o mesmo valia para os dois pares de jerk e para as duas medidas de bearing. Ficou um representante de cada grupo, escolhido pela correlação com os alvos, o que reduziu a lista de 51 para 44 features. A verificação nos três alvos tabulares mostrou variação entre -0,006 e +0,008, ou seja, dentro do ruído: a redundância não estava ajudando, só atrapalhava a leitura de importância e os modelos lineares.

As features de jerk têm caudas muito pesadas (assimetria acima de 12), o que era esperado, já que são derivadas do acelerômetro que agora sabemos ser vibração.

## O que saiu daqui

Tudo o que esta análise apontou foi executado até agosto de 2026, com uma exceção e uma surpresa.

O alvo de risco deixou de ser o principal e depois foi removido por inteiro, o piso do raio subiu, a whitelist perdeu a redundância e todas as features de acelerômetro, e a simulação de Monte Carlo saiu junto. A surpresa foi o item que propunha testar um alvo de risco feito só com GPS: ele foi testado e refutou a expectativa. Os dois critérios de GPS que restavam apontam para lados opostos da física, então o rótulo combinado foi descartado em vez de melhorado. O raciocínio está em [CORRECAO_TARDIA.md](CORRECAO_TARDIA.md).

Segue aberto um único item, e é o único que pode levantar o teto de informação em vez de só limpar ruído: dar ao modelo um perfil de curvatura amostrado ao longo do trecho à frente, no lugar dos escalares de geometria que ele recebe hoje.
