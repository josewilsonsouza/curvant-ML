# Correção tardia

O rótulo de comportamento do motorista. Roda depois da detecção de curvas e antes da extração de features. Marca os pontos de cada segmento de curva, e na extração isso vira o alvo por curva: positiva se algum ponto foi marcado.

## O critério

Frear antes da curva é condução prudente. Frear já dentro dela indica que o motorista não antecipou o que vinha e precisou corrigir. A desaceleração sai da variação da velocidade entre pontos consecutivos do próprio segmento:

$$a(t_i) = \frac{v(t_i) - v(t_{i-1})}{\Delta t_i}$$

O critério dispara quando a maior desaceleração do segmento passa do limiar de `config.yaml > correcao_tardia.limiar_desaceleracao`, o que marca cerca de um quarto das curvas. Leituras absurdamente altas são descartadas como falha do sensor.

A velocidade é o sensor confiável do conjunto. A aceleração longitudinal calculada a partir dela pelo GPS e pelo OBD, dois caminhos independentes, concorda com correlação de 0,71. O acelerômetro fica em 0,002 no mesmo teste.

### A faixa morta em torno do limiar

A velocidade do OBD chega em passos de 1 km/h, uma vez por segundo. Cada degrau desses vale cerca de 0,28 m/s² na conta, o que perto do limiar é grande o bastante para decidir o rótulo pelo arredondamento em vez do comportamento. Por isso `margem_histerese` marca como indefinidas as curvas que caem nessa faixa, e elas saem do treino.

## Por que não comparar com a velocidade segura

O critério mais natural seria comparar a velocidade de entrada com a velocidade segura da curva. Ele não é usado de propósito: uma regressão logística com a velocidade de aproximação e o raio reproduz esse rótulo quase perfeitamente, então prever uma reescrita das próprias features não mediria aprendizado nenhum. Essa dimensão fica com os alvos `isl_p95` e `v_critica`.

Vale saber que este alvo e o de ISL medem coisas opostas, e isso não é acidente. Frear dentro da curva reduz a velocidade, e o ISL é v²/R, então corrigir derruba justamente a grandeza que o ISL acompanha. A taxa de correção tardia cai de 0,338 nas curvas de ISL baixo para 0,124 nas de ISL alto. Um alvo descreve o erro que foi compensado, o outro o que não foi, e juntá-los num rótulo só produz uma mistura sem significado.

## O que este alvo acrescenta

É a única medida do projeto que fala mais do condutor do que da estrada. Apenas 12% da variação do rótulo vem da rota, contra 39% da `v_critica` e 41% do ISL. Os alvos contínuos descrevem em boa parte a geometria do trecho; este descreve uma decisão de quem dirige.
