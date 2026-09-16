# Correção tardia

O rótulo de comportamento do motorista. Roda depois da detecção de curvas e antes da extração de features. Marca os pontos de cada segmento de curva, e na extração isso vira o alvo por curva: positiva se algum ponto foi marcado.

## O critério

Frear antes da curva é condução prudente. Frear já dentro dela indica que o motorista não antecipou o que vinha e precisou corrigir. A desaceleração sai da variação da velocidade entre pontos consecutivos do próprio segmento:

$$a(t_i) = \frac{v(t_i) - v(t_{i-1})}{\Delta t_i}$$

O critério dispara quando a maior desaceleração do segmento passa do limiar de `config.yaml > correcao_tardia.limiar_desaceleracao`, o que marca cerca de um quarto das curvas. Leituras absurdamente altas são descartadas como falha do sensor.

A velocidade do OBD vem em passos de 1 km/h, uma vez por segundo, e um passo desses já significa 0,28 m/s² de desaceleração. Uma curva que freou perto do limiar pode então cair de um lado ou do outro só pelo arredondamento da medida, não pelo que o motorista fez. `margem_histerese` define uma faixa em torno do limiar: a curva que cai nela é marcada como indefinida e sai do treino.