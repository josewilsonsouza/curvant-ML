# Correção tardia

O rótulo de comportamento do motorista. Roda depois da detecção de curvas e antes da extração de features. Marca os pontos de cada segmento de curva, e na extração isso vira o alvo por curva: positiva se algum ponto foi marcado.

## O critério

Frear antes da curva é condução prudente. Frear já dentro dela indica que o motorista não antecipou o que vinha e precisou corrigir. A desaceleração sai da variação da velocidade entre pontos consecutivos do próprio segmento:

$$a(t_i) = \frac{v(t_i) - v(t_{i-1})}{\Delta t_i}$$

O critério dispara quando a maior desaceleração do segmento passa do limiar de `config.yaml > correcao_tardia.limiar_desaceleracao`, o que marca cerca de um quarto das curvas. Leituras absurdamente altas são descartadas como falha do sensor.

A velocidade é o sensor confiável do conjunto. A aceleração longitudinal calculada a partir dela pelo GPS e pelo OBD, dois caminhos independentes, concorda com correlação de 0,71. O acelerômetro fica em 0,002 no mesmo teste.

### A faixa morta em torno do limiar

A velocidade do OBD chega em passos de 1 km/h, uma vez por segundo. Cada degrau desses vale cerca de 0,28 m/s² na conta, o que perto do limiar é grande o bastante para decidir o rótulo pelo arredondamento em vez do comportamento. Por isso `margem_histerese` marca como indefinidas as curvas que caem nessa faixa, e elas saem do treino.

## Por que sobrou só este critério

**Kamm e aceleração lateral saíram em julho de 2026.** Vinham do acelerômetro, que é do celular usado na coleta e não da central do veículo. A correlação entre a leitura lateral e a aceleração centrípeta calculada pela física é de 0,055, não há inversão de sinal entre curvas para lados opostos, e a melhor reorientação linear possível recupera 0,3% do sinal. Detalhe em [ANALISE_DADOS.md](ANALISE_DADOS.md).

**O zigue-zague e o rótulo combinado de risco saíram em agosto.** Aqui o motivo é outro e mais interessante. Medindo a taxa de cada critério dentro de cada faixa de ISL:

| Critério | ISL baixo | ISL alto | correlação com `isl_p95` |
|---|---|---|---|
| correção tardia | 0,338 | 0,124 | -0,197 |
| zigue-zague | 0,047 | 0,446 | +0,418 |

Os dois apontam para lados opostos, e a correlação entre eles é de -0,092, com 7% de sobreposição nas curvas marcadas. O mecanismo é direto: frear dentro da curva reduz a velocidade, e o ISL é v²/R, então a frenagem derruba justamente a grandeza que o outro critério acompanha. O rótulo combinado era o OU dos dois, ou seja, misturava o erro que foi compensado com o que não foi. A F1 aparentemente boa que ele alcançava vinha da taxa de positivos mais alta, não de aprendizado melhor.

Isso não é defeito de implementação. Qualquer critério de "corrigiu depois de entrar" vai ser anti-correlacionado com qualquer critério de "exigiu demais da aderência", então uma proposta futura de rótulo composto de risco encontra o mesmo problema.

**A velocidade de entrada nunca entrou.** Comparar a velocidade de entrada com a velocidade segura da curva seria o critério mais natural, mas uma regressão logística com a velocidade de aproximação e o raio reproduz esse rótulo quase perfeitamente. Prever uma reescrita das próprias features não mede aprendizado nenhum. Essa dimensão fica com os alvos `isl_p95` e `v_critica`.

## O que este alvo acrescenta

É a única medida do projeto que fala mais do condutor do que da estrada. Apenas 12% da variação do rótulo vem da rota, contra 39% da `v_critica` e 41% do ISL. Os alvos contínuos descrevem em boa parte a geometria do trecho; este descreve uma decisão de quem dirige.
