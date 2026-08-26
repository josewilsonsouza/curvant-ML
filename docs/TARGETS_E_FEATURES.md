# Targets e Features

A ideia é prever, antes do carro entrar numa curva, o quão arriscada ela vai ser. O target é avaliado estritamente dentro da curva e a janela de features fica toda antes da entrada, terminando `lead_gap` metros antes.

## CLI

O `cvt` tem dois eixos: a **representação** da entrada (`tab`, as features agregadas por curva;
`seq`, a série temporal bruta) e o **alvo**. O mesmo alvo roda nas duas representações, que é
como se compara se a série bruta traz algo além das features. Sem subcomando, mostra a ajuda.

| Alvo | O que prevê | Tipo | Coluna |
|---|---|---|---|
| `risk` | curva é **Segura ou de Risco** | sim/não | `manobra_combinado_curva` |
| `isl` | a **faixa de ISL** da curva | baixo/medio/alto | `isl_class` |
| `velocity` | a **velocidade crítica** da curva | número (km/h) | `v_critica` |

Então `cvt tab risk`, `cvt seq isl`, `cvt seq velocity`, e assim por diante. Cada célula escreve
em `results/<repr>/<alvo>/`.

Em `cvt seq velocity` o modelo prevê a velocidade **e** a faixa de ISL na mesma passada, por uma
segunda cabeça de classificação. A faixa passa a ser aprendida direto, em vez de sair de um corte
por limiar sobre a velocidade prevista.

Análises complementares do risco (importância de features, Optuna) ficam fora da CLI, nas funções
`importancia_features` e `otimizado` de `curvant/steps/train_risk.py`.

## 2. Targets

Todos são medidos **dentro da curva**, então nunca são usados como features.

### 2.1 Alvos treinados pela CLI

| Coluna | Descrição |
|---|---|
| `manobra_combinado_curva` | A curva foi de **Risco** (1) ou **Segura** (0). |
| `manobra_frenagem_curva` | Frenagem tardia: o motorista freou forte já dentro da curva, sinal de que não a antecipou. |
| `manobra_ziguezague_curva` | Zigue-zague |
| `isl_class` | A faixa de ISL: **baixo, medio ou alto**. |
| `v_critica` | A velocidade (km/h) no ponto de maior risco da curva. |

Os dois critérios individuais de manobra não têm subcomando próprio: eles são treinados de
brinde por `cvt tab risk`, para mostrar qual critério limita o F1 do alvo combinado.

### Como o `isl_class` é formado

O ISL mede o quanto a curva exige da aderência disponível. Em cada ponto da curva:

$$\text{isl}_i = \frac{(v_i/3.6)^2}{\max(|R_i|,\;R_{\min})\cdot g\cdot\mu}$$

O raio $R_i$ vem do B-spline sobre o GPS, e é ruidoso. Daí os dois cuidados:

- **Piso no raio** (`curve_detection.raio_min_isl`, 20 m). Sem ele, um raio espúrio de poucos
  metros faz $v^2/R$ explodir: o rótulo chegava a ISL de 22, fisicamente impossível.
- **A curva é resumida pelo p95** dos seus pontos, não pelo máximo. O máximo se deixa sequestrar
  por um único ponto de ruído; o p95 pega o instante quase pior.

Depois, dois limiares (`physics.isl_baixo` = 0,5 e `physics.isl_alto` = 0,8) dão a faixa. Boa
parte das curvas cai perto de um desses cortes, então eles são o que mais influencia a
dificuldade da classificação.

### 2.2 Targets calculados mas sem subcomando

Estão prontos no `features_df`, mas nenhum subcomando os treina. Servem de material para análise.

| Coluna | O que é |
|---|---|
| `manobra` | Mesmo que `manobra_combinado_curva` (nome antigo, mantido por compatibilidade). |
| `isl_p95` | O ISL robusto da curva, que dá origem ao `isl_class`. |
| `isl_max` / `isl_mean` | ISL máximo e médio da curva. O máximo é sensível a ponto de ruído, por isso não é ele que forma a faixa. |
| `isl_alto` | Sinal de sim/não: o ISL passou de 0,8? |


## 3. As features

Estão organizadas em grupos. Todas vêm da janela pré-curva ou de algo que já se conhece de antemão (como a geometria da estrada à frente, porque a rota é conhecida). Nenhuma é medida dentro da curva.

### Grupo 1 - Sensores OBD (27 colunas)
As estatísticas básicas dos sensores na janela antes da curva. Para `vehicle_speed`, `accel_x` e `accel_y`, calculamos 9 estatísticas cada:

`_mean` (média), `_std` (desvio), `_median` (mediana), `_max`, `_min`, `_slope` (tendência: está acelerando ou freando?), `_cv` (variação relativa), `_mean_tarde` e `_slope_tarde` (os mesmos, mas só na metade final da janela).

> O `engine_rpm` está desligado no `features.yaml` (`extracao.vars_sensor`), então não há colunas de RPM.

> [!NOTE]
> Todas continuam sendo calculadas, mas a whitelist da representação tabular usa só parte delas
> desde jul/2026. As cinco estatísticas de nível da velocidade (`_mean`, `_median`, `_max`,
> `_min`, `_mean_tarde`) tinham correlação acima de 0,95 entre si, então ficou apenas
> `vehicle_speed_min`, que é a mais correlacionada com os alvos. As medidas de dispersão e
> tendência (`_std`, `_cv`, `_slope`, `_slope_tarde`) continuam, porque medem outra coisa.

### Grupo 2 - Dinâmica derivada da aproximação (11 colunas)
- `jerk_x_max`, `jerk_x_std`, `jerk_y_max`, `jerk_y_std`: o jerk é a variação brusca da aceleração. Cada par (max e std do mesmo eixo) tem correlação acima de 0,99, então a whitelist tabular leva só `jerk_x_std` e `jerk_y_max`.
- `n_perigo_accel_precurva`, `n_perigo_lateral_precurva`: quantas vezes a aceleração passou de um limite na janela.
- `precurva_abs_accel_max`: máximo de $\sqrt{(a_x^2+a_y^2)}$ na janela pré-curva
- `distance_car_curve`: o comprimento da janela pré-curva (a distância coberta pela aproximação). ⚠️ **Não é** a distância até a curva, veja a nota abaixo.
- `precurva_bearing_std`: desvio padrão das variações de bearing (Δθ) na janela pré-curva. Correlação de 0,99 com a amplitude abaixo, então fica fora da whitelist tabular.
- `precurva_bearing_range`: amplitude total do bearing na janela (max − min)
- `precurva_n_mudancas_dir`: contagem de alternâncias de sinal de Δbearing (|Δθ| > 15°) na janela

> A variável `distance_car_curve` é o tamanho da janela, em `m` de aproximação. A distância da ponta da janela até a entrada da curva é fixa e vale `lead_gap`  (**30 m** default). Só na representação de sequência existe uma variável que mede a distância que falta para a curva a cada instante: `distancia_restante` (ver Seção 4).

### Grupo 3 - Geometria da janela pré-curva (3 colunas)
`precurva_raio_min`, `precurva_raio_mean`, `precurva_raio_last`: o raio da pista durante a aproximação, pois mede o quanto a estrada já estava curvando antes da curva-alvo.

### Grupo 4 - Geometria da curva à frente (1 coluna)
`f4_raio_min`: o menor raio da curva que vem pela frente, sob a suposição de que a rota é
conhecida de antemão. `f4_raio_mean` e `f4_dnit_num` continuam sendo calculadas mas saíram da
whitelist em ago/2026, por serem a mesma grandeza repetida (correlação de 0,948 e 0,843 com o
mínimo, e desempenho pior nos dois alvos).

Uma ressalva que vale registrar: esse raio não vem de mapa, vem da B-spline ajustada ao GPS da
própria passagem, nos mesmos pontos que geram o rótulo de ISL. Feature e alvo compartilham a
mesma realização de ruído, então parte do acerto nos alvos de ISL e velocidade vem daí. Resolver
isso de verdade exigiria a geometria de um mapa, que é o que a suposição acima já assume.

### Grupo 5 - Contexto das curvas anteriores (4 colunas)
Olham para o histórico do trajeto até aqui, usando só as curvas **anteriores**: `n_curvas_antes`,
`prev_isl_max` (ISL da curva imediatamente anterior), `mean_isl_antes` (ISL médio de todas as
curvas anteriores) e `prev_raio_min` (o menor raio da curva imediatamente anterior).
`prev_raio_mean` e `prev_dnit_num` saíram da whitelist na mesma poda do grupo 4.

### O grupo de Monte Carlo foi removido
Havia um sexto grupo, `mc_p_baixo` / `mc_p_medio` / `mc_p_alto`, que simulava a velocidade de
entrada e devolvia a probabilidade de cada classe de ISL. Ele saiu em ago/2026 junto com o
módulo que o produzia. O motivo é que a simulação usava o raio cru enquanto o rótulo de ISL
aplica um piso de 20 m, então os dois calculavam a mesma física com regras diferentes. Nas 523
curvas com raio abaixo de 20 m, o `mc_p_alto` médio era 0,995 contra 31,7% de curvas realmente
rotuladas como alto, e a concordância com `isl_class` caía de 56,1% para 31,9%.

## 4. A representação de sequência usa uma entrada diferente

Os grupos acima são as features da representação **tabular** (`cvt tab`), e é a lista
`flags.tab.features` do [features.yaml](../features.yaml) que decide quais entram. Os três alvos
tabulares compartilham essa mesma lista: as features pré-curva são as mesmas, o que muda entre
eles é só o que se prevê.

A representação de **sequência** (`cvt seq`) **não** usa essas features. Ela trabalha com a série
no tempo dos sensores, ou seja, a aproximação inteira, não só as estatísticas resumidas:

- **Sensores ao longo do tempo** (50 instantes): `vehicle_speed` e `engine_rpm` + o canal **`distancia_restante`**, que é a distância que falta para a entrada da curva em cada instante (decresce até zero na entrada, normalizada para [0, 1]).
- **Alguns números fixos por curva**, repetidos ao longo do tempo como canais constantes: o raio na janela (`precurva_raio_min`, `_mean`, `_last`), o raio da curva à frente (`f4_raio_min`) e o contexto (`mean_isl_antes`, `prev_isl_max`, `n_curvas_antes`).
- **A resposta da curva anterior** (`prev_<alvo>`): usa o valor do alvo na curva passada como pista.

Para mudar os canais da sequência, edite `flags.seq.sensors` e `flags.seq.scalares_extras` no [features.yaml](../features.yaml).

> [!TIP]
> - **A previsão é antecipada.** A janela pré-curva termina 30 m **antes** da entrada
  (`features.yaml` > `extracao.lead_gap`). Nada medido na entrada ou dentro da curva pode ser feature.
> - **O contexto das curvas anteriores é atualizado de forma sequencial:** cada curva percorrida alimenta o histórico
da próxima. Neste dataset isso é reconstruído da mesma forma.