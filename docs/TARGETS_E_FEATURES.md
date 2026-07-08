# Targets e Features

A ideia é prever, antes do carro entrar numa curva, o quão arriscada ela vai ser. O target é avaliado estritamente dentro da curva e a janela de features fica toda antes da entrada, terminando `lead_gap` metros antes.

## CLI

Cada flag do `cvt` treina um alvo diferente (sem flag, mostra a ajuda):

| Comando | O que prevê | Tipo |
|---|---|---|
| `--risk` | curva é **Segura ou de Risco** (`manobra_combinado_curva`) | sim/não |
| `--velocity` | a **velocidade crítica** da curva (`v_critica`) | número |

Análises complementares do risco (importância de features, Optuna) ficam em `notebooks/analise_risco.ipynb`.

## 2. Targets

Todos são medidos **dentro da curva**, então nunca são usados como features.

### 2.1 Alvos realmente treinados por algum comando

| Coluna | Descrição |
|---|---|
| `manobra_combinado_curva` | A curva foi de **Risco** (1) ou **Segura** (0). |
| `manobra_accel_curva` | Critério do limite de aderência do pneu (círculo de Kamm): a aceleração total passou de uma fração do que o pneu aguenta. |
| `manobra_lateral_curva` | Aceleração lateral alta numa curva fechada o bastante. |
| `manobra_ziguezague_curva` | Zigue-zague |
| `isl_class` | A classe de risco ISL: **baixo, médio ou alto**. |
| `isl_max` | O **maior ISL** atingido na curva. $ISL = v^2 / (R·g·\mu)$: quanto a curva chegou perto do limite de derrapagem. |
| `v_critica` | A velocidade (km/h) no ponto de maior risco da curva. Este é o alvo do `--velocity`. |

### 2.2 Targets Opcionais

Estão calculados e prontos, mas nenhum comando os usa por enquanto. Para treinar um deles,
basta apontá-lo como alvo (ex.: `temporais.target` no `config.yaml`).

| Coluna | O que é |
|---|---|
| `manobra` | Mesmo que `manobra_combinado_curva` (nome antigo, mantido por compatibilidade). |
| `isl_mean` | ISL **médio** da curva (em vez do máximo). |
| `isl_alto` | Sinal de sim/não: o ISL passou de 0,8? |
| `isl_sensor_max` / `_mean` / `_class` | Uma versão do ISL calculada pelo **acelerômetro** (\|accel_y\|/g·μ) em vez do raio do GPS. Não depende do raio, então não sofre com ruído de GPS. |


## 3. As features

Estão organizadas em grupos. Todas vêm da janela pré-curva ou de algo que já se conhece de antemão (como a geometria da estrada à frente, porque a rota é conhecida). Nenhuma é medida dentro da curva.

### Grupo 1 - Sensores OBD (27 colunas)
As estatísticas básicas dos sensores na janela antes da curva. Para `vehicle_speed`, `accel_x` e `accel_y`, calculamos 9 estatísticas cada:

`_mean` (média), `_std` (desvio), `_median` (mediana), `_max`, `_min`, `_slope` (tendência: está acelerando ou freando?), `_cv` (variação relativa), `_mean_tarde` e `_slope_tarde` (os mesmos, mas só na metade final da janela).

> O `engine_rpm` está desligado no `features.yaml` (`extracao.vars_sensor`), então não há colunas de RPM.

### Grupo 2 - Dinâmica derivada da aproximação (11 colunas)
- `jerk_x_max`, `jerk_x_std`, `jerk_y_max`, `jerk_y_std`: o jerk é a variação brusca da aceleração.
- `n_perigo_accel_precurva`, `n_perigo_lateral_precurva`: quantas vezes a aceleração passou de um limite na janela.
- `precurva_abs_accel_max`: máximo de $\sqrt{(a_x^2+a_y^2)}$ na janela pré-curva
- `distance_car_curve`: o comprimento da janela pré-curva (a distância coberta pela aproximação). ⚠️ **Não é** a distância até a curva, veja a nota abaixo.
- `precurva_bearing_std`: desvio padrão das variações de bearing (Δθ) na janela pré-curva
- `precurva_bearing_range`: amplitude total do bearing na janela (max − min)
- `precurva_n_mudancas_dir`: contagem de alternâncias de sinal de Δbearing (|Δθ| > 15°) na janela

> A variável `distance_car_curve` é o tamanho da janela, em `m` de aproximação. A distância da ponta da janela até a entrada da curva é fixa e vale `lead_gap`  (**30 m** default). Só no `--velocity` existe uma variável que mede a distância que falta para a curva a cada instante: `distancia_restante` (ver Seção 4).

### Grupo 3 - Geometria da janela pré-curva (3 colunas)
`precurva_raio_min`, `precurva_raio_mean`, `precurva_raio_last`: o raio da pista durante a aproximação, pois mede o quanto a estrada já estava curvando antes da curva-alvo.

### Grupo 4 - Geometria da curva à frente (3 colunas)
`f4_raio_min`, `f4_raio_mean`, `f4_dnit_num`: o raio real da curva que vem pela frente e a sua classe oficial (seguindo o DNIT). Isso sob a suposição de que a rota é conhecida de antemão.

### Grupo 5 - Contexto das curvas anteriores (5 colunas)
Olham para o histórico do trajeto até aqui, usando só as curvas **anteriores**: `n_curvas_antes`,
`prev_isl_max` (ISL da curva imediatamente anterior), `mean_isl_antes` (ISL médio de todas as
curvas anteriores), `prev_raio_min`, `prev_raio_mean`, `prev_dnit_num` (geometria da curva
imediatamente anterior).

### Grupo 6 - Monte Carlo (3 colunas)
`mc_p_baixo`, `mc_p_medio`, `mc_p_alto`: uma simulação que, a partir da velocidade na aproximação e do raio da curva, estima a probabilidade de cada classe de ISL. Na prática é um "chute físico" embutido como feature, e que também foi usado para comparar com o modelo.

## 4. O comando `--velocity` usa um conjunto de features diferente

O `--velocity` **não** usa as 49 features acima. Ele trabalha com a **sequência no tempo** dos sensores. Ou seja, a série inteira da aproximação, não só as estatísticas resumidas. Ele usa:

- **Sensores ao longo do tempo** (50 instantes): `vehicle_speed`, `accel_x`, `accel_y`,  `engine_rpm` + o canal **`distancia_restante`**, que é a distância que falta para a entrada da curva em cada instante (decresce até zero na entrada, normalizada para [0, 1]).
- **Alguns números fixos por curva**: `jerk_y_max`, as contagens de perigo, a geometria da curva à frente, as probabilidades de Monte Carlo, e o contexto (`mean_isl_antes`, `prev_isl_max`, `n_curvas_antes`).
- **A resposta da curva anterior** (`prev_v_critica`): usa o valor da velocidade critica da curva passada como pista.

Para mudar as features usadas pela flag `--velocity`, edite `flags.velocity.sensors` e `.scalares_extras` no [features.yaml](../features.yaml).

> [!TIP]
> - **A previsão é antecipada.** A janela pré-curva termina 30 m **antes** da entrada
  (`features.yaml` > `extracao.lead_gap`). Nada medido na entrada ou dentro da curva pode ser feature.
> - **O contexto das curvas anteriores é atualizado de forma sequencial:** cada curva percorrida alimenta o histórico
da próxima. Neste dataset isso é reconstruído da mesma forma.