# Targets e Features — guia explicativo

Este arquivo explica, de forma simples, **o que o projeto tenta prever** e **com quais dados**.
Serve para não se perder entre as muitas colunas que o pipeline gera.

---

## 0. Conceitos básicos (leia primeiro)

O projeto quer dizer, **antes do carro entrar numa curva**, o quão arriscada ela vai ser.
Para isso, cada curva vira uma linha de uma tabela, e essa tabela tem dois tipos de coluna:

- **Alvo** (o que queremos prever): por exemplo, a velocidade no ponto mais crítico da curva,
  ou a classe de risco. É a "resposta".
- **Feature** (o que usamos para prever): por exemplo, a velocidade média na aproximação,
  o raio da curva à frente. São as "pistas" que o modelo usa para chegar na resposta.

Dois conceitos que aparecem o tempo todo:

- **Janela pré-curva:** é o trecho do percurso **antes** da curva. Todas as features saem daí,
  porque a previsão é feita *antes de chegar na curva*. A janela termina um pouco antes da
  entrada (uma folga, o `lead_gap`, hoje 30 m), simulando um sistema que avisa o motorista
  com antecedência.
- **Vazamento de informação (como colar numa prova):** se a gente usasse como feature algo que
  só dá para medir **dentro** da curva (ex.: a velocidade no meio dela), o modelo estaria
  "vendo a resposta". Por isso essas colunas ficam de fora das features. Há uma lista única no
  arquivo [src/models.py](src/models.py) (a função `colunas_features`) que decide o que é
  feature e o que não é — qualquer coluna que não esteja na lista de exclusão vira feature.

Hoje a tabela tem **84 colunas**: 50 features + 21 alvos + 4 identificadores + 9 colunas que
ficam de fora por serem medidas dentro/na entrada da curva.

---

## 1. O que cada comando prevê

Cada flag do `scripts/run.py` treina um alvo diferente:

| Comando | O que prevê | Tipo |
|---|---|---|
| `run.py` (padrão), `--classical`, `--optuna`, `--mlp` | curva é **Segura ou de Risco** (`manobra_combinado_curva`) | sim/não |
| `--fi` | mesma coisa, mas só mostra **quais features pesam mais** | — |
| `--isl` | a **classe de risco ISL** (baixo / médio / alto) | 3 classes |
| `--isl` | o **pico de aceleração dentro da curva** | número |
| `--pytorch` | o **ISL máximo** + as 4 colunas de manobra, tudo de uma vez | misto |
| `--ts` | a **velocidade crítica** da curva (`v_critica`) | número |

**Importante:** o comando padrão (sem flag) **não prevê ISL** — ele prevê apenas se a curva é
Segura ou de Risco. O ISL só aparece com `--isl`, `--pytorch` ou `--ts`.

> **Direção atual do projeto:** o caminho mais promissor é o `--ts` — prever a **velocidade
> crítica** e, a partir dela, calcular o ISL pela fórmula da física. É onde o modelo de fato
> supera um chute simples (ver Seção 6).

---

## 2. Os alvos — o que tentamos prever (21 colunas)

Todos são medidos **dentro da curva**, então nunca são usados como features.

### 2.1 Alvos realmente treinados por algum comando

| Coluna | O que é, em palavras simples |
|---|---|
| `manobra_combinado_curva` | A curva foi de **Risco** (1) ou **Segura** (0). É "sim se qualquer um dos 3 critérios de risco disparou". |
| `manobra_accel_curva` | Disparou o critério do **limite de aderência do pneu** (círculo de Kamm): a aceleração total passou de uma fração do que o pneu aguenta. |
| `manobra_lateral_curva` | Houve **aceleração lateral alta** numa curva fechada o bastante. |
| `manobra_ziguezague_curva` | Houve **zigue-zague** (mudanças de direção repetidas com aceleração lateral). |
| `isl_class` | A classe de risco ISL: **baixo, médio ou alto**. |
| `isl_max` | O **maior ISL** atingido na curva. ISL = v² / (R·g·μ): quanto a curva chegou perto do limite de derrapagem. |
| `curve_accel_y_max` | O **pico de aceleração lateral** (g lateral) dentro da curva. |
| `curve_abs_accel_max` | O **pico de aceleração total** dentro da curva. |
| `v_critica` | A **velocidade (km/h) no ponto de maior risco** da curva. Este é o alvo do `--ts`. |

### 2.2 Alvos que existem mas não são treinados por padrão (alternativas)

Estão calculados e prontos, mas nenhum comando os usa por enquanto. Para treinar um deles,
basta apontá-lo como alvo (ex.: `time_series_regression.target` no `config.yaml`).

| Coluna | O que é |
|---|---|
| `manobra` | Mesmo que `manobra_combinado_curva` (nome antigo, mantido por compatibilidade). |
| `isl_mean` | ISL **médio** da curva (em vez do máximo). |
| `isl_alto` | Sinal de sim/não: o ISL passou de 0,8? |
| `isl_sensor_max` / `_mean` / `_class` | Uma versão do ISL calculada pelo **acelerômetro** (\|accel_y\|/g·μ) em vez do raio do GPS. Não depende do raio, então não sofre com ruído de GPS. |
| `curve_accel_y_mean` / `curve_abs_accel_mean` | As **médias** das acelerações (os treinados são os picos `_max`). |
| `manobra_velocidade` / `v_excess` | O motorista **entrou acima da velocidade segura** para aquele raio. |
| `v_safe_dnit` | A **velocidade segura** teórica para o raio da curva = √(R·g·μ). |
| `v_entry_ratio` | Quão acima da velocidade segura o carro entrou (razão). |

---

## 3. As features — o que usamos para prever (50 colunas)

Estão organizadas em grupos. **Todas vêm da janela pré-curva ou de algo que já se conhece de
antemão** (a geometria da estrada à frente, porque a rota é conhecida). Nenhuma é medida dentro
da curva.

### Grupo 1 — Sensores OBD na aproximação (27 colunas)
As estatísticas básicas dos sensores na janela antes da curva. Para `vehicle_speed`,
`accel_x` e `accel_y`, calculamos 9 números cada:

`_mean` (média), `_std` (desvio), `_median` (mediana), `_max`, `_min`, `_slope` (tendência:
está acelerando ou freando?), `_cv` (variação relativa), `_mean_tarde` e `_slope_tarde`
(os mesmos, mas só na **metade final** da janela — o comportamento mais perto da curva).

> O `engine_rpm` está desligado no `config.yaml` (`features.vars_sensor`), então não há colunas
> de RPM.

### Grupo 2 — Dinâmica derivada da aproximação (8 colunas)
- `jerk_x_max`, `jerk_x_std`, `jerk_y_max`, `jerk_y_std`: o **jerk** é a variação brusca da
  aceleração (solavanco). Capta freadas/esterçadas nervosas.
- `n_perigo_accel_janela`, `n_perigo_lateral_janela`: **quantas vezes** a aceleração passou de
  um limite na janela (contagem de "sustos").
- `distance_car_curve`: tamanho da janela usada (distância coberta antes da curva).
- `v_pred_kinematica`: uma tentativa de **estimar a velocidade de entrada** por física.
  ⚠️ **Está com problema** (ver Seção 6) — enviesada, deveria ser revista.

### Grupo 3 — Geometria da janela pré-curva (3 colunas)
`janela_raio_min`, `janela_raio_mean`, `janela_raio_last`: o **raio da pista** durante a
aproximação (o quanto a estrada já estava curvando antes da curva-alvo).

### Grupo 4 — Geometria da curva à frente (3 colunas)
`f4_raio_min`, `f4_raio_mean`, `f4_dnit_num`: o **raio real da curva** que vem pela frente e a
sua classe oficial (DNIT = classificação de curvas por raio, do manual de rodovias). É feature
legítima porque a rota é conhecida de antemão.

### Grupo 5 — Contexto das curvas anteriores (6 colunas)
Olham para o **histórico do trajeto até aqui**, usando só as curvas **anteriores** (nunca a
atual, então não é trapaça):
`n_curvas_antes`, `n_perigosas_antes`, `prop_perigosas_antes` (quantas/que fração das curvas
anteriores foram de risco), `prev_raio_min`, `prev_raio_mean`, `prev_dnit_num` (geometria da
curva imediatamente anterior).

### Grupo 6 — Monte Carlo (3 colunas)
`mc_p_baixo`, `mc_p_medio`, `mc_p_alto`: uma **simulação** que, a partir da velocidade na
aproximação e do raio da curva, estima a **probabilidade de cada classe de ISL**. Na prática é
um "chute físico" embutido como feature — e também serve de referência para comparar com o modelo.

---

## 4. O comando `--ts` usa um conjunto de features diferente

Atenção: o `--ts` **não** usa as 50 features acima. Ele trabalha com a **sequência no tempo**
dos sensores (a série inteira da aproximação, não só as estatísticas resumidas). Ele usa:

- **Sensores ao longo do tempo** (50 instantes): `vehicle_speed`, `accel_x`, `accel_y`,
  `engine_rpm` + a distância que falta para a curva.
- **Alguns números fixos por curva** (15, um subconjunto escolhido): os raios da janela,
  `jerk_y_max`, as contagens de perigo, a geometria da curva à frente, as probabilidades de
  Monte Carlo, `v_pred_kinematica`, e o contexto (`prop_perigosas_antes`, `n_curvas_antes`).
- **A resposta da curva anterior** (`prev_v_critica`): usa o valor da curva passada como pista.

Para mudar as features do `--ts`, edite `time_series_regression.sensors` e `.scalares_extras`
no [config.yaml](config.yaml) — **não** a lista do `models.py`.

---

## 5. Colunas que ficam DE FORA das features (e por quê)

| Por quê | Quantas | Quais |
|---|---|---|
| **Identificador** (não é dado, é etiqueta da linha) | 4 | `id_route`, `id_trecho_curvo`, `time_inicio`, `time_fim` |
| **É um alvo** (é a resposta, não pode ser pista) | 21 | ver Seção 2 |
| **Medido dentro da curva** (seria ver a resposta) | 3 | `curve_raio_min`, `curve_raio_mean`, `curve_dnit_num` |
| **Medido na entrada da curva** (já é tarde demais; a previsão é antecipada) | 6 | `v_entry`, `v_speed_drop`, `v_speed_drop_pct`, `v_entry_vs_mean`, `v_entry_sq_over_raio_est`, `isl_entry_estimate` |

**Como adicionar uma feature nova:** se ela for honesta (sai da janela pré-curva ou é geometria
conhecida de antemão), basta criá-la no [src/features.py](src/features.py) — ela já entra como
feature automaticamente. Se ela for medida na ou depois da entrada da curva, **adicione o nome
dela à lista de exclusão** no `models.py`, senão ela vira "cola".

---

## 6. Avisos importantes

- **A previsão é antecipada.** A janela pré-curva termina 30 m **antes** da entrada
  (`config.features.lead_gap`). Nada medido na entrada ou dentro da curva pode ser feature.

- **A feature `v_pred_kinematica` está quebrada.** Como preditor da velocidade crítica ela é
  pior que chutar a média (testamos: R² = −1,09). A fórmula `v² = v_mean² − 2·a·d` zera a
  velocidade prevista em janelas longas. Ela ainda está na lista de features e no `--ts`, mas
  **deveria ser revista ou removida**.

- **O projeto compara o modelo com um "chute simples".** Sob `--isl` e `--ts` o pipeline mostra,
  ao lado do modelo, o resultado de uma referência sem aprendizado (a simulação de Monte Carlo,
  ou "a velocidade na curva ≈ a velocidade de aproximação"). Isso mede **o quanto o modelo
  realmente acrescenta**. Conclusão até agora: para a **classe** de ISL o ganho é pequeno (a
  física já acerta quase tudo), mas para a **velocidade crítica** o modelo ganha bastante
  (erro cai de ~11 para ~7 km/h).

- **Por que `v_critica` é o alvo mais limpo.** Como ISL = v² / (R·g·μ) e o raio R é conhecido de
  antemão, prever o ISL com R conhecido é basicamente prever a velocidade na curva. Então faz
  mais sentido prever a velocidade direto e calcular o ISL pela fórmula.
