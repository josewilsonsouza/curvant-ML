# Medidas de Risco

Este módulo é quem decide se uma curva foi **Segura ou de Risco**. Ele olha o que o motorista fez ao passar pela curva e aplica **três critérios** independentes; se qualquer um disparar, a curva é marcada como de Risco. É daqui que sai o rótulo `manobra_combinado_curva`, o alvo que o comando `--risco` aprende a prever (veja [TARGETS_E_FEATURES.md](TARGETS_E_FEATURES.md)).

Código: [curvant/driving/risk_measures.py](../curvant/driving/risk_measures.py).

## 1. Onde isso entra no pipeline

A caracterização de risco roda **depois** da detecção de curvas e **antes** da extração de features. Ela produz colunas de rótulo (`manobra_*`, `conducao`) que:

- viram o **target** Segura/Risco (uma curva é de Risco se **algum** dos seus pontos foi marcado);
- alimentam features de contexto (`prop_perigosas_antes`, etc.), usando só curvas anteriores.

Importante: esses rótulos são **targets** (o que queremos prever), não features. Por isso a janela de avaliação pode olhar a entrada da curva, ao contrário das features (veja a nota das três janelas em [TARGETS_E_FEATURES.md](TARGETS_E_FEATURES.md)).

Para cada segmento de curva (trecho contíguo com `curva = True`), os critérios são avaliados sobre uma janela que inclui:

$$\text{janela de avaliação} = (\text{aproximação: } \texttt{janela\_aproximacao}\text{ s antes}) \;+\; (\text{pontos da curva})$$

Os rótulos resultantes são atribuídos apenas aos pontos do segmento da curva; pontos fora de curvas ficam `Segura`. A aproximação (5 s por padrão) entra porque uma manobra perigosa costuma começar antes da curva (uma freada brusca na entrada, por exemplo).

## 3. Os três critérios de risco

Sejam $a_x$ e $a_y$ as acelerações longitudinal (frear/acelerar) e lateral, $\mu$ o coeficiente de atrito e $g$ a gravidade. Os critérios olham o pico ao longo da janela.

### 3.1 Kamm - aceleração total (`manobra_accel`)

O pneu tem um limite de aderência, a aceleração total que ele aguenta antes de derrapar é $\mu g$. Frear e curvar dividem o mesmo círculo de Kamm, então o que importa é a soma vetorial:

$$a_{\text{total}}(t) = \sqrt{a_x(t)^2 + a_y(t)^2}$$

O critério dispara se, em qualquer instante, essa soma passa de uma fração $\alpha$ do limite:

$$\max_t \; a_{\text{total}}(t) \;>\; \alpha \,\mu\, g$$

Com os valores padrão ($\alpha = 0{,}7$, $\mu = 0{,}6$, $g = 9{,}81$):

$$\text{limiar} = 0{,}7 \times 0{,}6 \times 9{,}81 \;\approx\; 4{,}12 \ \text{m/s}^2$$

O $\alpha = 0{,}7$ é uma **margem de segurança**: alarma já aos 70% do limite real, capturando a condução perto da borda antes da derrapagem de fato.

### 3.2 Lateral - aceleração de lado em curva real (`manobra_lateral`)

Dispara quando há pico de aceleração lateral alto **e** a curva é geometricamente relevante:

$$\max_t \; |a_y(t)| \;>\; \tau_{\text{lat}} \quad\textbf{e}\quad \text{classe\_dnit} \ge \text{aberta}$$

com $\tau_{\text{lat}} = 2{,}0 \ \text{m/s}^2$ por padrão. O "portão" do DNIT (curva no mínimo*aberta*, $R \le 500$ m) evita que troca de faixa ou de buracos em retas seja contada como curva perigosa.

### 3.3 Zigue-zague - oscilação de direção (`manobra_ziguezague`)

Para cada par de pontos consecutivos calcula-se a mudança de
direção (bearing), no intervalo $(-180°, 180°]$ para preservar o sentido:

$$\Delta\theta_i = \big[(\theta_i - \theta_{i-1}) + 180\big] \bmod 360 - 180$$

Um ponto conta como **evento** quando a mudança é grande **e** há aceleração centrípeta ($a_{cp} = v^2/R$) acima de um limiar pequeno:

$$|\Delta\theta_i| > \tau_\theta \quad\textbf{e}\quad |a_{cp,i}| > \tau_{\text{lat,zz}}$$

O critério dispara se ocorrerem pelo menos `min_mudancas` eventos que **alternam de sentido** (direita, esquerda, direita…). Curvas contínuas (sempre o mesmo sentido) **não** contam, só a alternância caracteriza zigue-zague.

> 🚗 Pontos com $v < 5$ km/h são ignorados: em baixa velocidade o espaçamento do GPS (~1–3 m) é da ordem do erro de posição (~3–5 m), gerando mudanças de bearing fictícias mesmo em linha reta.

## 4. Combinação e rótulo final

$$\texttt{manobra\_combinado} = \texttt{manobra\_accel} \;\lor\; \texttt{manobra\_lateral} \;\lor\; \texttt{manobra\_ziguezague}$$

A coluna `conducao` é o mesmo resultado em texto: `'Perigosa'` se combinado, senão `'Segura'`.

## 5. Classes DNIT

A classe DNIT vem da detecção de curvas (por faixa de raio). Aqui ela é convertida num número de severidade (`risco_dnit`), usado no critério lateral:

| Classe | Raio | `risco_dnit` |
|---|---|---|
| `suave` | $R > 500$ m | 0 |
| `aberta` | $R \le 500$ m | 1 |
| `media` | $R \le 200$ m | 2 |
| `fechada` | $R \le 100$ m | 3 |
| `muito_fechada` | $R \le 50$ m | 4 |

O critério lateral exige `risco_dnit` $\ge 1$ (aberta ou mais fechada).

## 6. Parâmetros (`config.yaml` > `risk_measures`)

| Parâmetro | Default | Efeito |
|---|---|---|
| `kamm_alpha` | 0.7 | fração $\alpha$ do limite de aderência no critério de Kamm |
| `limiar_accel_lateral` | 2.0 m/s² | $\tau_{\text{lat}}$ do critério lateral |
| `zigue_zague.limiar_bearing` | 15° | $\tau_\theta$ — mudança mínima de direção |
| `zigue_zague.limiar_ctp` | 0.3 m/s² | $\tau_{\text{ctp}}$ — aceleração centrípeta mínima (ctp_accel = v²/R) |
| `zigue_zague.min_mudancas` | 3 | nº de alternâncias para caracterizar zigue-zague |

Em geral: subir um limiar deixa o critério mais permissivo (menos curvas viram Risco); descer deixa mais sensível (mais curvas viram Risco).

## 7. Colunas resultantes

| Coluna | O que é |
|---|---|
| `manobra_accel` | critério de Kamm disparou (bool, por ponto do segmento) |
| `manobra_lateral` | critério lateral disparou |
| `manobra_ziguezague` | critério de zigue-zague disparou |
| `manobra_combinado` | OR dos três |
| `conducao` | `'Perigosa'` / `'Segura'` (versão em texto do combinado) |
| `risco_dnit` | severidade DNIT da curva (0–4) |
| `id_janela` | identificador sequencial do segmento de curva avaliado |

Na extração de features, esses rótulos por ponto viram os **targets por curva** (`manobra_accel_curva`, `manobra_lateral_curva`, `manobra_ziguezague_curva`, `manobra_combinado_curva`): a curva recebe 1 se **algum** dos seus pontos foi marcado.
