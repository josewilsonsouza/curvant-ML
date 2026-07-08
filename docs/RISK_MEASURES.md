A caracterização de risco roda **depois** da detecção de curvas e **antes** da extração de features. Ela produz colunas de rótulo por ponto (`manobra_*`, `conducao`) que, na extração de features, viram o **target** Segura/Risco por curva: uma curva é de Risco se **algum** dos seus pontos foi marcado.

Cada segmento de curva é um trecho contíguo com `curva = True`. Os três critérios são avaliados sobre os pontos desse segmento, e os rótulos resultantes são atribuídos a esses mesmos pontos. Pontos fora de curvas ficam `Segura`.

## 2. Critérios de risco

Sejam $a_x$ e $a_y$ as acelerações longitudinal (frear/acelerar) e lateral, $\mu$ o coeficiente de atrito e $g$ a gravidade. Os critérios olham o pico ao longo do segmento da curva.

### 2.1 Kamm (`manobra_accel`)

O pneu tem um limite de aderência, a aceleração total que ele aguenta antes de derrapar é $\mu g$. Frear e curvar dividem o mesmo círculo de Kamm, então o que importa é a soma vetorial:

$$a_{\text{total}}(t) = \sqrt{a_x(t)^2 + a_y(t)^2}$$

O critério dispara se, em qualquer instante, essa soma passa de uma fração $\alpha$ do limite:

$$\max_t \; a_{\text{total}}(t) \;>\; \alpha \,\mu\, g$$

Com os valores padrão ($\alpha = 0{,}7$, $\mu = 0{,}6$, $g = 9{,}81$):

$$\text{limiar} = 0{,}7 \times 0{,}6 \times 9{,}81 \;\approx\; 4{,}12 \ \text{m/s}^2$$

O $\alpha = 0{,}7$ é uma **margem de segurança**: alarma já aos 70% do limite real, capturando a condução perto da borda antes da derrapagem de fato.

### 2.2 Lateral (`manobra_lateral`)

Dispara quando há pico de aceleração lateral alto **e** a curva é geometricamente relevante:

$$\max_t \; |a_y(t)| \;>\; \tau_{\text{lat}} \quad\textbf{e}\quad R \le 500m$$

com $\tau_{\text{lat}} = 2{,}0 \ \text{m/s}^2$ por padrão. O $R$ evita que troca de faixa ou de buracos em retas seja contada como curva perigosa.

### 2.3 Zigue-zague (`manobra_ziguezague`)

Para cada par de pontos consecutivos calcula-se a mudança de direção (bearing), no intervalo $(-180°, 180°]$ para preservar o sentido:

$$\Delta\theta_i = \big[(\theta_i - \theta_{i-1}) + 180\big] \bmod (360) - 180$$

Um ponto conta como **evento** quando a mudança é grande **e** há aceleração centrípeta ($a_{cp} = v^2/R$) acima de um limiar pequeno:

$$|\Delta\theta_i| > \tau_\theta \quad\textbf{e}\quad |a_{cp,i}| > \tau_{\text{ctp}}$$

O critério dispara se ocorrerem pelo menos `min_mudancas` eventos que alternam de sentido (direita, esquerda, direita…).

> [!NOTE]
> 🚗 Pontos com $v < 5$ km/h são ignorados: em baixa velocidade o espaçamento do GPS (~1–3 m) é da ordem do erro de posição (~3–5 m), gerando mudanças de bearing fictícias mesmo em linha reta.

## 3. Combinação e rótulo final

$$\texttt{manobra\_combinado} = \texttt{manobra\_accel} \;\lor\; \texttt{manobra\_lateral} \;\lor\; \texttt{manobra\_ziguezague}$$

A coluna `conducao` é o mesmo resultado em texto: `'Perigosa'` se combinado, senão `'Segura'`.

## 4. Classes DNIT

A classe DNIT vem da detecção de curvas e calculo do raio. Aqui ela é convertida num número de severidade (`risco_dnit`).

| Classe | Raio | `risco_dnit` |
|---|---|---|
| `suave` | $R > 500$ m | 0 |
| `aberta` | $R \le 500$ m | 1 |
| `media` | $R \le 200$ m | 2 |
| `fechada` | $R \le 100$ m | 3 |
| `muito_fechada` | $R \le 50$ m | 4 |

O critério lateral exige `risco_dnit` $\ge 1$ (aberta ou mais fechada).

## 5. Parâmetros (`config.yaml` > `risk_measures`)

| Parâmetro | Default | Efeito |
|---|---|---|
| `kamm_alpha` | 0.7 | fração $\alpha$ do limite de aderência no critério de Kamm |
| `limiar_accel_lateral` | 2.0 m/s² | $\tau_{\text{lat}}$ do critério lateral |
| `zigue_zague.limiar_bearing` | 15° | $\tau_\theta$ — mudança mínima de direção |
| `zigue_zague.limiar_ctp` | 0.3 m/s² | $\tau_{\text{ctp}}$ — aceleração centrípeta mínima (ctp_accel = v²/R) |
| `zigue_zague.min_mudancas` | 3 | nº de alternâncias para caracterizar zigue-zague |