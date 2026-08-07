A caracterização de risco roda **depois** da detecção de curvas e **antes** da extração de features. Ela produz colunas de rótulo por ponto (`manobra_*`, `conducao`) que, na extração de features, viram o **target** Segura/Risco por curva: uma curva é de Risco se **algum** dos seus pontos foi marcado.

Cada segmento de curva é um trecho contíguo com `curva = True`. Os dois critérios são avaliados sobre os pontos desse segmento, e os rótulos resultantes são atribuídos a esses mesmos pontos. Pontos fora de curvas ficam `Segura`.

> [!IMPORTANT]
> Os critérios de Kamm e de aceleração lateral foram removidos em julho de 2026. Ambos vinham do acelerômetro, que é do celular usado na coleta e não da central do veículo. A análise em `ANALISE_DADOS.md` mostra que esse sensor não acompanha a dinâmica do carro: correlação de 0,055 entre a leitura lateral e $v^2/R$, nenhuma inversão de sinal entre curvas para lados opostos, e apenas 0,3% do sinal recuperável pela melhor reorientação linear. Como controle, a aceleração longitudinal calculada pelo GPS concorda com a do OBD a 0,712, o que mostra que o problema é do sensor e não do método.

## Critérios de risco

Os dois critérios que restaram medem **conduta**: o motorista precisou corrigir depois de já estar dentro da curva. Ambos usam apenas velocidade e GPS.

### Frenagem tardia (`manobra_frenagem`)

Frear antes da curva é condução prudente. Frear já dentro dela indica que o motorista não antecipou o que vinha pela frente. A aceleração longitudinal sai da variação da velocidade entre pontos consecutivos do próprio segmento:

$$a_{\text{long}}(t_i) = \frac{v(t_i) - v(t_{i-1})}{\Delta t_i}$$

O critério dispara quando a maior desaceleração observada passa do limiar:

$$\max_i \; \big[-a_{\text{long}}(t_i)\big] > \tau_{\text{des}}$$

com $\tau_{\text{des}} = 2{,}0 \ \text{m/s}^2$ por padrão, o que marca cerca de 26% das curvas. Valores acima de 6 m/s² são descartados como falha de leitura da velocidade.

### Zigue-zague (`manobra_ziguezague`)

Para cada par de pontos consecutivos calcula-se a mudança de direção (bearing), no intervalo $(-180°, 180°]$ para preservar o sentido:

$$\Delta\theta_i = \big[(\theta_i - \theta_{i-1}) + 180\big] \bmod (360) - 180$$

Um ponto conta como **evento** quando a mudança é grande **e** há aceleração centrípeta ($a_{cp} = v^2/R$) acima de um limiar pequeno:

$$|\Delta\theta_i| > \tau_\theta \quad\textbf{e}\quad |a_{cp,i}| > \tau_{\text{ctp}}$$

O critério dispara se ocorrerem pelo menos `min_mudancas` eventos que alternam de sentido (direita, esquerda, direita…).

> [!NOTE]
> 🚗 Pontos com $v < 5$ km/h são ignorados: em baixa velocidade o espaçamento do GPS (~1–3 m) é da ordem do erro de posição (~3–5 m), gerando mudanças de bearing fictícias mesmo em linha reta.

## Combinação e rótulo final

$$\texttt{manobra-combinado} = \texttt{manobra-frenagem} \lor \texttt{manobra-ziguezague}$$

A coluna `conducao` é o mesmo resultado em texto: `'Perigosa'` se combinado, senão `'Segura'`.

### Por que a velocidade de entrada ficou de fora

O critério mais natural seria comparar a velocidade de entrada com a velocidade segura da curva, $v_{\text{seg}} = 3{,}6\sqrt{\mu g R}$. Ele foi testado e descartado de propósito: uma regressão logística com apenas a velocidade de aproximação e o raio da curva reproduz esse rótulo com F1 de 0,91. Prever algo que já é quase uma reescrita das próprias features não mede aprendizado nenhum. Essa dimensão continua coberta pelos alvos de ISL e `v_critica`, que são regressões sobre a mesma física.

## Classes DNIT

A classe DNIT vem da detecção de curvas e calculo do raio. Aqui ela é convertida num número de severidade (`risco_dnit`).

| Classe | Raio | `risco_dnit` |
|---|---|---|
| `suave` | $R > 500$ m | 0 |
| `aberta` | $R \le 500$ m | 1 |
| `media` | $R \le 200$ m | 2 |
| `fechada` | $R \le 100$ m | 3 |
| `muito_fechada` | $R \le 50$ m | 4 |

A `risco_dnit` é mantida como descritivo da geometria nas análises. Nenhum critério a usa como filtro desde a remoção do critério lateral.

## Parâmetros (`config.yaml` > `risk_measures`)

| Parâmetro | Default | Efeito |
|---|---|---|
| `limiar_desaceleracao` | 2.0 m/s² | $\tau_{\text{des}}$ do critério de frenagem tardia |
| `margem_histerese` | 0.10 | faixa morta em torno de $\tau_{\text{des}}$; curvas dentro dela saem do treino |
| `zigue_zague.limiar_bearing` | 15° | $\tau_\theta$ - mudança mínima de direção |
| `zigue_zague.limiar_ctp` | 0.3 m/s² | $\tau_{\text{ctp}}$ - aceleração centrípeta mínima (ctp_accel = v²/R) |
| `zigue_zague.min_mudancas` | 3 | nº de alternâncias para caracterizar zigue-zague |