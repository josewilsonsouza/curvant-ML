# Alvos e features
| Subcomando | Coluna | O que responde | Tipo |
|---|---|---|---|
| `velocity` | `v_critica` | com que velocidade o motorista vai chegar no ponto mais exigente | km/h |
| `isl` | `isl_p95` | quanto isso vai consumir da aderência do pneu | número |
| `correcao` | `correcao_tardia_curva` | se ele vai precisar corrigir depois de já estar na curva | sim ou não |

Cada alvo roda nas duas representações (`cvt tab <alvo>` e `cvt seq <alvo>`), que é como se compara se a série bruta traz algo além das features resumidas. As saídas vão para `results/<repr>/<alvo>/`. Em `cvt seq velocity` o modelo prevê a velocidade e a faixa de ISL na mesma passada, por uma segunda cabeça de classificação que usa a faixa medida como gabarito.

### Como o ISL é calculado

O ISL diz quanto da aderência disponível a curva consome. Em cada ponto:

$$\text{isl} = \frac{(v/3.6)^2}{\max(|R|,\,R_{\min}) \cdot g \cdot \mu}$$

O raio vem da B-spline ajustada ao GPS, e é ruidoso. Assim, o raio recebe um piso antes de entrar na divisão. A curva é resumida pelo percentil 95 dos seus pontos, não pelo máximo.

As faixas baixo, médio e alto saem de dois cortes fixos sobre essa grandeza contínua, e cerca de um quarto das curvas cai bem em cima de um dos cortes. Por isso o alvo é o valor contínuo: o modelo regride o `isl_p95` e a faixa sai do corte da predição, o que preserva a distância até o limiar.

## As features

Todas saem da janela de aproximação ou da geometria da rota, que é conhecida de antemão. A lista completa e ativa está em [features.yaml](../features.yaml). A representação tabular usa a lista `flags.tab.features`. A de sequência não usa essa lista: ela trabalha com a série no tempo, e por isso tem duas entradas separadas em `flags.seq`.