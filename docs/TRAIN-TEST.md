# Treino, validação e teste

- **Treino ~70%**, **teste ~30%**.
- O corte é por **rota**, não por curva.

O treino não é usado inteiro de uma vez. Como se obtém a validação depende da representação, porque as duas precisam de coisas diferentes.

### Tabular (`cvt tab`)

O treino é dividido em **5 dobras**. Em cada rodada, 4 dobras treinam e 1 valida, girando 5 vezes. A métrica reportada é a média. As dobras respeitam a rota, ou seja, nenhuma gravação aparece ao mesmo tempo no que treina e no que valida, e nos alvos de sim ou não cada dobra também guarda a mesma proporção entre as classes. Nos alvos numéricos, como `cvt tab velocity`, guardar proporção de classe não faz sentido, então as dobras só respeitam a rota.

### Sequência (`cvt seq`)

As redes neurais treinam por épocas e precisam de um conjunto de validação **fixo** para acompanhar a curva de perda e decidir a hora de parar. Então uma fração fixa do treino é separada (`config.yaml > temporais.val_size`, 30% por padrão) e usada só para monitorar a perda. O treino para quando ela deixa de melhorar por `early_stopping_patience` épocas.

## SMOTE

Nos alvos de sim ou não, o desbalanceamento entre as classes é tratado criando exemplos sintéticos da classe rara, o que é o papel do SMOTE. Ele roda apenas sobre a parte de treino de cada dobra, nunca sobre o que valida nem sobre o teste. Nos modelos de sequência isso não é usado: interpolar séries no tempo distorceria o sinal, então o equilíbrio vem de dar mais peso à classe rara na função de perda.

| Papel | O que é | Como é criado | SMOTE? |
|---|---|---|---|
| **Treino** | ~70% das rotas | corte por rota | sim |
| **Validação** | fatia do treino | tabular: 5 dobras rotativas; sequência: fatia fixa | não |
| **Teste** | ~30% das rotas, intocado | corte por rota | não |
