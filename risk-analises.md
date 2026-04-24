A utilização de sensores baseados no protocolo OBD-II (On-Board Diagnostics) para caracterizar a condução de risco permite o acesso direto à telemetria da Unidade de Controle do Motor (ECU), fornecendo dados precisos sobre a dinâmica do veículo. Os métodos que utilizam essas variáveis como *input* variam de algoritmos de aprendizado de máquina a sistemas de regras estatísticas e lógicas. 

Abaixo, apresento um grande resumo dos métodos e como as variáveis do OBD são utilizadas na caracterização dos riscos:

### Variáveis de Entrada (Inputs) do OBD-II
As abordagens extraem informações fundamentais e contínuas diretamente do barramento do veículo, sendo as principais:
*   **Velocidade do veículo (Speed)**
*   **Rotações Por Minuto (RPM)**
*   **Posição do pedal do acelerador (borboleta)**
*   **Carga calculada do motor (Engine Load)**

### 1. Método de Aprendizado de Máquina (AdaBoost)
O algoritmo AdaBoost é frequentemente utilizado para criar um classificador forte capaz de separar a "condução segura" da "condução perigosa" com alta precisão (atingindo até 99,8%). 
A caracterização do risco neste método não avalia apenas os dados brutos, mas as **taxas de mudança no tempo** (aceleração da velocidade, taxa de variação do RPM e da posição do acelerador). A partir dessas taxas, extraem-se três características matemáticas centrais:
*   A razão (proporção) entre a velocidade do veículo e o RPM do motor.
*   A razão entre a posição do acelerador e o RPM.
*   O nível de carga do motor.

**Caracterização do risco:** O algoritmo aprende que, em uma condução **normal e segura**, a proporção entre velocidade e RPM, e entre acelerador e RPM, costuma se manter no intervalo de 0,9 a 1,3, enquanto a carga do motor fica entre 20% e 50%. Quando os limites excedem ou caem abaixo dessas proporções (ex: carga acima de 50% ou abaixo de 20%, proporções maiores que 1,3), o comportamento é classificado como uma anomalia ou má condução (*bad vehicle condition*).

### 2. Método de Sistema de Pontuação Baseado em Limiares (Regras Heurísticas)
Este método atua como um sistema avaliativo que debita pontos (Score) do motorista com base na identificação de eventos que violem parâmetros operacionais seguros.
Nesse método de limites fixos, a caracterização de anomalias ocorre pela verificação de condições específicas:
*   **Acelerações bruscas:** Detectadas cruzando os dados da velocidade e do RPM; por exemplo, se a rotação do motor for superior a **3500 RPM** atrelada ao acionamento abrupto da tração ou diferença súbita na velocidade, é caracterizada a infração.
*   **Frenagens bruscas e excesso de velocidade:** A velocidade lida pelo OBD é comparada com limites pré-estabelecidos e a desaceleração é calculada matematicamente para determinar se a frenagem foi perigosa. 

### 3. Método Baseado em Lógica Fuzzy (Lógica Difusa)
A lógica Fuzzy é usada para modelar o comportamento de condução focando primariamente no desperdício de combustível, que atua como um indicador direto de condução agressiva ou ineficiente. O método converte variáveis contínuas lidas pelo adaptador OBD (velocidade, RPM e posição do acelerador) em variáveis linguísticas (conjuntos fuzzy), variando de "muito baixo" a "muito alto".

**Caracterização do risco:** O método estabelece que a **posição do acelerador é a variável mais importante** para inferir um comportamento de risco/alto consumo, seguida pelo RPM. Regras de inferência, definidas com a ajuda de especialistas (ex: *"SE a velocidade é muito baixa E o RPM é muito baixo E o acelerador é MUITO pressionado ENTÃO o consumo é ALTO"*), classificam a severidade e a ineficiência do estilo de condução.

### 4. Fusão de Dados Sensoriais (OBD + Sensores de Smartphone)
Muitos métodos contemporâneos utilizam uma arquitetura híbrida (Gateway). Como o OBD captura estritamente o comportamento mecânico, ele é pareado via Bluetooth a *smartphones* para combinar a telemetria do motor (RPM, Velocidade, Carga) com dados espaciais (Acelerômetro, Giroscópio, GPS). 
Nesse modelo arquitetônico, o comportamento de risco ganha contexto:
*   O OBD informa o pico do **RPM e velocidade**.
*   O smartphone informa a **aceleração lateral (força G em curvas)** e a **localização exata** para confrontar a velocidade do OBD com o limite de velocidade imposto na via lida pelo GPS. A integração desses sensores permite um processamento em tempo real que penaliza atitudes no volante de forma muito mais completa.
