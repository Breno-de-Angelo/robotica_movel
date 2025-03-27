# Controlador laço interno e laço externo para quadrimotor Bebob2

Este repositório contém os códigos e resultados do trabalho final da disciplina de Robótica Móvel da Engenharia Elétrica - UFES, ministrada por Mário Sarcinelli Filho.

# Referencial teórico

A estrutura de controle utilizada foi a estrutura de laço externo e laço interno. Nesta estrutura, uma malha externa é responsável pelo controle cinemático do robô, enquanto a malha interna é responsável pelo controle dinâmico.
A figura abaixo apresenta diagrama de bloco do controlador.

![Screenshot from 2025-03-27 08-17-21](https://github.com/user-attachments/assets/72777fc2-660d-4ac5-9626-486a0344d33f)

Dessa forma, o controlador cinemático tem a responsabilidade de minimizar o erro de pose, recebendo como entrada a pose atual e entregando em sua saída a velocidade a ser imprimida para alcançar a pose.

A implementação do controlador cinemático é feita pelo seguinte diagrama de blocos:

![Screenshot from 2025-03-27 08-27-25](https://github.com/user-attachments/assets/09a57461-4810-4623-8f4e-cc3a8c3d662f)

Neste diagrama, $\dot{\mathbf{x}}_d^w(t)+\mathbf{L}_s\tanh(\mathbf{K}_p\tilde{x}^w)$ gera a velocidade desejada no referencial do mundo. O termo $\dot{\mathbf{x}}_d^w(t)$ é responsável por impôr
a velocidade de seguimento desejada do caminho/trajetória e o termo $\mathbf{L}_s\tanh(\mathbf{K}_p\tilde{x}^w)$ é responsável por minimizar a distância do robô ao caminho/trajetória.

Note que a presença da matriz de cinemática inversa $\mathbf{A}^{-1}$ realiza a conversão do sinal de controle do referencial do mundo para o referencial do robô.

De posse do vetor de velocidades desejado, o controlador dinâmico gera os sinais de controle que serão efetivamente enviados para planta. Ao introduzir um controlador para imprimir a velocidade desejada, é possível
projetar o controlador de forma a se obter melhores resultados para uma planta conhecida.

No caso do quadrimotor, seu modelo dinâmico se caracteriza pela equação a seguir:

$$
\ddot{\mathbf{x}}^b = \mathbf{K}_u\mathbf{U} - \mathbf{K}_v\dot{\mathbf{x}}^b
$$

Ou seja, a aceleração causada no quadrimotor é uma combinação do sinal de controle enviado a ele, devido à propulsão, e de sua velocidade atual, devido ao atrito viscoso. Ambas são matrizes diagonais positivo-definidas.

Assim, ao se projetar um controlador dinâmico segundo a lei abaixo, é possível se comprovar a estabilidade assintótica do controlador através do método de Lyapunov.

$$
\mathbf{v}_r = \mathbf{K}_u^{-1}[\dot{\mathbf{v}}_d + \mathbf{K}(\mathbf{v}_d - \dot{\mathbf{x}}^b) + \mathbf{K}_v\dot{\mathbf{x}}^b]
$$

Onde $\mathbf{v}_d$ é a velocidade desejada advinda do controlador cinemático e $\mathbf{K}$ é uma matriz diagonal de ganhos. Para que o erro seja minimizado, $\mathbf{K}$ deve ser uma matriz diagonal positivo-definida (neste caso se prova
a estabilidade assintótica).

Assim, a estrutura de controle completa receberá como entrada $\mathbf{x}_d^w$ e $\dot{\mathbf{x}}_d^w$ e produzirá o sinal de controle do robô. Para que o controlador tenha sucesso, é necessário que a planta seja corretamente identificada
($\mathbf{K}_u$ e $\mathbf{K}_v$), através de uma experimentação que leve o quadrimotor a operar em condições diversas. Além disso, é preciso uma etapa experimental de ajuste dos ganhos ($\mathbf{K}_p$) e saturação ($\mathbf{L}_s$) do controlador cinemático e dos ganhos ($\mathbf{K}$) do controlador dinâmico.

É necessário ainda determinar $\mathbf{x}_d^w$ e $\dot{\mathbf{x}}_d^w$. 

Para o caso da trajetória o reultado é imediato. A trajetória é definida pela função $\mathbf{x}(t)$ e a sua derivada pode ser calculada analiticamente (como impelementada neste repositório) ou numericamente.

Para o caso de seguimento de caminho, o caminho é definido como um conjunto de pontos sequencias, determinados pela abcissa curvilínea $s$. Para cada ponto do caminho é computado o vetor tangente ao caminho chamado $\mathbf{T}$.
será implementado utilizando as seguintes regras:
- Calcular a distância do robô ao caminho e descobrir o ponto mais próximo ao robô no caminho ($\mathbf{x}_{closest}$).
- $\mathbf{x}_ d = \mathbf{x}_{closest}$
- Caso a distância seja maior que $\epsilon$, $\dot{\mathbf{x}}_d = \mathbf{0}$
- Caso contrário,  $\dot{\mathbf{x}}_ d = V_e \cdot \mathbf{T}_{closest}$

# Método experimental

O trabalho consiste em realizar o seguimento de trajetória e o seguimento de caminho através de uma malha de controle de laço interno e laço externo para o quadrimotor Bebob2.

Tanto para o caso de seguimento de trajetória quanto para o caso de seguimento de caminho, o drone deverá desenvolver um percurso circular de raio 1 m a uma altura constante de 1,5 m e mantendo sua orientação no espaço de 45º, devendo repetir o movimento 3 vezes.

Para um resultado preliminar, foi realizada a tarefa de posicionamento usando o controlador de trajetória. Para isso, a trajetória foi descrita como:
- $x_d(t) = 0$
- $y_d(t) = 0$
- $z_d(t) = 1,5$
- $\psi_d(t) = 45\degree$

Com $0 \le t \le 75$

Dessa forma, a trajetória pode ser descrita como
- $x_d(t) = \cos (\frac{2\pi t}{25})$
- $y_d(t) = \sin (\frac{2\pi t}{25})$
- $z_d(t) = 1,5$
- $\psi_d(t) = 45\degree$

Com $0 \le t \le 75$

Já o caminho é descrito por:
- $x_d(s) = \cos (\frac{2\pi s}{25})$
- $y_d(s) = \sin (\frac{2\pi s}{25})$
- $z_d(s) = 1,5$
- $\psi_d(s) = 45\degree$

Com $0 \le s \le 75$ e discretizado em 5001 pontos.

Para o modelo do quadrimotor foram utilizados valores obtidos previamente em experimentos do laboratório, sendo $\mathbf{K}_u = diag(0,8417; 0,8354; 3,966; 9,8524)$ e sendo $\mathbf{K}_v = diag(0,18227; 0,17095; 4,001; 4,7295)$.

Os ganhos e valores de saturação do controlador cinemático e dinâmico foram testados e escolheu-se $\mathbf{K}_p = diag(1,0; 1,0; 1,0; 1,0)$, $\mathbf{L}_s = [1,0; 1,0; 1,0; 1,0]$ e $\mathbf{K} = diag(1,0; 1,0; 1,0; 1,0)$.

Para o seguidor de caminho utilizou-se $V_e=0,1$ m/s e $\epsilon = 0,2$ m.

A saída do controlador dinâmico foi limitada entre -1.0 e 1.0, como o quadrimotor Bebob2 espera receber.

O código foi inicialmente executado em modo de simulação para realizar o ajustes de todos os parâmetros e, uma vez incluídas as medidas de segurança no código, foi feito o experimento real.

# Resultados

Para cada um dos experimentos foi gerado um arquivo .mat com os resultados e analisados por scripts matlab que estão no diretório de resultados.

## Posicionamento



## Seguimento de Trajetória

## Seguimento de Caminho

# Conclusões

O controlador de laço interno e laço externo controlou com sucesso o quadrimotor. No posicionamento os erros convergiram para zero e no seguimento de trajetória e no seguimento de caminho, o erro em x e y oscilou em torno de zero com uma amplitude inferior a 20 cm (considerado o limite de distância ao caminho no controlador seguidor de caminho).

Na implementação foi considerado que o caminho seria o conjunto de potos que formariam três circunferências. Dessa forma, para garantir que o robô realizaria as três voltas, foi necessário implementar uma janela de observação ao fazer o cálculo do ponto mais próximo. Assim, ao invés de considerar todos os pontos do caminho, apenas uma parte era considerada. Uma solução alternativa seria definir o caminho como uma única circunferência e contar a quantidade de vezes que o robô voltou para o início do caminho para realizar 3 voltas.
