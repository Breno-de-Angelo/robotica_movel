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
($\mathbf{K}_u$ e \mathbf{K}_v$), através de uma experimentação que leve o quadrimotor a operar em condições diversas. Além disso, é preciso uma etapa experimental de ajuste dos ganhos (\mathbf{K}_p) e saturação (\mathbf{L}_s) do controlador cinemático e dos ganhos (\mathbf{K}) do controlador dinâmico.

É necessário ainda determinar $\mathbf{x}_d^w$ e $\dot{\mathbf{x}}_d^w$. 

Para o caso da trajetória o reultado é imediato. A trajetória é definida pela função $\mathbf{x}(t)$ e a sua derivada pode ser calculada analiticamente (como impelementada neste repositório) ou numericamente.

Para o caso de seguimento de caminho, o caminho é definido como um conjunto de pontos sequencias, determinados pela abcissa curvilínea `s`. Para cada ponto do caminho é computado o vetor tangente ao caminho chamado $\mathbf{T}$.
será implementado utilizando as seguintes regras:
- Calcular a distância do robô ao caminho e descobrir o ponto mais próximo ao robô no caminho.
- Caso a distância 

# Método experimental

O trabalho consiste em realizar o seguimento de trajetória e o seguimento de caminho através de uma malha de controle de laço interno e laço externo para o quadrimotor Bebob2.

Tanto para o caso de seguimento de trajetória quanto para o caso de seguimento de caminho, o drone deverá desenvolver um percurso circular de raio 1 m a uma altura constante de 1,5 m e mantendo sua orientação no espaço de 45º, devendo repetir o movimento 3 vezes.

Dessa forma, a trajetória pode ser descrita como

$$x_d$$
