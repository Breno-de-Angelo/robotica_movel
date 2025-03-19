import numpy as np
from quadrimotor import Quadrimotor


class ControladorCinematico:
    """
    \\dot{x} = A^{-1} (\\dot{x}_d + l \\tanh(kp (x_d - x)))

    @param delta: Máximo erro de posição (m)
    @param Vd: Velocidade desejada (m/s)
    """
    Kp: np.ndarray
    Ls: np.ndarray
    delta: float
    Vd: float
    modelo: Quadrimotor

    def __init__(
        self,
        Kp: np.ndarray,
        Ls: np.ndarray,
        delta: float,
        Vd: float,
        modelo: Quadrimotor,
    ):
        self.Kp = Kp
        self.Ls = Ls
        self.delta = delta
        self.Vd = Vd
        self.modelo = modelo
    
    def forward(
        self,
        s_atual: int,
        caminho: np.ndarray,
    ):
        posicao_atual = self.modelo.pose
        vetor_distancias = np.linalg.norm(caminho[:,:4] - posicao_atual, axis=0)
        s_ponto_mais_proximo = np.argmin(vetor_distancias)

        print(s_ponto_mais_proximo)
        print(vetor_distancias)
        print(f"Distancia: {vetor_distancias[s_ponto_mais_proximo]}")
        
        if (vetor_distancias[s_ponto_mais_proximo] > self.delta):
            velocidade_seguimento = 0
            s = s_ponto_mais_proximo
        else:
            velocidade_seguimento = self.Vd
            s = s_atual + 1

        cinematica_inversa = self.modelo.cinemática_inversa()
        velocidade_seguimento_mundo = np.array([
            velocidade_seguimento * np.cos(caminho[s][4]) * np.cos(caminho[s][5]),
            velocidade_seguimento * np.cos(caminho[s][4]) * np.sin(caminho[s][5]),
            velocidade_seguimento * np.sin(caminho[s][4]),
            0.0
        ])
        # velocidade_aproximacao = np.array([
        #     self.Ls.x * np.tanh(self.Kp[0] * (caminho[s].x - posicao_atual.x)),
        #     self.Ls.y * np.tanh(self.Kp[1] * (caminho[s].y - posicao_atual.y)),
        #     self.Ls.z * np.tanh(self.Kp[2] * (caminho[s].z - posicao_atual.z)),
        # ])
        velocidade_aproximacao = self.Ls * np.tanh(self.Kp) * (caminho[s][:4] - posicao_atual)

        vd = cinematica_inversa @ (velocidade_seguimento_mundo + velocidade_aproximacao)

        return vd, s