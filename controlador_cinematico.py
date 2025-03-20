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
        # posicao_atual = self.modelo.pose
        # vetor_distancias = np.linalg.norm(caminho[:,1:4] - posicao_atual[:3], axis=1)
        # index_ponto_mais_proximo = np.argmin(vetor_distancias)

        posicao_atual = self.modelo.pose[:3]
        pose_atual = self.modelo.pose[:4]
        index_ponto_mais_proximo = np.where(caminho[:, 0] == s_atual)[0][0]
        distancia = np.linalg.norm(caminho[index_ponto_mais_proximo][1:4] - posicao_atual)
        print(f"Distancia: {distancia}")
        if (distancia > self.delta):
            velocidade_seguimento = 0
        else:
            velocidade_seguimento = self.Vd
            index_ponto_mais_proximo += 1

        if index_ponto_mais_proximo >= len(caminho):
            # Finalizou caminho
            return None
        s = caminho[index_ponto_mais_proximo, 0]
        print(f"s={s}")

        cinematica_inversa = self.modelo.cinemática_inversa()
        velocidade_seguimento_mundo = np.array([
            velocidade_seguimento * np.cos(caminho[index_ponto_mais_proximo][6]) * np.cos(caminho[index_ponto_mais_proximo][5]),
            velocidade_seguimento * np.cos(caminho[index_ponto_mais_proximo][6]) * np.sin(caminho[index_ponto_mais_proximo][5]),
            velocidade_seguimento * np.sin(caminho[index_ponto_mais_proximo][6]),
            0.0
        ])
        velocidade_aproximacao = self.Ls * np.tanh(self.Kp) * (caminho[index_ponto_mais_proximo][1:5] - pose_atual)

        vd = cinematica_inversa @ (velocidade_seguimento_mundo + velocidade_aproximacao)

        return vd, s