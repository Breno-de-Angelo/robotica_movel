import numpy as np
from typing import Callable


class Caminho:
    """
    @param caminho: função que retorna a posição e orientação do caminho
    @param s_min: ponto inicial do caminho
    @param s_max: ponto final do caminho
    @param N: número de pontos do caminho
    """
    caminho: Callable[[int], np.ndarray]
    s_min: int
    s_max: int
    N: int
    delta: float
    Vd: float

    def __init__(
        self,
        caminho: Callable[[int], np.ndarray],
        s_min: int,
        s_max: int,
        N: int,
        delta: float,
        Vd: float
    ):
        self.caminho = caminho
        self.s_min = s_min
        self.s_max = s_max
        self.N = N
        self.delta = delta
        self.Vd = Vd

    def gerar_caminho(self):
        caminho = [[s, *self.caminho(s)] for s in np.linspace(self.s_min, self.s_max, self.N)]
        for i in range(1, len(caminho)):
            if (caminho[i][5] - caminho[i-1][5]) > np.pi / 2:
                for j in range(i, len(caminho)):
                    caminho[j][5] -= np.pi
            if (caminho[i][6] - caminho[i-1][6]) > np.pi / 2:
                for j in range(i, len(caminho)):
                    caminho[j][6] -= np.pi
        return np.array(caminho)