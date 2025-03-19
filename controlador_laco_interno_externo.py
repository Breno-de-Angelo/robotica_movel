import numpy as np
from caminho import Caminho
from controlador_cinematico import ControladorCinematico
from controlador_dinamico import ControladorDinamico
from quadrimotor import Quadrimotor


class ControladorLacoInternoLacoExterno:
    """
    @param T: Período de amostragem (s)
    """
    T: float
    s_max: int
    controlador_cinemático: ControladorCinematico
    controlador_dinâmico: ControladorDinamico
    caminho: np.ndarray
    quadrimotor: Quadrimotor

    def __init__(
        self,
        T: float,
        controlador_cinemático: ControladorCinematico,
        controlador_dinâmico: ControladorDinamico,
        caminho: Caminho,
        quadrimotor: Quadrimotor,
    ):
        self.T = T
        self.controlador_cinemático = controlador_cinemático
        self.controlador_dinâmico = controlador_dinâmico
        self.s_max = caminho.s_max
        self.caminho = caminho.gerar_caminho()
        self.velocidade_caminho = np.diff(self.caminho, axis=0) / self.T
        self.quadrimotor = quadrimotor
        self.s = 0
    
    def forward(self):
        vd, self.s = self.controlador_cinemático.forward(
            s_atual=self.s,
            caminho=self.caminho,
        )
        vr = self.controlador_dinâmico.forward(vd)
        self.quadrimotor.update(vr, self.T)
        
        if self.s > self.s_max:
            return True
        return False
