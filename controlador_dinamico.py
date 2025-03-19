import numpy as np
from utils import Vector3d
from quadrimotor import Quadrimotor


class ControladorDinamico:
    """
    vr = ku^{-1} (K (vd - vb) + Kv vb)
    """
    K: Vector3d
    modelo: Quadrimotor

    def __init__(
        self,
        K: Vector3d,
        modelo: Quadrimotor,
    ):
        self.K = K
        self.modelo = modelo
    
    def forward(self, vd: np.ndarray) -> np.ndarray:
        # inverse_Ku = np.array([
        #     self.modelo.Ku.x ** -1,
        #     self.modelo.Ku.y ** -1,
        #     self.modelo.Ku.z ** -1
        # ])
        return vd