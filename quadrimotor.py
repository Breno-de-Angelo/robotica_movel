import numpy as np


class Quadrimotor:
    """
    \\ddot{x} = ku U - kv \\dot{x}
    """
    pose: np.ndarray
    velocity: np.ndarray
    Ku: np.ndarray
    Kv: np.ndarray

    def __init__(
        self,
        Ku: np.ndarray,
        Kv: np.ndarray,
        pose: np.ndarray,
        velocity: np.ndarray,
    ):
        self.Ku = Ku
        self.Kv = Kv
        self.pose = pose
        self.velocity = velocity

    def cinematica_direta(self) -> np.ndarray:
        yaw = self.pose[3]
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def cinemática_inversa(self) -> np.ndarray:
        return np.linalg.inv(self.cinematica_direta())
    
    def update(self, U: np.ndarray, delta_t: float):
        self.velocity = (self.Ku * U - self.Kv * self.velocity) * delta_t
        self.pose = self.pose + self.velocity * delta_t
