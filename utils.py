import numpy as np
from dataclasses import dataclass


@dataclass
class Vector3d:
    x: float
    y: float
    z: float

    def __add__(self, other: 'Vector3d') -> 'Vector3d':
        return Vector3d(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    
    def __sub__(self, other: 'Vector3d') -> 'Vector3d':
        return Vector3d(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )
    
    def __mul__(self, scalar: float) -> 'Vector3d':
        return Vector3d(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def __truediv__(self, scalar: float) -> 'Vector3d':
        return Vector3d(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
        )


@dataclass
class Pose(Vector3d):
    roll: float
    pitch: float
    yaw: float

    def __add__(self, other: 'Pose') -> 'Pose':
        return Pose(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.roll + other.roll,
            self.pitch + other.pitch,
            self.yaw + other.yaw,
        )
    
    def __sub__(self, other: 'Pose') -> 'Pose':
        return Pose(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.roll - other.roll,
            self.pitch - other.pitch,
            self.yaw - other.yaw,
        )

    def __mul__(self, scalar: float) -> 'Pose':
        return Pose(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.roll * scalar,
            self.pitch * scalar,
            self.yaw * scalar,
        )
    
    def __truediv__(self, scalar: float) -> 'Pose':
        return Pose(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
            self.roll / scalar,
            self.pitch / scalar,
            self.yaw / scalar,
        )


def Vector3d_to_numpyArray(vector: Vector3d) -> np.ndarray:
    return np.array([
        vector.x,
        vector.y,
        vector.z
    ])

def numpyArray_to_Vector3d(array: np.ndarray) -> Vector3d:
    return Vector3d(
        x=array[0],
        y=array[1],
        z=array[2],
    )

def Pose_to_numpyArray(pose: Pose) -> np.ndarray:
    return np.array([
        pose.x,
        pose.y,
        pose.z,
        pose.roll,
        pose.pitch,
        pose.yaw
    ])

def numpyArray_to_Pose(array: np.ndarray) -> Vector3d:
    return Pose(
        x=array[0],
        y=array[1],
        z=array[2],
        roll=array[3],
        pitch=array[4],
        yaw=array[5]
    )