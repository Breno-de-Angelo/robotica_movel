import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from caminho import Caminho
from quadrimotor import Quadrimotor
from controlador_cinematico import ControladorCinematico
from controlador_dinamico import ControladorDinamico
from controlador_laco_interno_externo import ControladorLacoInternoLacoExterno


if __name__ == '__main__':

    def caminho_generator(s: int) -> np.ndarray:
        xpc = np.cos(2 * np.pi * s / 25)
        ypc = np.sin(2 * np.pi * s / 25)
        zpc = 1.5

        dxpc = -2 * np.pi / 25 * np.sin(2 * np.pi * s / 25)
        dypc = 2 * np.pi / 25 * np.cos(2 * np.pi * s / 25)
        dzpc = 0

        betat = np.arctan2(dypc, dxpc)
        vxy = np.sqrt(dxpc**2 + dypc**2)
        tetat = np.arctan2(dzpc, vxy)

        return np.array([xpc, ypc, zpc, betat, tetat])

    caminho = Caminho(
        caminho=caminho_generator,
        s_min=0,
        s_max=60,
        N=500,
        delta=0.025,
        Vd=0.20,
    )

    quadrimotor = Quadrimotor(
        Ku=np.array([0.8417, 0.8354, 3.966]), # phi -> 9,8524
        Kv=np.array([0.18227, 0.17095, 4.001]), # phi -> 4,7295
        # pose=Pose(2.0, 1.0, 0.5, 0.0, 0.0, np.deg2rad(45.0)),
        pose=np.array([1.0, 0.0, 1.5, 0.0, 0.0, np.deg2rad(45.0)]),
        velocity=(0.0, 0.0, 0.0),
    )

    controlador_cinematico = ControladorCinematico(
        Kp=np.array([1.0, 1.0, 1.0]),
        Ls=np.array([1.0, 1.0, 1.0]),
        delta=0.025,
        Vd=0.20,
        modelo=quadrimotor,
    )

    controlador_dinamico = ControladorDinamico(
        K=np.array([1.0, 1.0, 1.0]),
        modelo=quadrimotor,
    )

    controlador = ControladorLacoInternoLacoExterno(
        T=0.2,
        controlador_cinemático=controlador_cinematico,
        controlador_dinâmico=controlador_dinamico,
        caminho=caminho,
        quadrimotor=quadrimotor,
    )

    # Extract X, Y, Z coordinates from target_path
    target_x = [pose.x for pose in controlador.caminho]
    target_y = [pose.y for pose in controlador.caminho]
    target_z = [pose.z for pose in controlador.caminho]

    x_vals, y_vals, z_vals = [], [], []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(target_x, target_y, target_z, 'r--', label="Target Path", marker="x")

    while not (finished := controlador.forward()):
        print(f"s: {controlador.s}, Robô: {quadrimotor.pose}")
        pose = quadrimotor.pose

        # Clear and replot
        ax.clear()
        ax.plot(target_x, target_y, target_z, 'r--', label="Target Path", marker="x")
        ax.plot(x_vals, y_vals, z_vals, 'b-', label="Robot Trajectory", marker="o")
        ax.plot([pose.x], [pose.y], [pose.z], 'y-', label="Robot Current Location", marker="o")

        x_vals.append(pose.x)
        y_vals.append(pose.y)
        z_vals.append(pose.z)

        # Labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Movement in 3D Space")
        ax.legend()
        
        plt.pause(0.01)  # Pause to update the plot
    
    plt.show
