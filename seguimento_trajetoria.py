import numpy as np
import time
import matplotlib.pyplot as plt


# Four Axis model
Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
Ku_inverse = np.linalg.inv(Ku)
Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])

# Kinematic controller gains
Kp = np.diag([10.0, 10.0, 10.0, 10.0])
Ls=np.array([1.0, 1.0, 1.0, 1.0])

# Dynamic controller gains
K = np.diag([1.0, 1.0, 1.0, 1.0])

current_pose = np.array([1.0, 0.0, 1.5, np.deg2rad(45.0)])
# current_pose = np.array([3.0, -1.0, 0.5, np.deg2rad(5.0)])
world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])

def trajetoria_fn(t: float):
    xd = np.cos(2 * np.pi * t / 25)
    yd = np.sin(2 * np.pi * t / 25)
    zd = 1.5
    psid = np.deg2rad(45)

    dxdt = -2 * np.pi / 25 * np.sin(2 * np.pi * t / 25)
    dydt = 2 * np.pi / 25 * np.cos(2 * np.pi * t / 25)
    dzdt = 0.0
    dpsidt = 0.0

    # d2xdt2 = - (2 * np.pi / 25) ** 2 * np.cos(2 * np.pi * t / 25)
    # d2ydt2 = - (2 * np.pi / 25) ** 2 * np.sin(2 * np.pi * t / 25)
    # d2zdt2 = 0.0
    # d2psidt2 = 0.0

    return np.array([xd, yd, zd, psid]), np.array([dxdt, dydt, dzdt, dpsidt]) #, np.array([d2xdt2, d2ydt2, d2zdt2, d2psidt2])

def direct_kinematics_matrix(psi: float):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

t_values = np.linspace(0, 75, 500)
trajetoria = np.array([trajetoria_fn(t)[0] for t in t_values])
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
xlim, ylim, zlim = None, None, None

body_frame_velocity_kinematic_command = np.array([0.0, 0.0, 0.0, 0.0])

t_start = time.time()
current_time = t_start
while True:
    delta_t = time.time() - current_time
    current_time = time.time()
    t = current_time - t_start  # Total time elapsed
    if t > 75:
        print("End of trajectory")
        break

    desired_pose, desired_velocity = trajetoria_fn(current_time - t_start)
    pose_error = desired_pose - current_pose
    distance = np.linalg.norm(pose_error[:3])
    psi_error = pose_error[-1]
    print(f"Distance: {distance}, Psi error: {psi_error}")

    # Kinematics matrices
    A = direct_kinematics_matrix(current_pose[3])
    A_inverse = np.linalg.inv(A)

    # Kinematic controller
    world_frame_velocity_kinematic_command = desired_velocity + Ls * np.tanh(Kp @ pose_error)

    # Inverse kinematics
    new_body_frame_velocity_kinematic_command = A_inverse @ world_frame_velocity_kinematic_command
    derivative_body_frame_velocity_kinematic_command = (new_body_frame_velocity_kinematic_command - body_frame_velocity_kinematic_command) / delta_t
    body_frame_velocity_kinematic_command = new_body_frame_velocity_kinematic_command

    body_frame_current_velocity = A_inverse @ world_frame_current_velocity

    # Dynamic controller
    body_frame_velocity_dynamic_command = Ku_inverse @ (derivative_body_frame_velocity_kinematic_command + K @ (body_frame_velocity_kinematic_command - body_frame_current_velocity) + Kv @ body_frame_current_velocity)

    ax.clear()
    ax.plot(trajetoria[:, 0], trajetoria[:, 1], trajetoria[:, 2], 'r--', label="Target Path", marker="x")
    ax.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 'y-', label="Robot Goal", marker="o")
    ax.plot([current_pose[0]], [current_pose[1]], [current_pose[2]], 'b-', label="Robot Current Location", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajetória 3D")
    ax.legend()

    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    else:
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()

    plt.pause(0.01)  # Pause to update the plot

    # Robot model
    acceleration = Ku @ body_frame_velocity_dynamic_command - Kv @ body_frame_current_velocity
    world_frame_current_velocity += acceleration * delta_t
    current_pose += world_frame_current_velocity * delta_t

plt.show()