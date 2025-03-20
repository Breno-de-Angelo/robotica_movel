import numpy as np
import time
import matplotlib.pyplot as plt

# Four Axis model
Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
Ku_inverse = np.linalg.inv(Ku)
Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])

# Kinematic controller gains
Kp = np.diag([10.0, 10.0, 10.0, 10.0])
Ls = np.array([1.0, 1.0, 1.0, 1.0])

# Dynamic controller gains
K = np.diag([1.0, 1.0, 1.0, 1.0])

current_pose = np.array([1.0, 0.0, 1.5, np.deg2rad(45.0)])
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

    return np.array([xd, yd, zd, psid]), np.array([dxdt, dydt, dzdt, dpsidt])

def direct_kinematics_matrix(psi: float):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Initialize plots
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212)

# Time and error storage
error_data = []
time_data = []

body_frame_velocity_kinematic_command = np.array([0.0, 0.0, 0.0, 0.0])
t_start = time.time()
current_time = t_start

while True:
    delta_t = time.time() - current_time
    current_time = time.time()
    t = current_time - t_start
    if t > 75:
        print("End of trajectory")
        break

    desired_pose, desired_velocity = trajetoria_fn(t)
    pose_error = desired_pose - current_pose
    
    # Store error data for plotting
    error_data.append(pose_error)
    time_data.append(t)
    
    # Kinematics matrices
    A = direct_kinematics_matrix(current_pose[3])
    A_inverse = np.linalg.inv(A)

    # Kinematic controller
    world_frame_velocity_kinematic_command = desired_velocity + Ls * np.tanh(Kp @ pose_error)
    new_body_frame_velocity_kinematic_command = A_inverse @ world_frame_velocity_kinematic_command
    derivative_body_frame_velocity_kinematic_command = (new_body_frame_velocity_kinematic_command - body_frame_velocity_kinematic_command) / delta_t
    body_frame_velocity_kinematic_command = new_body_frame_velocity_kinematic_command
    body_frame_current_velocity = A_inverse @ world_frame_current_velocity

    # Dynamic controller
    body_frame_velocity_dynamic_command = Ku_inverse @ (
        derivative_body_frame_velocity_kinematic_command + K @ (body_frame_velocity_kinematic_command - body_frame_current_velocity) + Kv @ body_frame_current_velocity
    )

    # Update trajectory plot
    ax1.clear()
    t_values = np.linspace(0, 75, 500)
    trajetoria = np.array([trajetoria_fn(t)[0] for t in t_values])
    ax1.plot(trajetoria[:, 0], trajetoria[:, 1], trajetoria[:, 2], 'r--', label="Target Path")
    ax1.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 'yo', label="Robot Goal")
    ax1.plot([current_pose[0]], [current_pose[1]], [current_pose[2]], 'bo', label="Robot Current Location")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Trajetória 3D")
    ax1.legend()

    # Update error plot
    ax2.clear()
    errors = np.array(error_data)
    ax2.plot(time_data, errors[:, 0], label="X Error")
    ax2.plot(time_data, errors[:, 1], label="Y Error")
    ax2.plot(time_data, errors[:, 2], label="Z Error")
    ax2.plot(time_data, np.rad2deg(errors[:, 3]), label="Psi Error (deg)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error")
    ax2.set_title("Pose Error Over Time")
    ax2.legend()

    plt.pause(0.01)

    # Robot model
    acceleration = Ku @ body_frame_velocity_dynamic_command - Kv @ body_frame_current_velocity
    world_frame_current_velocity += acceleration * delta_t
    current_pose += world_frame_current_velocity * delta_t

plt.show()
