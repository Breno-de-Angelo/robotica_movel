import numpy as np
import time
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class PathController(Node):
    def __init__(self):
        super().__init__('path_controller')
        
        # Declare parameters
        self.declare_parameter('simulation', True)
        self.simulation = self.get_parameter('simulation').get_parameter_value().bool_value
        self.declare_parameter('epsilon', 0.2)
        self.epsilon = self.get_parameter('epsilon').get_parameter_value().double_value

        self.timer = self.create_timer(1.0 / 30.0, self.control_loop)  # 30 Hz timer
        
        # Four Axis model
        self.Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
        self.Ku_inverse = np.linalg.inv(self.Ku)
        self.Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])

        # Kinematic controller gains
        self.Kp = np.diag([10.0, 10.0, 10.0, 10.0])
        self.Ls = np.array([1.0, 1.0, 1.0, 1.0])

        # Dynamic controller gains
        self.K = np.diag([1.0, 1.0, 1.0, 1.0])

        # Initialize state variables
        self.current_pose = np.array([1.0, 0.0, 1.5, np.deg2rad(45.0)])
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Setup odometry subscription if not in simulation
        if not self.simulation:
            self.subscription = self.create_subscription(
                Odometry,
                'robot_pose',
                self.odom_callback,
                10
            )
            self.get_logger().info("Using real robot odometry")

        # Precompute the path
        self.path_time_steps = np.arange(0.0, 75.0, 0.1)  # Time steps at 0.1s intervals
        self.path_points = [self.path_fn(t) for t in self.path_time_steps]
        self.path_xyzs = np.array([pose[:3] for (pose, vel) in self.path_points])
        self.path_velocities = np.array([vel for (pose, vel) in self.path_points])

        self.body_frame_velocity_kinematic_command = np.array([0.0, 0.0, 0.0, 0.0])
        self.t_start = time.time()
        self.error_data = []
        self.time_data = []

        # Initialize plots
        self.fig = plt.figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(211, projection='3d')
        self.ax2 = self.fig.add_subplot(212)

    def odom_callback(self, msg):
        # Placeholder for real odometry processing
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0])
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])

    def path_fn(self, t: float):
        xd = np.cos(2 * np.pi * t / 25)
        yd = np.sin(2 * np.pi * t / 25)
        zd = 1.5
        psid = np.deg2rad(45)
        dxdt = -2 * np.pi / 25 * np.sin(2 * np.pi * t / 25)
        dydt = 2 * np.pi / 25 * np.cos(2 * np.pi * t / 25)
        dzdt = 0.0
        dpsidt = 0.0
        return np.array([xd, yd, zd, psid]), np.array([dxdt, dydt, dzdt, dpsidt])

    def direct_kinematics_matrix(self, psi: float):
        return np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def control_loop(self):
        current_time = time.time()
        t = current_time - self.t_start
        if t > 75:
            self.get_logger().info("End of trajectory")
            return

        # Find closest point in the path
        current_xyz = self.current_pose[:3]
        distances = np.linalg.norm(self.path_xyzs - current_xyz, axis=1)
        closest_idx = np.argmin(distances)
        min_dist = distances[closest_idx]

        # Determine target index
        if min_dist > self.epsilon:
            target_idx = closest_idx
        else:
            target_idx = closest_idx + 1
            if target_idx >= len(self.path_points):
                target_idx = len(self.path_points) - 1

        desired_pose, desired_velocity = self.path_points[target_idx]

        pose_error = desired_pose - self.current_pose
        self.error_data.append(pose_error)
        self.time_data.append(t)

        A = self.direct_kinematics_matrix(self.current_pose[3])
        A_inverse = np.linalg.inv(A)
        world_frame_velocity_kinematic_command = desired_velocity + self.Ls * np.tanh(self.Kp @ pose_error)
        new_body_frame_velocity_kinematic_command = A_inverse @ world_frame_velocity_kinematic_command
        derivative_body_frame_velocity_kinematic_command = (new_body_frame_velocity_kinematic_command - self.body_frame_velocity_kinematic_command) * 30.0
        self.body_frame_velocity_kinematic_command = new_body_frame_velocity_kinematic_command
        body_frame_current_velocity = A_inverse @ self.world_frame_current_velocity

        body_frame_velocity_dynamic_command = self.Ku_inverse @ (
            derivative_body_frame_velocity_kinematic_command + self.K @ (self.body_frame_velocity_kinematic_command - body_frame_current_velocity) + self.Kv @ body_frame_current_velocity
        )

        # Update trajectory plot
        self.ax1.clear()
        trajetoria = np.array([pose for (pose, vel) in self.path_points])
        self.ax1.plot(trajetoria[:, 0], trajetoria[:, 1], trajetoria[:, 2], 'r--', label="Target Path")
        self.ax1.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 'yo', label="Robot Goal")
        self.ax1.plot([self.current_pose[0]], [self.current_pose[1]], [self.current_pose[2]], 'bo', label="Robot Current Location")
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")
        self.ax1.set_zlabel("Z")
        self.ax1.set_title("3D Trajectory")
        self.ax1.legend()

        # Update error plot
        self.ax2.clear()
        if self.error_data:
            errors = np.array(self.error_data)
            self.ax2.plot(self.time_data, errors[:, 0], label="X Error")
            self.ax2.plot(self.time_data, errors[:, 1], label="Y Error")
            self.ax2.plot(self.time_data, errors[:, 2], label="Z Error")
            self.ax2.plot(self.time_data, np.rad2deg(errors[:, 3]), label="Psi Error (deg)")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Error")
        self.ax2.set_title("Pose Error Over Time")
        self.ax2.legend()

        plt.pause(0.01)

        # Update robot model only in simulation
        if self.simulation:
            # Robot model update
            acceleration = self.Ku @ body_frame_velocity_dynamic_command - self.Kv @ self.world_frame_current_velocity
            self.world_frame_current_velocity += acceleration / 30.0
            self.current_pose += self.world_frame_current_velocity / 30.0


def main(args=None):
    rclpy.init(args=args)
    controller = PathController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
    plt.show()

if __name__ == '__main__':
    main()