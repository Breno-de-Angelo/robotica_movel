import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
import time
import matplotlib.pyplot as plt
import scipy.io

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')

        # Kinematic controller gains
        self.Kp = 0.4 * np.diag([1.0, 1.0, 1.0, 1.0])
        self.Ls = 1.0 * np.array([1.0, 1.0, 1.0, 1.0])

        # Dynamic controller gains
        self.K = 0.0 * np.diag([1.0, 1.0, 1.0, 1.0])

        # Quadcopter model
        self.Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
        self.Ku_inverse = np.linalg.inv(self.Ku)
        self.Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])

        # Initialize state variables
        self.current_pose = None
        self.body_frame_velocity_kinematic_command = np.array([0.0, 0.0, 0.0, 0.0])
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])
        self.t_start = time.time()

        # Subscriber for current pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.pose_callback,
            10
        )

        # Publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(1.0 / 30.0, self.control_loop)

        # Data to be saved
        self.error_data = []
        self.time_data = []
        self.robot_pose_data = []
        self.desired_pose_data = []
        self.velocity_command_data = []

        # Initialize plots
        self.fig = plt.figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(311, projection='3d')
        self.ax1_xlim, self.ax1_ylim, self.ax1_zlim = None, None, None
        self.ax2 = self.fig.add_subplot(312)
        self.ax2_psi = self.ax2.twinx()
        self.ax3 = self.fig.add_subplot(313)

        # Precompute trajectory points for plotting
        self.trajectory_times = np.linspace(0.0, 75.0, 501)
        self.trajectory_points = np.array([self.trajectory_fn(t)[0] for t in self.trajectory_times])

    def pose_callback(self, msg):
        # Extract current pose from PoseStamped
        self.get_logger().info(f"{msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if abs(msg.pose.position.x) > 20.0 or \
            abs(msg.pose.position.y) > 20.0 or \
            not (0.1 <= msg.pose.position.z < 2.5):
            cmd_vel_msg = TwistStamped()
            cmd_vel_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_vel_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(cmd_vel_msg)
            self.get_logger().info("World exceeded...")
            shutdown_module()

        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)
        ])

    def trajectory_fn(self, t: float):
        # xd = np.cos(2 * np.pi * t / 25)
        # yd = np.sin(2 * np.pi * t / 25)
        # zd = 1.5
        # psid = np.deg2rad(45)
        # dxdt = -2 * np.pi / 25 * np.sin(2 * np.pi * t / 25)
        # dydt = 2 * np.pi / 25 * np.cos(2 * np.pi * t / 25)
        # dzdt = 0.0
        # dpsidt = 0.0
        
        xd = 0.0
        yd = 0.0
        zd = 0.5
        psid = 0.0
        dxdt = 0.0
        dydt = 0.0
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
        if self.current_pose is None:
            self.get_logger().warning("Pose not received yet")
            return

        current_time = time.time()
        t = current_time - self.t_start
        if t > 75:
            cmd_vel_msg = TwistStamped()
            cmd_vel_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_vel_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(cmd_vel_msg)
            self.save_results()
            self.get_logger().info("End of trajectory. Exiting...")
            shutdown_module()
            return

        desired_pose, desired_velocity = self.trajectory_fn(t)
        pose_error = desired_pose - self.current_pose
        self.error_data.append(pose_error)
        self.time_data.append(t)
        self.robot_pose_data.append(self.current_pose)
        self.desired_pose_data.append(desired_pose)

        # Kinematic controller
        A = self.direct_kinematics_matrix(self.current_pose[3])
        A_inverse = np.linalg.inv(A)
        world_frame_velocity_kinematic_command = desired_velocity + self.Ls * np.tanh(self.Kp @ pose_error)
        new_body_frame_velocity_kinematic_command = A_inverse @ world_frame_velocity_kinematic_command

        # Uncomment these lines to ignore the kinematic controller and send a constant velocity to the dynamic controller
        # world_frame_velocity_kinematic_command = np.array([0.5, 0.0, 0.0, 0.0])
        # new_body_frame_velocity_kinematic_command = A_inverse @ world_frame_velocity_kinematic_command

        # Dynamic controller
        derivative_body_frame_velocity_kinematic_command = (new_body_frame_velocity_kinematic_command - self.body_frame_velocity_kinematic_command) * 30.0
        self.body_frame_velocity_kinematic_command = new_body_frame_velocity_kinematic_command
        body_frame_current_velocity = A_inverse @ self.world_frame_current_velocity
        body_frame_velocity_dynamic_command = self.Ku_inverse @ (
            derivative_body_frame_velocity_kinematic_command + self.K @ (self.body_frame_velocity_kinematic_command - body_frame_current_velocity) + self.Kv @ body_frame_current_velocity
        )
        
        # Uncomment this line to use only kinematic controller
        # body_frame_velocity_dynamic_command = new_body_frame_velocity_kinematic_command

        self.velocity_command_data.append(body_frame_velocity_dynamic_command)

        # Publish cmd_vel as TwistStamped
        cmd_vel_msg = TwistStamped()
        cmd_vel_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_vel_msg.header.frame_id = 'base_link'
        cmd_vel_msg.twist.linear.x = body_frame_velocity_dynamic_command[0]
        cmd_vel_msg.twist.linear.y = body_frame_velocity_dynamic_command[1]
        cmd_vel_msg.twist.linear.z = body_frame_velocity_dynamic_command[2]
        cmd_vel_msg.twist.angular.z = body_frame_velocity_dynamic_command[3]
        self.cmd_vel_pub.publish(cmd_vel_msg)

        # Update plots
        self.update_plots(desired_pose)

    def update_plots(self, desired_pose):
        # Update trajectory plot
        self.ax1.clear()
        self.ax1.plot(self.trajectory_points[:, 0], self.trajectory_points[:, 1], self.trajectory_points[:, 2], 'r--', label="Target Trajectory")
        self.ax1.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 'yo', label="Desired Position")
        self.ax1.plot([self.current_pose[0]], [self.current_pose[1]], [self.current_pose[2]], 'bo', label="Robot Position")
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")
        self.ax1.set_zlabel("Z")
        self.ax1.set_title("3D Trajectory Tracking")
        if self.ax1_xlim is None:
            self.ax1_xlim = self.ax1.get_xlim()
            self.ax1_ylim = self.ax1.get_ylim()
            self.ax1_zlim = self.ax1.get_zlim()
        else:
            self.ax1.set_xlim(self.ax1_xlim)
            self.ax1.set_ylim(self.ax1_ylim)
            self.ax1.set_zlim(self.ax1_zlim)
        self.ax1.legend()

        # Update error plot
        self.ax2.clear()
        self.ax2_psi.cla()
        self.ax3.clear()
        if self.time_data:
            errors = np.array(self.error_data)
            self.ax2.plot(self.time_data, errors[:, 0], label="X Error")
            self.ax2.plot(self.time_data, errors[:, 1], label="Y Error")
            self.ax2.plot(self.time_data, errors[:, 2], label="Z Error")
            self.ax2_psi.plot(self.time_data, np.rad2deg(errors[:, 3]), color='purple', linestyle='dashed', label="Psi Error (deg)")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Error (m)")
            self.ax2_psi.set_ylabel("Psi Error (deg)")
            self.ax2.set_title("Tracking Errors Over Time")
            self.ax2.legend(loc="upper left")
            self.ax2_psi.legend(loc="upper right")

            x_cmd, y_cmd, z_cmd, _ = zip(*self.velocity_command_data)  # Ignorando a quarta dimensão
            self.ax3.plot(self.time_data, x_cmd, label="X command")
            self.ax3.plot(self.time_data, y_cmd, label="Y command")
            self.ax3.plot(self.time_data, z_cmd, label="Z command")
            self.ax3.legend(loc="upper left")
        plt.pause(0.01)

    def save_results(self):
        if self.error_data and self.time_data:
            data = {
                'time': np.array(self.time_data),
                'error': np.array(self.error_data),
                'robot_pose': np.array(self.robot_pose_data),
                'desired_pose': np.array(self.desired_pose_data),
                'velocity_command': np.array(self.velocity_command_data)
            }
            scipy.io.savemat('simulation_results.mat', data)
            self.get_logger().info("Results saved at 'simulation_results.mat'.")

def shutdown_module():
    rclpy.shutdown()
    plt.close('all')
    exit(0)

def main(args=None):
    rclpy.init(args=args)
    controller = TrajectoryController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
