import numpy as np
import time
import matplotlib.pyplot as plt
import rospy
import scipy.io
from geometry_msgs.msg import PoseStamped, TwistStamped

class PathController:
    def __init__(self):
        rospy.init_node('path_controller', anonymous=True)
        
        self.simulation = rospy.get_param('~simulation', True)
        self.epsilon = rospy.get_param('~epsilon', 0.05)
        
        self.rate = rospy.Rate(30)  # 30 Hz
        
        self.Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
        self.Ku_inverse = np.linalg.inv(self.Ku)
        self.Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])
        
        self.Kp = np.diag([10.0, 10.0, 10.0, 10.0])
        self.Ls = np.array([1.0, 1.0, 1.0, 1.0])
        
        self.K = np.diag([1.0, 1.0, 1.0, 1.0])
        
        if self.simulation:
            self.current_pose = np.array([2.0, -1.0, 0.5, np.deg2rad(5.0)])
        else:
            self.current_pose = None
        
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])
        self.last_pose_timestamp = time.time()
        
        if not self.simulation:
            self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback, queue_size=1)
            self.cmd_vel_pub = rospy.Publisher('/cmd_vel', TwistStamped, queue_size=10)
        
        self.path_s_steps = np.linspace(0.0, 75.0, 501)
        self.path_points = [self.path_fn(t) for t in self.path_s_steps]
        self.path_xyzs = np.array([pose[:3] for (pose, vel) in self.path_points])
        self.path_velocities = np.array([vel for (pose, vel) in self.path_points])
        self.current_target_index = 0
        
        self.body_frame_velocity_kinematic_command = np.array([0.0, 0.0, 0.0, 0.0])
        self.t_start = time.time()

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

    def pose_callback(self, msg):
        rospy.loginfo(f"{msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if abs(msg.pose.position.x) > 1.2 or abs(msg.pose.position.y) > 1.2 or not (0.1 <= msg.pose.position.z < 2.5):
            cmd_vel_msg = TwistStamped()
            self.cmd_vel_pub.publish(cmd_vel_msg)
            rospy.loginfo("World exceeded...")
            self.shutdown_module()
        self.last_pose_timestamp = time.time()

    def path_fn(self, s):
        xd = np.cos(2 * np.pi * s / 25)
        yd = np.sin(2 * np.pi * s / 25)
        zd = 1.5
        psid = np.deg2rad(45)
        dxdt = -2 * np.pi / 25 * np.sin(2 * np.pi * s / 25)
        dydt = 2 * np.pi / 25 * np.cos(2 * np.pi * s / 25)
        return np.array([xd, yd, zd, psid]), np.array([dxdt, dydt, 0.0, 0.0])

    def control_loop(self):
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.loginfo("Current pose not available yet")
                continue
            
            if not self.simulation and time.time() - self.last_pose_timestamp > 2.0:
                rospy.logerr("2 seconds without odom")
                continue
            
            t = time.time() - self.t_start
            current_xyz = self.current_pose[:3]
            distances = np.linalg.norm(self.path_xyzs - current_xyz, axis=1)
            closest_idx = np.argmin(distances)
            min_dist = distances[closest_idx]
            
            if min_dist > self.epsilon:
                self.current_target_index = closest_idx
            else:
                self.current_target_index += 1
                if self.current_target_index >= len(self.path_points):
                    rospy.loginfo("Reached end of path. Exiting...")
                    self.shutdown_module()
            
            rospy.loginfo(f"index: {self.current_target_index}, min_dist: {min_dist}")
            desired_pose, desired_velocity = self.path_points[self.current_target_index]
            pose_error = desired_pose - self.current_pose

            self.error_data.append(pose_error)
            self.time_data.append(t)
            self.robot_pose_data.append(self.current_pose)
            self.desired_pose_data.append(desired_pose)
            
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
            self.velocity_command_data.append(body_frame_velocity_dynamic_command)
            
            # Update path plot
            self.ax1.clear()
            t_values = np.linspace(0, 75, 500)
            path = np.array([self.path_fn(t)[0] for t in t_values])
            self.ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'r--', label="Target Path")
            self.ax1.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 'yo', label="Robot Goal")
            self.ax1.plot([self.current_pose[0]], [self.current_pose[1]], [self.current_pose[2]], 'bo', label="Robot Current Location")
            self.ax1.set_xlabel("X")
            self.ax1.set_ylabel("Y")
            self.ax1.set_zlabel("Z")
            self.ax1.set_title("Caminho 3D")
            if self.ax1_xlim is None:
                self.ax1_xlim, self.ax1_ylim, self.ax1_zlim = self.ax1.get_xlim(), self.ax1.get_ylim(), self.ax1.get_ylim()
            else:
                self.ax1.set_xlim(self.ax1_xlim)
                self.ax1.set_ylim(self.ax1_ylim)
                self.ax1.set_zlim(self.ax1_zlim)
            self.ax1.legend()

            # Update error plot
            self.ax2.clear()
            self.ax2_psi.cla()
            self.ax3.clear()
            errors = np.array(self.error_data)
            self.ax2.plot(self.time_data, errors[:, 0], label="X Error")
            self.ax2.plot(self.time_data, errors[:, 1], label="Y Error")
            self.ax2.plot(self.time_data, errors[:, 2], label="Z Error")
            self.ax2_psi.plot(self.time_data, np.rad2deg(errors[:, 3]), label="Psi Error (deg)", color='purple', linestyle='dashed')
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Error")
            self.ax2.set_title("Pose Error Over Time")
            self.ax2.legend(loc="upper left")
            self.ax2_psi.legend(loc="upper right")

            x_cmd, y_cmd, z_cmd, _ = zip(*self.velocity_command_data)  # Ignorando a quarta dimensão
            self.ax3.plot(self.time_data, x_cmd, label="X command")
            self.ax3.plot(self.time_data, y_cmd, label="Y command")
            self.ax3.plot(self.time_data, z_cmd, label="Z command")
            self.ax3.legend(loc="upper left")
            plt.pause(0.01)

            # Update robot model only in simulation
            if self.simulation:
                # Robot model update
                acceleration = self.Ku @ body_frame_velocity_dynamic_command - self.Kv @ self.world_frame_current_velocity
                self.world_frame_current_velocity += acceleration / 30.0
                self.current_pose += self.world_frame_current_velocity / 30.0
            else:
                # Publish cmd_vel as TwistStamped
                cmd_vel_msg = TwistStamped()
                cmd_vel_msg.header.stamp = self.get_clock().now().to_msg()
                cmd_vel_msg.header.frame_id = 'base_link'
                cmd_vel_msg.twist.linear.x = body_frame_velocity_dynamic_command[0]
                cmd_vel_msg.twist.linear.y = body_frame_velocity_dynamic_command[1]
                cmd_vel_msg.twist.linear.z = body_frame_velocity_dynamic_command[2]
                cmd_vel_msg.twist.angular.z = body_frame_velocity_dynamic_command[3]
                self.cmd_vel_pub.publish(cmd_vel_msg)
            
            self.rate.sleep()

    def direct_kinematics_matrix(self, psi):
        return np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def shutdown_module(self):
        rospy.signal_shutdown("Shutting down")
        plt.close('all')

if __name__ == '__main__':
    controller = PathController()
    controller.control_loop()
