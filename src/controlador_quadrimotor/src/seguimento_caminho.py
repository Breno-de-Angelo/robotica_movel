#!/usr/bin/env python
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
import scipy.io
from geometry_msgs.msg import PoseStamped, TwistStamped

class PathController:
    def __init__(self):
        rospy.init_node('path_controller')
        
        # Parameters
        self.epsilon = rospy.get_param('~epsilon', 0.05)

        # Kinematic controller gains
        self.Kp = np.diag([10.0, 10.0, 10.0, 10.0])
        self.Ls = np.array([1.0, 1.0, 1.0, 1.0])

        # Initialize state variables
        self.current_pose = np.array([2.0, -1.0, 0.5, np.deg2rad(5.0)])
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])
        self.body_frame_velocity_kinematic_command = np.array([0.0, 0.0, 0.0, 0.0])

        # Subscriber for current pose
        self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback, queue_size=1)

        # Publisher for cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', TwistStamped, queue_size=10)

        # Precompute the path
        self.path_s_steps = np.linspace(0.0, 75.0, 501)
        self.path_points = [self.path_fn(t) for t in self.path_s_steps]
        self.path_xyzs = np.array([pose[:3] for (pose, vel) in self.path_points])
        self.path_velocities = np.array([vel for (pose, vel) in self.path_points])
        self.current_target_index = 0

        # Data to be saved
        self.error_data = []
        self.time_data = []
        self.robot_pose_data = []
        self.desired_pose_data = []
        self.velocity_command_data = []
        self.t_start = time.time()

        # Initialize plots
        self.fig = plt.figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(211, projection='3d')
        self.ax1_xlim, self.ax1_ylim, self.ax1_zlim = None, None, None
        self.ax2 = self.fig.add_subplot(212)
        self.ax2_psi = self.ax2.twinx()

        # Control loop
        self.rate = rospy.Rate(30)
        self.control_loop()

    def pose_callback(self, msg):
        rospy.loginfo(f"{msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if (abs(msg.pose.position.x) > 1.2 or 
            abs(msg.pose.position.y) > 1.2 or 
            not (0.0 <= msg.pose.position.z <= 2.5)):
            cmd_vel_msg = TwistStamped()
            cmd_vel_msg.header.stamp = rospy.Time.now()
            cmd_vel_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(cmd_vel_msg)
            rospy.loginfo("World exceeded...")
            self.shutdown_module()

        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)
        ])

    def path_fn(self, s: float):
        xd = np.cos(2 * np.pi * s / 25)
        yd = np.sin(2 * np.pi * s / 25)
        zd = 1.5
        psid = np.deg2rad(45)
        
        dxds = -2 * np.pi / 25 * np.sin(2 * np.pi * s / 25)
        dyds = 2 * np.pi / 25 * np.cos(2 * np.pi * s / 25)
        dsdt = 1.0
        return np.array([xd, yd, zd, psid]), np.array([dxds*dsdt, dyds*dsdt, 0.0, 0.0])

    def direct_kinematics_matrix(self, psi: float):
        return np.array([
            [np.cos(psi), -np.sin(psi), 0, 0],
            [np.sin(psi), np.cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def control_loop(self):
        while not rospy.is_shutdown():
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
                    cmd_vel_msg = TwistStamped()
                    cmd_vel_msg.header.stamp = rospy.Time.now()
                    cmd_vel_msg.header.frame_id = 'base_link'
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    self.save_results()
                    rospy.loginfo("Reached end of path. Exiting...")
                    self.shutdown_module()
                    return

            rospy.loginfo(f"index: {self.current_target_index}, min_dist: {min_dist}")
            desired_pose, desired_velocity = self.path_points[self.current_target_index]

            if min_dist > self.epsilon:
                desired_velocity = np.array([0.0, 0.0, 0.0, 0.0])

            pose_error = desired_pose - self.current_pose
            self.error_data.append(pose_error)
            self.time_data.append(t)
            self.robot_pose_data.append(self.current_pose)
            self.desired_pose_data.append(desired_pose)

            A = self.direct_kinematics_matrix(self.current_pose[3])
            A_inverse = np.linalg.inv(A)
            world_frame_velocity_kinematic_command = desired_velocity + self.Ls * np.tanh(self.Kp @ pose_error)
            new_body_frame_velocity_kinematic_command = A_inverse @ world_frame_velocity_kinematic_command

            cmd_vel_msg = TwistStamped()
            cmd_vel_msg.header.stamp = rospy.Time.now()
            cmd_vel_msg.header.frame_id = 'base_link'
            cmd_vel_msg.twist.linear.x = new_body_frame_velocity_kinematic_command[0]
            cmd_vel_msg.twist.linear.y = new_body_frame_velocity_kinematic_command[1]
            cmd_vel_msg.twist.linear.z = new_body_frame_velocity_kinematic_command[2]
            cmd_vel_msg.twist.angular.z = new_body_frame_velocity_kinematic_command[3]
            self.cmd_vel_pub.publish(cmd_vel_msg)

            self.update_plots(desired_pose)
            self.rate.sleep()

    def update_plots(self, desired_pose):
        self.ax1.clear()
        trajetoria = np.array([pose for (pose, vel) in self.path_points])
        self.ax1.plot(trajetoria[:,0], trajetoria[:,1], trajetoria[:,2], 'r--', label="Target Path")
        self.ax1.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 'yo', label="Robot Goal")
        self.ax1.plot([self.current_pose[0]], [self.current_pose[1]], [self.current_pose[2]], 'bo', label="Current Position")
        self.ax1.set_xlabel("X"); self.ax1.set_ylabel("Y"); self.ax1.set_zlabel("Z")
        self.ax1.set_title("3D Trajectory")
        
        self.ax2.clear()
        self.ax2_psi.cla()
        if self.time_data:
            errors = np.array(self.error_data)
            self.ax2.plot(self.time_data, errors[:,0], label="X Error")
            self.ax2.plot(self.time_data, errors[:,1], label="Y Error")
            self.ax2.plot(self.time_data, errors[:,2], label="Z Error")
            self.ax2_psi.plot(self.time_data, np.rad2deg(errors[:,3]), 'purple', linestyle='dashed', label="Psi Error (deg)")
            self.ax2.set_xlabel("Time (s)"); self.ax2.set_ylabel("Error (m)")
            self.ax2_psi.set_ylabel("Psi Error (deg)")
            self.ax2.legend(loc="upper left"); self.ax2_psi.legend(loc="upper right")

        plt.pause(0.01)

    def save_results(self):
        if self.error_data:
            data = {
                'time': np.array(self.time_data),
                'error': np.array(self.error_data),
                'robot_pose': np.array(self.robot_pose_data),
                'desired_pose': np.array(self.desired_pose_data)
            }
            scipy.io.savemat('simulation_results.mat', data)
            rospy.loginfo("Results saved to simulation_results.mat")

    def shutdown_module(self):
        plt.close('all')
        rospy.signal_shutdown("End of path")

if __name__ == '__main__':
    try:
        PathController()
    except rospy.ROSInterruptException:
        pass