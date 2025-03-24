#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
import time
import matplotlib.pyplot as plt
import scipy.io

class TrajectoryController:
    def __init__(self):
        rospy.init_node('trajectory_controller')

        self.Kp = np.diag([10.0, 10.0, 10.0, 10.0])
        self.Ls = np.array([1.0, 1.0, 1.0, 1.0])

        self.current_pose = np.array([2.0, -1.0, 0.5, np.deg2rad(5.0)])
        self.t_start = time.time()

        self.pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', TwistStamped, queue_size=10)

        self.error_data = []
        self.time_data = []
        self.robot_pose_data = []
        self.desired_pose_data = []
        self.velocity_command_data = []

        self.fig = plt.figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(211, projection='3d')
        self.ax2 = self.fig.add_subplot(212)
        self.ax2_psi = self.ax2.twinx()

        self.trajectory_times = np.linspace(0.0, 75.0, 501)
        self.trajectory_points = np.array([self.trajectory_fn(t)[0] for t in self.trajectory_times])

        self.rate = rospy.Rate(30)
        self.control_loop()

    def pose_callback(self, msg):
        rospy.loginfo(f"{msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if (abs(msg.pose.position.x) > 1.2 or 
            abs(msg.pose.position.y) > 1.2 or 
            not (0.0 <= msg.pose.position.z <= 2.5)):
            cmd_vel_msg = TwistStamped()
            cmd_vel_msg.header.stamp = rospy.Time.now()
            self.cmd_vel_pub.publish(cmd_vel_msg)
            rospy.loginfo("Safety limits exceeded!")
            self.shutdown_module()

        self.current_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)
        ])

    def trajectory_fn(self, t: float):
        xd = np.cos(2 * np.pi * t / 25)
        yd = np.sin(2 * np.pi * t / 25)
        psid = np.deg2rad(45)
        dxdt = -2 * np.pi / 25 * np.sin(2 * np.pi * t / 25)
        dydt = 2 * np.pi / 25 * np.cos(2 * np.pi * t / 25)
        return np.array([xd, yd, 1.5, psid]), np.array([dxdt, dydt, 0.0, 0.0])

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
            if t > 75:
                self.save_results()
                rospy.loginfo("Trajectory completed")
                self.shutdown_module()
                return

            desired_pose, desired_velocity = self.trajectory_fn(t)
            pose_error = desired_pose - self.current_pose

            A = self.direct_kinematics_matrix(self.current_pose[3])
            A_inv = np.linalg.inv(A)
            world_vel = desired_velocity + self.Ls * np.tanh(self.Kp @ pose_error)
            body_vel = A_inv @ world_vel

            cmd_vel = TwistStamped()
            cmd_vel.header.stamp = rospy.Time.now()
            cmd_vel.twist.linear.x = body_vel[0]
            cmd_vel.twist.linear.y = body_vel[1]
            cmd_vel.twist.linear.z = body_vel[2]
            cmd_vel.twist.angular.z = body_vel[3]
            self.cmd_vel_pub.publish(cmd_vel)

            # Update data and plots
            self.error_data.append(pose_error)
            self.time_data.append(t)
            self.update_plots(desired_pose)
            self.rate.sleep()

    def update_plots(self, desired_pose):
        self.ax1.clear()
        self.ax1.plot(self.trajectory_points[:,0], self.trajectory_points[:,1], 
                     self.trajectory_points[:,2], 'r--', label="Reference")
        self.ax1.plot([desired_pose[0]], [desired_pose[1]], [desired_pose[2]], 
                     'yo', markersize=8, label="Target")
        self.ax1.plot([self.current_pose[0]], [self.current_pose[1]], [self.current_pose[2]], 
                     'bo', markersize=8, label="Actual")
        self.ax1.set_xlabel("X"); self.ax1.set_ylabel("Y"); self.ax1.set_zlabel("Z")
        self.ax1.legend()

        self.ax2.clear()
        self.ax2_psi.cla()
        errors = np.array(self.error_data)
        self.ax2.plot(self.time_data, errors[:,0], 'b', label="X Error")
        self.ax2.plot(self.time_data, errors[:,1], 'g', label="Y Error")
        self.ax2.plot(self.time_data, errors[:,2], 'r', label="Z Error")
        self.ax2_psi.plot(self.time_data, np.rad2deg(errors[:,3]), 'm--', label="Yaw Error")
        self.ax2.set_xlabel("Time (s)"); self.ax2.set_ylabel("Position Error (m)")
        self.ax2_psi.set_ylabel("Yaw Error (deg)")
        self.ax2.legend(loc="upper left"); self.ax2_psi.legend(loc="upper right")

        plt.pause(0.01)

    def save_results(self):
        data = {
            'time': np.array(self.time_data),
            'error': np.array(self.error_data),
            'robot_pose': np.array(self.robot_pose_data),
            'desired_pose': np.array(self.desired_pose_data)
        }
        scipy.io.savemat('trajectory_results.mat', data)
        rospy.loginfo("Data saved to trajectory_results.mat")

    def shutdown_module(self):
        plt.close('all')
        rospy.signal_shutdown("Normal shutdown")

if __name__ == '__main__':
    try:
        TrajectoryController()
    except rospy.ROSInterruptException:
        pass