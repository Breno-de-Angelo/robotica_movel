import numpy as np
import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped

class RobotSimulator(Node):
    def __init__(self):
        super().__init__('robot_simulator')
        
        # Robot dynamics parameters
        self.Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
        self.Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])

        # Initialize state variables
        self.current_pose = np.array([-1.0, 0.0, 0.5, np.deg2rad(5.0)])  # [x, y, z, yaw]
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])
        self.command = np.array([0.0, 0.0, 0.0, 0.0])
        self.t_start = time.time()

        # Subscriber for cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publisher for current pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/current_pose',
            10
        )

        # Timer for updating and publishing pose at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self.update_pose)

    def cmd_vel_callback(self, msg):
        # Extract body frame velocities from TwistStamped
        vx_body = msg.twist.linear.x
        vy_body = msg.twist.linear.y
        vz_body = msg.twist.linear.z
        yaw_rate = msg.twist.angular.z

        # Convert body velocities to world frame
        yaw = self.current_pose[3]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw
        vz_world = vz_body

        # Update world frame velocity
        self.command = np.array([vx_world, vy_world, vz_world, yaw_rate])

    def update_pose(self):
        current_time = time.time()
        delta_t = current_time - self.t_start
        self.t_start = current_time

        # Simple model
        # self.world_frame_current_velocity = self.command
        # self.current_pose += self.world_frame_current_velocity * delta_t

        # Dynamic model
        acceleration = self.Ku @ self.command - self.Kv @ self.world_frame_current_velocity
        self.world_frame_current_velocity += acceleration / 30.0
        self.current_pose += self.world_frame_current_velocity / 30.0

        # Publish current pose as PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = self.current_pose[0]
        pose_msg.pose.position.y = self.current_pose[1]
        pose_msg.pose.position.z = self.current_pose[2]
        pose_msg.pose.orientation.w = np.cos(self.current_pose[3] / 2)
        pose_msg.pose.orientation.z = np.sin(self.current_pose[3] / 2)
        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    simulator = RobotSimulator()
    rclpy.spin(simulator)
    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
