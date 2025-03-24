import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from tf.transformations import quaternion_from_euler

class RobotSimulator:
    def __init__(self):
        rospy.init_node('robot_simulator', anonymous=True)
        
        # Robot dynamics parameters
        self.Ku = np.diag([0.8417, 0.8354, 3.966, 9.8524])
        self.Kv = np.diag([0.18227, 0.17095, 4.001, 4.7295])

        # Initialize state variables
        self.current_pose = np.array([2.0, -1.0, 0.5, np.deg2rad(5.0)])  # [x, y, z, yaw]
        self.world_frame_current_velocity = np.array([0.0, 0.0, 0.0, 0.0])

        # Subscriber for cmd_vel
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', TwistStamped, self.cmd_vel_callback)

        # Publisher for current pose
        self.pose_pub = rospy.Publisher('/current_pose', PoseStamped, queue_size=10)

        # Timer for updating and publishing pose at 200 Hz
        self.timer = rospy.Timer(rospy.Duration(1.0 / 200.0), self.update_pose)

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
        self.world_frame_current_velocity = np.array([vx_world, vy_world, vz_world, yaw_rate])

    def update_pose(self, event):
        # Update robot pose using the dynamics model
        acceleration = self.Ku @ self.world_frame_current_velocity - self.Kv @ self.world_frame_current_velocity
        self.world_frame_current_velocity += acceleration / 200.0
        self.current_pose += self.world_frame_current_velocity / 200.0

        # Publish current pose as PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = self.current_pose[0]
        pose_msg.pose.position.y = self.current_pose[1]
        pose_msg.pose.position.z = self.current_pose[2]
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, self.current_pose[3])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        simulator = RobotSimulator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
