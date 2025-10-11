#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import csv
import argparse
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
import message_filters

class F1TenthDataCollector(Node):
    def __init__(self, csv_filename):
        super().__init__('f1tenth_data_collector')
        self.csv_filename = csv_filename
        self.csv_file = None
        self.csv_writer = None

        # Initialize CSV file
        self.init_csv()

        # ROS2 subscribers with QoS profiles
        # Create QoS profile for odometry with BEST_EFFORT reliability
        odom_qos = QoSProfile(depth=10)
        odom_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Create subscribers
        self.odom_sub = message_filters.Subscriber(
            self,
            Odometry,
            '/ego_racecar/odom',
            qos_profile=odom_qos
        )

        self.drive_sub = message_filters.Subscriber(
            self,
            AckermannDriveStamped,
            '/drive',
            qos_profile=10
        )

        # Synchronize messages based on timestamps
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.drive_sub],
            queue_size=10,
            slop=0.03  # 30 ms tolerance
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info("F1Tenth data collector initialized with message_filters synchronization")

    def init_csv(self):
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV headers
        headers = [
            'timestamp',
            'pos_x', 'pos_y', 'pos_z',
            'quat_x', 'quat_y', 'quat_z', 'quat_w',
            'roll', 'pitch', 'yaw',
            'linear_vel_x', 'linear_vel_y', 'linear_vel_z',
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
            'speed',
            'steering_angle'
        ]
        self.csv_writer.writerow(headers)
        self.get_logger().info(f"CSV file '{self.csv_filename}' created with headers")

    def synchronized_callback(self, odom_msg, drive_msg):
        # Use odometry message timestamp
        timestamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9

        # Extract position
        pos_x = odom_msg.pose.pose.position.x
        pos_y = odom_msg.pose.pose.position.y
        pos_z = odom_msg.pose.pose.position.z

        # Extract quaternion
        quat_x = odom_msg.pose.pose.orientation.x
        quat_y = odom_msg.pose.pose.orientation.y
        quat_z = odom_msg.pose.pose.orientation.z
        quat_w = odom_msg.pose.pose.orientation.w

        # Convert quaternion to euler angles
        euler = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])
        roll, pitch, yaw = euler

        # Extract velocities
        linear_vel_x = odom_msg.twist.twist.linear.x
        linear_vel_y = odom_msg.twist.twist.linear.y
        linear_vel_z = odom_msg.twist.twist.linear.z
        angular_vel_x = odom_msg.twist.twist.angular.x
        angular_vel_y = odom_msg.twist.twist.angular.y
        angular_vel_z = odom_msg.twist.twist.angular.z

        # Extract control inputs from AckermannDriveStamped message
        speed = drive_msg.drive.speed
        steering_angle = drive_msg.drive.steering_angle

        # Write to CSV
        row = [
            timestamp,
            pos_x, pos_y, pos_z,
            quat_x, quat_y, quat_z, quat_w,
            roll, pitch, yaw,
            linear_vel_x, linear_vel_y, linear_vel_z,
            angular_vel_x, angular_vel_y, angular_vel_z,
            speed,
            steering_angle
        ]

        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.get_logger().info(f"Wrote data row at timestamp {timestamp}")

    def close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            self.get_logger().info(f"CSV file '{self.csv_filename}' closed")

def main():
    parser = argparse.ArgumentParser(description='Collect F1Tenth data and save to CSV')
    parser.add_argument('csv_filename', help='Output CSV filename')

    args = parser.parse_args()

    rclpy.init()

    collector = F1TenthDataCollector(args.csv_filename)

    try:
        collector.get_logger().info("Starting data collection with synchronized messages. Press Ctrl+C to stop.")
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Data collection stopped by user")
    finally:
        collector.close_csv()
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
