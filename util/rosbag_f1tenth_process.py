#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
import csv
import argparse
import time
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion

class F1TenthDataCollector(Node):
    def __init__(self, csv_filename):
        super().__init__('f1tenth_data_collector')
        self.csv_filename = csv_filename
        self.csv_file = None
        self.csv_writer = None

        # Store latest message from each topic
        self.latest_odometry = None
        self.latest_motor_speed = None
        self.latest_servo_position = None

        # Initialize CSV file
        self.init_csv()

        # Create timer for 0.1 second sampling
        self.timer = self.create_timer(0.05, self.timer_callback)

        # ROS2 subscribers with QoS profiles
        # Create QoS profile for odometry with BEST_EFFORT reliability
        odom_qos = QoSProfile(depth=10)
        odom_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.odom_sub = self.create_subscription(
            Odometry,
            '/fixposition/odometry_enu',
            self.odometry_callback,
            odom_qos
        )

        self.motor_sub = self.create_subscription(
            Float64,
            '/commands/motor/speed',
            self.motor_speed_callback,
            10
        )

        self.servo_sub = self.create_subscription(
            Float64,
            '/commands/servo/position',
            self.servo_position_callback,
            10
        )

        self.get_logger().info("F1Tenth data collector initialized")

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
            'motor_speed',
            'servo_position'
        ]
        self.csv_writer.writerow(headers)
        self.get_logger().info(f"CSV file '{self.csv_filename}' created with headers")

    def odometry_callback(self, msg):
        current_time = self.get_clock().now()
        timestamp = current_time.nanoseconds * 1e-9

        # Extract position
        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        pos_z = msg.pose.pose.position.z

        # Extract quaternion
        quat_x = msg.pose.pose.orientation.x
        quat_y = msg.pose.pose.orientation.y
        quat_z = msg.pose.pose.orientation.z
        quat_w = msg.pose.pose.orientation.w

        # Convert quaternion to euler angles
        euler = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])
        roll, pitch, yaw = euler

        # Extract velocities
        linear_vel_x = msg.twist.twist.linear.x
        linear_vel_y = msg.twist.twist.linear.y
        linear_vel_z = msg.twist.twist.linear.z
        angular_vel_x = msg.twist.twist.angular.x
        angular_vel_y = msg.twist.twist.angular.y
        angular_vel_z = msg.twist.twist.angular.z

        self.latest_odometry = {
            'timestamp': timestamp,
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'quat_x': quat_x, 'quat_y': quat_y, 'quat_z': quat_z, 'quat_w': quat_w,
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'linear_vel_x': linear_vel_x, 'linear_vel_y': linear_vel_y, 'linear_vel_z': linear_vel_z,
            'angular_vel_x': angular_vel_x, 'angular_vel_y': angular_vel_y, 'angular_vel_z': angular_vel_z
        }

    def motor_speed_callback(self, msg):
        current_time = self.get_clock().now()
        timestamp = current_time.nanoseconds * 1e-9

        self.latest_motor_speed = {
            'timestamp': timestamp,
            'motor_speed': msg.data
        }

    def servo_position_callback(self, msg):
        current_time = self.get_clock().now()
        timestamp = current_time.nanoseconds * 1e-9

        self.latest_servo_position = {
            'timestamp': timestamp,
            'servo_position': msg.data
        }

    def timer_callback(self):
        current_time = self.get_clock().now()
        timestamp = current_time.nanoseconds * 1e-9

        if self.latest_odometry:
            row = [
                timestamp,
                self.latest_odometry['pos_x'], self.latest_odometry['pos_y'], self.latest_odometry['pos_z'],
                self.latest_odometry['quat_x'], self.latest_odometry['quat_y'], self.latest_odometry['quat_z'], self.latest_odometry['quat_w'],
                self.latest_odometry['roll'], self.latest_odometry['pitch'], self.latest_odometry['yaw'],
                self.latest_odometry['linear_vel_x'], self.latest_odometry['linear_vel_y'], self.latest_odometry['linear_vel_z'],
                self.latest_odometry['angular_vel_x'], self.latest_odometry['angular_vel_y'], self.latest_odometry['angular_vel_z'],
                self.latest_motor_speed['motor_speed'] if self.latest_motor_speed else 0.0,
                self.latest_servo_position['servo_position'] if self.latest_servo_position else 0.0
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
        collector.get_logger().info("Starting data collection with 0.05s timer. Press Ctrl+C to stop.")
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Data collection stopped by user")
    finally:
        collector.close_csv()
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()