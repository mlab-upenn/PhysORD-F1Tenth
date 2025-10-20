#!/usr/bin/env python
"""
ROS1 script to subscribe to reachtruck topics and log data to CSV.
Logs data at the frequency of /cmd_vel_stamped topic.
"""

import rospy
import csv
import os
from datetime import datetime
from geometry_msgs.msg import TwistStamped, PoseWithCovariance
from std_msgs.msg import Float32
from ipa_navigation_msgs.msg import LTSNGStatus


class ReachtruckDataLogger:
    def __init__(self, output_csv_path=None):
        """
        Initialize the data logger.

        Args:
            output_csv_path: Path to output CSV file. If None, generates timestamped filename.
        """
        # Latest messages from each topic
        self.latest_speed = None
        self.latest_steer_angle = None
        self.latest_lts_status = None

        # CSV file setup
        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_path = f"reachtruck_data_{timestamp}.csv"

        self.csv_file = open(output_csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write CSV header
        self.csv_writer.writerow([
            'timestamp',
            'cmd_speed',
            'cmd_steer_angle',
            'measured_speed',
            'measured_steer_angle',
            'x',
            'y',
            'orientation_x',
            'orientation_y',
            'orientation_z',
            'orientation_w'
        ])

        rospy.loginfo(f"Logging data to: {output_csv_path}")

        # Initialize subscribers
        self.cmd_vel_sub = rospy.Subscriber(
            '/cmd_vel_stamped',
            TwistStamped,
            self.cmd_vel_callback,
            queue_size=1
        )

        self.speed_sub = rospy.Subscriber(
            '/firmware/feedback/speed',
            Float32,
            self.speed_callback,
            queue_size=1
        )

        self.steer_angle_sub = rospy.Subscriber(
            '/firmware/feedback/steer_angle',
            Float32,
            self.steer_angle_callback,
            queue_size=1
        )

        self.lts_status_sub = rospy.Subscriber(
            '/lts_ng/lts_status',
            LTSNGStatus,
            self.lts_status_callback,
            queue_size=1
        )

        rospy.loginfo("ReachtruckDataLogger initialized and ready")

    def speed_callback(self, msg):
        """Callback for measured speed."""
        self.latest_speed = msg.data

    def steer_angle_callback(self, msg):
        """Callback for measured steering angle."""
        self.latest_steer_angle = msg.data

    def lts_status_callback(self, msg):
        """Callback for localization status."""
        self.latest_lts_status = msg

    def cmd_vel_callback(self, msg):
        """
        Callback for command velocity (control inputs).
        Logs data to CSV when this message is received.
        """
        # Extract timestamp from header
        timestamp = msg.header.stamp.to_sec()

        # Extract command values
        cmd_speed = msg.twist.linear.x
        cmd_steer_angle = msg.twist.angular.z

        # Get measured values (use None if not yet received)
        measured_speed = self.latest_speed if self.latest_speed is not None else float('nan')
        measured_steer_angle = self.latest_steer_angle if self.latest_steer_angle is not None else float('nan')

        # Extract localization data
        if self.latest_lts_status is not None:
            x = self.latest_lts_status.pose.pose.position.x
            y = self.latest_lts_status.pose.pose.position.y
            orientation_x = self.latest_lts_status.pose.pose.orientation.x
            orientation_y = self.latest_lts_status.pose.pose.orientation.y
            orientation_z = self.latest_lts_status.pose.pose.orientation.z
            orientation_w = self.latest_lts_status.pose.pose.orientation.w
        else:
            x = y = float('nan')
            orientation_x = orientation_y = orientation_z = orientation_w = float('nan')

        # Write row to CSV
        self.csv_writer.writerow([
            timestamp,
            cmd_speed,
            cmd_steer_angle,
            measured_speed,
            measured_steer_angle,
            x,
            y,
            orientation_x,
            orientation_y,
            orientation_z,
            orientation_w
        ])

        # Flush to ensure data is written
        self.csv_file.flush()

        rospy.loginfo_throttle(5.0, f"Logging data... Latest timestamp: {timestamp}")

    def shutdown(self):
        """Clean up resources on shutdown."""
        rospy.loginfo("Shutting down data logger...")
        if self.csv_file:
            self.csv_file.close()
        rospy.loginfo("CSV file closed successfully")


def main():
    """Main function to run the data logger node."""
    rospy.init_node('reachtruck_data_logger', anonymous=True)

    # Get output path from ROS parameter, if specified
    output_path = rospy.get_param('~output_csv', None)

    # Create logger instance
    logger = ReachtruckDataLogger(output_csv_path=output_path)

    # Register shutdown hook
    rospy.on_shutdown(logger.shutdown)

    rospy.loginfo("Reachtruck data logger running. Press Ctrl+C to stop.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
