#!/usr/bin/env python
"""
ROS1 script to subscribe to reachtruck topics and log data to CSV.
Logs data at the frequency of synchronized /cmd_vel_stamped and /lts_ng/lts_status topics.
"""

import rospy
import csv
import os
from datetime import datetime
from geometry_msgs.msg import TwistStamped, PoseWithCovariance
from std_msgs.msg import Float32
from ipa_navigation_msgs.msg import LTSNGStatus
import message_filters


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

        # Initialize subscribers for speed and steer angle (non-synchronized)
        self.speed_sub = rospy.Subscriber(
            '/firmware/feedback/speed',
            Float32,
            self.speed_callback,
            queue_size=20
        )

        self.steer_angle_sub = rospy.Subscriber(
            '/firmware/feedback/steer_angle',
            Float32,
            self.steer_angle_callback,
            queue_size=20
        )

        # Initialize message_filters subscribers for synchronization
        self.cmd_vel_sub = message_filters.Subscriber(
            '/cmd_vel_stamped',
            TwistStamped
        )

        self.lts_status_sub = message_filters.Subscriber(
            '/lts_ng/lts_status',
            LTSNGStatus
        )

        # Create approximate time synchronizer
        # slop parameter defines the maximum time difference (in seconds) between messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.cmd_vel_sub, self.lts_status_sub],
            queue_size=20,
            slop=0.1  # 100 ms tolerance
        )
        self.ts.registerCallback(self.synchronized_callback)

        rospy.loginfo("ReachtruckDataLogger initialized and ready with message synchronization")

    def speed_callback(self, msg):
        """Callback for measured speed."""
        self.latest_speed = msg.data

    def steer_angle_callback(self, msg):
        """Callback for measured steering angle."""
        self.latest_steer_angle = msg.data

    def synchronized_callback(self, cmd_vel_msg, lts_status_msg):
        """
        Callback for synchronized /cmd_vel_stamped and /lts_ng/lts_status messages.
        Logs data to CSV when synchronized messages are received.

        Args:
            cmd_vel_msg: TwistStamped message from /cmd_vel_stamped
            lts_status_msg: LTSNGStatus message from /lts_ng/lts_status
        """
        # Extract timestamp from cmd_vel header
        timestamp = cmd_vel_msg.header.stamp.to_sec()

        # Extract command values from synchronized cmd_vel_msg
        cmd_speed = cmd_vel_msg.twist.linear.x
        cmd_steer_angle = cmd_vel_msg.twist.angular.z

        # Get latest measured values (use NaN if not yet received)
        measured_speed = self.latest_speed if self.latest_speed is not None else float('nan')
        measured_steer_angle = self.latest_steer_angle if self.latest_steer_angle is not None else float('nan')

        # Extract localization data from synchronized lts_status_msg
        x = lts_status_msg.pose.pose.position.x
        y = lts_status_msg.pose.pose.position.y
        orientation_x = lts_status_msg.pose.pose.orientation.x
        orientation_y = lts_status_msg.pose.pose.orientation.y
        orientation_z = lts_status_msg.pose.pose.orientation.z
        orientation_w = lts_status_msg.pose.pose.orientation.w

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
