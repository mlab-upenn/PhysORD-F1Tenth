#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_data(csv_file, arrow_spacing=10):
    """Plot trajectory and time series data from CSV file.

    Args:
        csv_file: Path to CSV file
        arrow_spacing: Number of timesteps between orientation arrows (default: 10)
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Check for required position and timestamp columns
    required_base_columns = ['pos_x', 'pos_y', 'timestamp']
    missing_base_columns = [col for col in required_base_columns if col not in df.columns]

    if missing_base_columns:
        print(f"Error: CSV file must contain columns: {required_base_columns}")
        print(f"Missing columns: {missing_base_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Detect which control input format is available
    has_motor_servo = 'motor_speed' in df.columns and 'servo_position' in df.columns
    has_speed_steering = 'speed' in df.columns and 'steering_angle' in df.columns

    if not has_motor_servo and not has_speed_steering:
        print(f"Error: CSV file must contain either:")
        print(f"  - 'motor_speed' and 'servo_position' columns, OR")
        print(f"  - 'speed' and 'steering_angle' columns")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Set control column names based on what's available
    if has_motor_servo:
        control_col_1 = 'motor_speed'
        control_col_2 = 'servo_position'
        control_label_1 = 'Motor Speed'
        control_label_2 = 'Servo Position'
    else:
        control_col_1 = 'speed'
        control_col_2 = 'steering_angle'
        control_label_1 = 'Speed'
        control_label_2 = 'Steering Angle'

    print(f"Detected control format: {control_label_1} and {control_label_2}")

    # Check if orientation data (yaw) is available
    has_yaw = 'yaw' in df.columns
    if not has_yaw:
        print("Warning: 'yaw' column not found - orientation arrows will not be plotted")

    print(f"Plotting data from {csv_file} with {len(df)} points")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'F1tenth Data: {os.path.basename(csv_file)}')

    # Plot 1: 2D Trajectory
    ax1 = axes[0, 0]
    ax1.plot(df['pos_x'].values, df['pos_y'].values, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], color='red', s=100, label='End', zorder=5)

    # Add orientation arrows at regularly spaced timesteps
    if has_yaw:
        # Calculate arrow scale based on trajectory extent
        x_range = df['pos_x'].max() - df['pos_x'].min()
        y_range = df['pos_y'].max() - df['pos_y'].min()
        arrow_scale = max(x_range, y_range) * 0.03  # Arrow length is 3% of plot range

        # Select regularly spaced indices
        arrow_indices = range(0, len(df), arrow_spacing)

        for idx in arrow_indices:
            x = df['pos_x'].iloc[idx]
            y = df['pos_y'].iloc[idx]
            yaw = df['yaw'].iloc[idx]

            # Calculate arrow direction (yaw is the heading angle)
            # In standard convention, yaw=0 points along x-axis
            dx = arrow_scale * np.cos(yaw)
            dy = arrow_scale * np.sin(yaw)

            # Draw arrow
            ax1.arrow(x, y, dx, dy,
                     head_width=arrow_scale*0.4,
                     head_length=arrow_scale*0.6,
                     fc='red', ec='red', alpha=0.6, zorder=4)

    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('2D Trajectory with Orientation Arrows')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()

    # Convert timestamp to relative time (seconds from start)
    time_relative = df['timestamp'] - df['timestamp'].iloc[0]

    # Plot 2: Control Input 1 vs Time (Motor Speed or Speed)
    ax2 = axes[0, 1]
    ax2.plot(time_relative.values, df[control_col_1].values, 'r-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(control_label_1)
    ax2.set_title(f'{control_label_1} vs Time')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control Input 2 vs Time (Servo Position or Steering Angle)
    ax3 = axes[1, 0]
    ax3.plot(time_relative.values, df[control_col_2].values, 'g-', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(control_label_2)
    ax3.set_title(f'{control_label_2} vs Time')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Control Input 1 vs Control Input 2
    ax4 = axes[1, 1]
    ax4.scatter(df[control_col_2], df[control_col_1], alpha=0.6, s=10)
    ax4.set_xlabel(control_label_2)
    ax4.set_ylabel(control_label_1)
    ax4.set_title(f'{control_label_1} vs {control_label_2}')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot trajectory and time series data from F1tenth CSV files')
    parser.add_argument('csv_file', help='Path to CSV file containing trajectory data')
    parser.add_argument('--arrow-spacing', type=int, default=10,
                       help='Number of timesteps between orientation arrows (default: 10)')

    args = parser.parse_args()
    plot_data(args.csv_file, arrow_spacing=args.arrow_spacing)

if __name__ == '__main__':
    main()