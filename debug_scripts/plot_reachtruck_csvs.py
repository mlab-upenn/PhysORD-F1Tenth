#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob

def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw angle (rotation about z-axis).

    Args:
        qx, qy, qz, qw: Quaternion components

    Returns:
        Yaw angle in radians
    """
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def plot_single_bag(csv_file, arrow_spacing=10):
    """Plot trajectory and time series data from a single reachtruck CSV file.

    Args:
        csv_file: Path to CSV file
        arrow_spacing: Number of timesteps between orientation arrows (default: 10)
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        return None

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return None

    # Check for required columns
    required_columns = ['timestamp', 'cmd_speed', 'cmd_steer_angle',
                       'measured_speed', 'measured_steer_angle', 'x', 'y',
                       'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: CSV file {csv_file} must contain columns: {required_columns}")
        print(f"Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return None

    print(f"Plotting data from {csv_file} with {len(df)} points")

    # Convert quaternion to yaw
    df['yaw'] = quaternion_to_yaw(df['orientation_x'].values,
                                   df['orientation_y'].values,
                                   df['orientation_z'].values,
                                   df['orientation_w'].values)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Reachtruck Data: {os.path.basename(csv_file)}')

    # Plot 1: 2D Trajectory
    ax1 = axes[0, 0]
    ax1.plot(df['x'].values, df['y'].values, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', s=100, label='End', zorder=5)

    # Add orientation arrows at regularly spaced timesteps
    # Calculate arrow scale based on trajectory extent
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    arrow_scale = max(x_range, y_range) * 0.03  # Arrow length is 3% of plot range

    # Select regularly spaced indices
    arrow_indices = range(0, len(df), arrow_spacing)

    for idx in arrow_indices:
        x = df['x'].iloc[idx]
        y = df['y'].iloc[idx]
        yaw = df['yaw'].iloc[idx]

        # Calculate arrow direction (yaw is the heading angle)
        dx = arrow_scale * np.cos(yaw)
        dy = arrow_scale * np.sin(yaw)

        # Draw arrow
        ax1.arrow(x, y, dx, dy,
                 head_width=arrow_scale*0.4,
                 head_length=arrow_scale*0.6,
                 fc='red', ec='red', alpha=0.6, zorder=4)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory with Orientation Arrows')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()

    # Convert timestamp to relative time (seconds from start)
    time_relative = df['timestamp'] - df['timestamp'].iloc[0]

    # Plot 2: Command Speed vs Time
    ax2 = axes[0, 1]
    ax2.plot(time_relative.values, df['cmd_speed'].values, 'r-', linewidth=1, label='Command', alpha=0.7)
    ax2.plot(time_relative.values, df['measured_speed'].values, 'b-', linewidth=1, label='Measured', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Speed vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Command Steering Angle vs Time
    ax3 = axes[1, 0]
    ax3.plot(time_relative.values, df['cmd_steer_angle'].values, 'r-', linewidth=1, label='Command', alpha=0.7)
    ax3.plot(time_relative.values, df['measured_steer_angle'].values, 'b-', linewidth=1, label='Measured', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Steering Angle (deg)')
    ax3.set_title('Steering Angle vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Command Speed vs Steering Angle
    ax4 = axes[1, 1]
    ax4.scatter(df['cmd_steer_angle'], df['cmd_speed'], alpha=0.6, s=10, label='Command')
    ax4.scatter(df['measured_steer_angle'], df['measured_speed'], alpha=0.6, s=10, label='Measured')
    ax4.set_xlabel('Steering Angle (deg)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Speed vs Steering Angle')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Yaw angle vs Time
    ax5 = axes[2, 0]
    ax5.plot(time_relative.values, np.rad2deg(df['yaw'].values), 'g-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Yaw Angle (deg)')
    ax5.set_title('Yaw Angle vs Time')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Speed tracking error
    ax6 = axes[2, 1]
    speed_error = df['measured_speed'] - df['cmd_speed']
    steer_error = df['measured_steer_angle'] - df['cmd_steer_angle']
    ax6_twin = ax6.twinx()

    ax6.plot(time_relative.values, speed_error.values, 'r-', linewidth=1, alpha=0.7, label='Speed Error')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Speed Error (m/s)', color='r')
    ax6.tick_params(axis='y', labelcolor='r')
    ax6.grid(True, alpha=0.3)

    ax6_twin.plot(time_relative.values, steer_error.values, 'b-', linewidth=1, alpha=0.7, label='Steer Error')
    ax6_twin.set_ylabel('Steering Error (deg)', color='b')
    ax6_twin.tick_params(axis='y', labelcolor='b')

    ax6.set_title('Command Tracking Errors')

    plt.tight_layout()
    return fig

def plot_all_bags(csv_files, arrow_spacing=10):
    """Plot trajectory data from multiple reachtruck CSV files.

    Args:
        csv_files: List of paths to CSV files
        arrow_spacing: Number of timesteps between orientation arrows (default: 10)
    """
    # Create a combined trajectory plot
    fig_combined, ax_combined = plt.subplots(figsize=(12, 10))
    fig_combined.suptitle('Reachtruck Data: All Trajectories')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for idx, csv_file in enumerate(csv_files):
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} does not exist, skipping...")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}, skipping...")
            continue

        color = colors[idx % len(colors)]
        label = os.path.basename(csv_file).replace('reachtruck_data_low_level_', 'Bag ').replace('.csv', '')

        ax_combined.plot(df['x'].values, df['y'].values, '-', linewidth=1.5,
                        alpha=0.7, color=color, label=label)
        ax_combined.scatter(df['x'].iloc[0], df['y'].iloc[0], color=color, s=100,
                          marker='o', zorder=5, edgecolors='black', linewidths=2)
        ax_combined.scatter(df['x'].iloc[-1], df['y'].iloc[-1], color=color, s=100,
                          marker='s', zorder=5, edgecolors='black', linewidths=2)

    ax_combined.set_xlabel('X Position (m)')
    ax_combined.set_ylabel('Y Position (m)')
    ax_combined.set_title('All Trajectories (circles=start, squares=end)')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.axis('equal')
    ax_combined.legend()

    plt.tight_layout()
    return fig_combined

def main():
    parser = argparse.ArgumentParser(description='Plot trajectory and time series data from reachtruck CSV files')
    parser.add_argument('--csv-files', nargs='+',
                       help='Paths to CSV files containing reachtruck data')
    parser.add_argument('--arrow-spacing', type=int, default=10,
                       help='Number of timesteps between orientation arrows (default: 10)')
    parser.add_argument('--all-bags', action='store_true',
                       help='Plot all bags matching pattern reachtruck_data_low_level_bag_*.csv in specified directory')
    parser.add_argument('--directory', type=str, default='util',
                       help='Directory containing CSV files (default: util)')
    parser.add_argument('--combined-only', action='store_true',
                       help='Only show combined trajectory plot (requires --all-bags or multiple --csv-files)')

    args = parser.parse_args()

    # Determine which files to plot
    csv_files = []
    if args.all_bags:
        # Look for all bag files matching the pattern in specified directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.directory)
        pattern = os.path.join(data_dir, 'reachtruck_data_low_level_bag_*.csv')
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print(f"Warning: No CSV files matching pattern 'reachtruck_data_low_level_bag_*.csv' found in {data_dir}")
    elif args.csv_files:
        csv_files = args.csv_files
    else:
        parser.print_help()
        print("\nError: Must specify either --all-bags or --csv-files")
        sys.exit(1)

    # Plot individual bags unless combined-only is specified
    if not args.combined_only:
        for csv_file in csv_files:
            fig = plot_single_bag(csv_file, arrow_spacing=args.arrow_spacing)

    # Plot combined trajectory if multiple files
    if len(csv_files) > 1:
        fig_combined = plot_all_bags(csv_files, arrow_spacing=args.arrow_spacing)

    plt.show()

if __name__ == '__main__':
    main()
