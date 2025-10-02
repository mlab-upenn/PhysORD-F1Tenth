#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_data(csv_file):
    """Plot trajectory and time series data from CSV file."""
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    required_columns = ['pos_x', 'pos_y', 'timestamp', 'motor_speed', 'servo_position']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: CSV file must contain columns: {required_columns}")
        print(f"Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Plotting data from {csv_file} with {len(df)} points")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'F1tenth Data: {os.path.basename(csv_file)}')

    # Plot 1: 2D Trajectory
    ax1 = axes[0, 0]
    ax1.plot(df['pos_x'].values, df['pos_y'].values, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], color='red', s=100, label='End', zorder=5)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('2D Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()

    # Convert timestamp to relative time (seconds from start)
    time_relative = df['timestamp'] - df['timestamp'].iloc[0]

    # Plot 2: Motor Speed vs Time
    ax2 = axes[0, 1]
    ax2.plot(time_relative.values, df['motor_speed'].values, 'r-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Motor Speed')
    ax2.set_title('Motor Speed vs Time')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Servo Position vs Time
    ax3 = axes[1, 0]
    ax3.plot(time_relative.values, df['servo_position'].values, 'g-', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Servo Position')
    ax3.set_title('Servo Position vs Time')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Motor Speed vs Servo Position
    ax4 = axes[1, 1]
    ax4.scatter(df['servo_position'], df['motor_speed'], alpha=0.6, s=10)
    ax4.set_xlabel('Servo Position')
    ax4.set_ylabel('Motor Speed')
    ax4.set_title('Motor Speed vs Servo Position')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot trajectory and time series data from F1tenth CSV files')
    parser.add_argument('csv_file', help='Path to CSV file containing trajectory data')

    args = parser.parse_args()
    plot_data(args.csv_file)

if __name__ == '__main__':
    main()