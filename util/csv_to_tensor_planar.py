import torch
import pandas as pd
import numpy as np
import os
import glob
import argparse
from scipy.spatial.transform import Rotation

def arrange_data_sample(x, num_points=21, sample_intervals=1):
    """Create overlapping sequences of length num_points from trajectory data"""
    assert num_points >= 2
    sample_tidxs = list(range(0, x.shape[0] - num_points + 1, sample_intervals))
    x_stack = []
    for tidx in sample_tidxs:
        x_stack.append(x[tidx:tidx + num_points, :])
    if len(x_stack) == 0:
        return np.array([]).reshape(0, num_points, x.shape[1])
    x_stack = np.stack(x_stack, axis=0)  # Shape: (num_sequences, num_points, features)
    return x_stack

def process_csv_to_state_tensor(csv_file, timesteps=21):
    """
    Convert a single CSV file to planar state tensor format.
    Returns tensor of shape (num_sequences, timesteps, 16) or None if insufficient data.

    16 features:
    - 2: position (x, y)
    - 1: angle (yaw)
    - 2: linear velocity (vx, vy)
    - 1: angular velocity (ωz)
    - 4: shock travel (all zeros)
    - 4: wheel RPM (all zeros)
    - 2: control inputs (speed, steering_angle)
    """
    try:
        # Load CSV data
        df = pd.read_csv(csv_file)

        if len(df) < timesteps:
            print(f"Skipping {csv_file}: insufficient timesteps ({len(df)} < {timesteps})")
            return None

        # Extract data from CSV
        position = df[['pos_x', 'pos_y']].values  # Shape: (N, 2) - only x, y
        quaternion = df[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values  # Shape: (N, 4)
        linear_vel = df[['linear_vel_x', 'linear_vel_y']].values  # Shape: (N, 2) - only vx, vy
        angular_vel_z = df['angular_vel_z'].values.reshape(-1, 1)  # Shape: (N, 1) - only ωz
        motor_speed = df['speed'].values.reshape(-1, 1)/6.0  # Shape: (N, 1)
        servo_position = df['steering_angle'].values.reshape(-1, 1)  # Shape: (N, 1)

        # Convert quaternions to yaw angle (rotation around z-axis)
        rot = Rotation.from_quat(quaternion)
        euler = rot.as_euler('xyz', degrees=False)  # Shape: (N, 3) - roll, pitch, yaw
        yaw = euler[:, 2].reshape(-1, 1)  # Shape: (N, 1) - extract yaw angle

        # Create placeholder zeros for shock travel and wheel RPM (4 values each)
        shock_travel = np.zeros((len(df), 4))  # Shape: (N, 4)
        wheel_rpm = np.zeros((len(df), 4))  # Shape: (N, 4)

        # Control inputs: motor_speed, servo_position
        controls = np.concatenate([motor_speed, servo_position], axis=1)  # Shape: (N, 2)

        # Combine all features: pos(2) + yaw(1) + lin_vel(2) + ang_vel_z(1) + shock(4) + wheel(4) + controls(2) = 16
        state = np.concatenate([
            position,      # 2 features: x, y
            yaw,           # 1 feature: yaw angle
            linear_vel,    # 2 features: vx, vy
            angular_vel_z, # 1 feature: ωz
            shock_travel,  # 4 features (zeros)
            wheel_rpm,     # 4 features (zeros)
            controls       # 2 features: speed, steering_angle
        ], axis=1)

        # Filter out sequences with low velocity (< 1 m/s)
        vel_magnitudes = np.sqrt(np.sum(linear_vel ** 2, axis=1))
        valid_indices = vel_magnitudes >= 1.0

        if np.sum(valid_indices) < timesteps:
            print(f"Skipping {csv_file}: insufficient high-velocity timesteps")
            return None

        state_filtered = state[valid_indices]

        # Create sequences of length timesteps
        sequences = arrange_data_sample(state_filtered, timesteps)

        if sequences.shape[0] == 0:
            print(f"Skipping {csv_file}: no valid sequences generated")
            return None

        # Normalize position by subtracting initial position for each sequence
        sequences_normalized = sequences.copy()
        for i in range(sequences.shape[0]):
            initial_pos = sequences[i, 0, :2]  # Initial x, y for this sequence
            sequences_normalized[i, :, :2] -= initial_pos  # Subtract from all timesteps

        return torch.from_numpy(sequences_normalized).float()

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def process_all_csvs_to_tensor(csv_dir, output_path, timesteps=21):
    """
    Process all CSV files in directory and create a single tensor.
    Final tensor shape: (timesteps, total_sequences, 16)
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    all_sequences = []
    total_sequences = 0

    for i, csv_file in enumerate(csv_files):
        print(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")

        sequences = process_csv_to_state_tensor(csv_file, timesteps)
        if sequences is not None:
            print(f"  Generated {sequences.shape[0]} sequences from {os.path.basename(csv_file)}")
            all_sequences.append(sequences)
            total_sequences += sequences.shape[0]
        else:
            print(f"  Skipped {os.path.basename(csv_file)}")

    if not all_sequences:
        raise ValueError("No valid sequences generated from any CSV files")

    # Concatenate all sequences along the sequence dimension
    combined_sequences = torch.cat(all_sequences, dim=0)  # Shape: (total_sequences, timesteps, 16)

    # Transpose to get desired shape: (timesteps, total_sequences, 16)
    final_tensor = combined_sequences.transpose(0, 1)

    print(f"\nFinal tensor shape: {final_tensor.shape}")
    print(f"Total sequences: {total_sequences}")

    # Save the tensor
    torch.save(final_tensor, output_path)
    print(f"Tensor saved to: {output_path}")

    return final_tensor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert CSV files to planar tensor format (16 features)')
    parser.add_argument('--timesteps', type=int, default=21,
                        help='Number of timesteps per sequence (default: 21)')
    parser.add_argument('--csv_dir', type=str, default="./csvs/",
                        help='Directory containing CSV files (default: ./csvs/)')
    parser.add_argument('--output', type=str, default="./data/planar_f1fifth.pt",
                        help='Output tensor file path (default: ./data/planar_f1fifth.pt)')
    args = parser.parse_args()

    # Configuration
    csv_directory = args.csv_dir
    output_file = args.output
    timesteps = args.timesteps

    print("Starting CSV to planar tensor conversion...")
    print(f"Input directory: {csv_directory}")
    print(f"Output file: {output_file}")
    print(f"Timesteps per sequence: {timesteps}")
    print("-" * 50)

    # Process all CSV files
    tensor = process_all_csvs_to_tensor(csv_directory, output_file, timesteps)

    print("-" * 50)
    print("Conversion completed successfully!")
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Expected format: [timesteps={timesteps}, num_trajectories={tensor.shape[1]}, features=16]")

    # Verify the structure
    print("\nTensor verification:")
    print(f"  Position features (0:2): {tensor[0, 0, :2]}")
    print(f"  Yaw angle feature (2): {tensor[0, 0, 2]}")
    print(f"  Linear velocity features (3:5): {tensor[0, 0, 3:5]}")
    print(f"  Angular velocity (ωz) feature (5): {tensor[0, 0, 5]}")
    print(f"  Shock travel features (6:10) - should be zeros: {tensor[0, 0, 6:10]}")
    print(f"  Wheel RPM features (10:14) - should be zeros: {tensor[0, 0, 10:14]}")
    print(f"  Control features (14:16): {tensor[0, 0, 14:16]}")

if __name__ == "__main__":
    main()
