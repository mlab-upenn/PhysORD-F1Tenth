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
    Convert a single CSV file to state tensor format.
    Returns tensor of shape (num_sequences, timesteps, 29) or None if insufficient data.
    """
    try:
        # Load CSV data
        df = pd.read_csv(csv_file)

        if len(df) < timesteps:
            print(f"Skipping {csv_file}: insufficient timesteps ({len(df)} < {timesteps})")
            return None

        # Extract data from CSV
        position = df[['pos_x', 'pos_y', 'pos_z']].values  # Shape: (N, 3)
        quaternion = df[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values  # Shape: (N, 4)
        linear_vel = df[['linear_vel_x', 'linear_vel_y', 'linear_vel_z']].values  # Shape: (N, 3)
        angular_vel = df[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']].values  # Shape: (N, 3)
        motor_speed = df['motor_speed'].values.reshape(-1, 1)/16000.0  # Shape: (N, 1)
        servo_position = df['servo_position'].values.reshape(-1, 1)  # Shape: (N, 1)

        # Convert quaternions to rotation matrices
        rot = Rotation.from_quat(quaternion).as_matrix()  # Shape: (N, 3, 3)
        rot_flat = rot.reshape(len(df), 9)  # Flatten to (N, 9)

        # Create placeholder zeros for shock travel and wheel RPM (4 values each)
        shock_travel = np.zeros((len(df), 4))  # Shape: (N, 4)
        wheel_rpm = np.zeros((len(df), 4))  # Shape: (N, 4)

        # Control inputs: motor_speed, servo_position, brake=0
        brake = np.zeros((len(df), 1))  # Shape: (N, 1)
        controls = np.concatenate([motor_speed, servo_position, brake], axis=1)  # Shape: (N, 3)

        # Combine all features: pos(3) + rot(9) + lin_vel(3) + ang_vel(3) + shock(4) + wheel(4) + controls(3) = 29
        state = np.concatenate([
            position,      # 3 features
            rot_flat,      # 9 features
            linear_vel,    # 3 features
            angular_vel,   # 3 features
            shock_travel,  # 4 features (zeros)
            wheel_rpm,     # 4 features (zeros)
            controls       # 3 features
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
            initial_pos = sequences[i, 0, :3]  # Initial x, y, z for this sequence
            sequences_normalized[i, :, :3] -= initial_pos  # Subtract from all timesteps

        return torch.from_numpy(sequences_normalized).float()

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def process_all_csvs_to_tensor(csv_dir, output_path, timesteps=21):
    """
    Process all CSV files in directory and create a single tensor.
    Final tensor shape: (timesteps, total_sequences, 29)
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
    combined_sequences = torch.cat(all_sequences, dim=0)  # Shape: (total_sequences, timesteps, 29)

    # Transpose to get desired shape: (timesteps, total_sequences, 29)
    final_tensor = combined_sequences.transpose(0, 1)

    print(f"\nFinal tensor shape: {final_tensor.shape}")
    print(f"Total sequences: {total_sequences}")

    # Save the tensor
    torch.save(final_tensor, output_path)
    print(f"Tensor saved to: {output_path}")

    return final_tensor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert CSV files to tensor format')
    parser.add_argument('--timesteps', type=int, default=21,
                        help='Number of timesteps per sequence (default: 21)')
    parser.add_argument('--csv_dir', type=str, default="./data/f1fifth_csvs/",
                        help='Directory containing CSV files (default: ./data/f1fifth_csvs/)')
    parser.add_argument('--output', type=str, default="./data/custom_f1fifth_6.pt",
                        help='Output tensor file path (default: ./data/custom_f1fifth_6.pt)')
    args = parser.parse_args()

    # Configuration
    csv_directory = args.csv_dir
    output_file = args.output
    timesteps = args.timesteps

    print("Starting CSV to tensor conversion...")
    print(f"Input directory: {csv_directory}")
    print(f"Output file: {output_file}")
    print(f"Timesteps per sequence: {timesteps}")
    print("-" * 50)

    # Process all CSV files
    tensor = process_all_csvs_to_tensor(csv_directory, output_file, timesteps)

    print("-" * 50)
    print("Conversion completed successfully!")
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Expected format: [timesteps={timesteps}, num_trajectories={tensor.shape[1]}, features=29]")

    # Verify the structure
    print("\nTensor verification:")
    print(f"  Position features (0:3): {tensor[0, 0, :3]}")
    print(f"  Rotation matrix features (3:12): {tensor[0, 0, 3:12]}")
    print(f"  Linear velocity features (12:15): {tensor[0, 0, 12:15]}")
    print(f"  Angular velocity features (15:18): {tensor[0, 0, 15:18]}")
    print(f"  Shock travel features (18:22) - should be zeros: {tensor[0, 0, 18:22]}")
    print(f"  Wheel RPM features (22:26) - should be zeros: {tensor[0, 0, 22:26]}")
    print(f"  Control features (26:29): {tensor[0, 0, 26:29]}")

if __name__ == "__main__":
    main()