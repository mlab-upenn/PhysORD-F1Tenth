import torch
import pandas as pd
import numpy as np
import os
import glob
import argparse
import fnmatch
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


def process_csv_to_state_tensor(csv_file, timesteps=20, past_history_input=2):
    """
    Convert a single reachtruck CSV file to position-only state tensor format.
    Returns tensor of shape (num_sequences, timesteps + past_history_input, 8) or None if insufficient data.

    8 features (for position_only_train.py):
    - 2: position (x, y)
    - 2: orientation (cos(theta), sin(theta))
    - 2: feedback (measured_speed, measured_steer_angle)
    - 2: control inputs (cmd_speed, cmd_steer_angle)

    CSV columns expected:
    - timestamp, cmd_speed, cmd_steer_angle, measured_speed, measured_steer_angle,
      x, y, orientation_x, orientation_y, orientation_z, orientation_w
    """
    sequence_length = timesteps + past_history_input

    try:
        # Load CSV data
        df = pd.read_csv(csv_file)

        if len(df) < sequence_length:
            print(f"Skipping {csv_file}: insufficient timesteps ({len(df)} < {sequence_length})")
            return None

        # Extract data from CSV
        position = df[['x', 'y']].values  # Shape: (N, 2)
        quaternion = df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].values  # Shape: (N, 4)

        # Feedback measurements
        feedback_speed = df['measured_speed'].values.reshape(-1, 1)  # Shape: (N, 1)
        feedback_steer = df['measured_steer_angle'].values.reshape(-1, 1)  # Shape: (N, 1)

        # Control inputs
        cmd_speed = df['cmd_speed'].values.reshape(-1, 1)  # Shape: (N, 1)
        cmd_steer = df['cmd_steer_angle'].values.reshape(-1, 1)  # Shape: (N, 1)
        # Normalize cmd_steer by dividing by 90.0.
        cmd_steer = cmd_steer / 90.0

        # Convert quaternions to yaw angle (rotation around z-axis)
        rot = Rotation.from_quat(quaternion)
        euler = rot.as_euler('xyz', degrees=False)  # Shape: (N, 3) - roll, pitch, yaw
        yaw = euler[:, 2].reshape(-1, 1)  # Shape: (N, 1) - extract yaw angle

        # Compute cos(theta) and sin(theta) instead of theta directly
        cos_theta = np.cos(yaw)  # Shape: (N, 1)
        sin_theta = np.sin(yaw)  # Shape: (N, 1)

        # Combine all features: pos(2) + orientation(2) + feedback(2) + controls(2) = 8
        state = np.concatenate([
            position,         # 2 features: x, y
            cos_theta,       # 1 feature: cos(theta)
            sin_theta,       # 1 feature: sin(theta)
            feedback_speed,  # 1 feature: measured_speed
            feedback_steer,  # 1 feature: measured_steer_angle
            cmd_speed,       # 1 feature: cmd_speed
            cmd_steer        # 1 feature: cmd_steer_angle
        ], axis=1)

        # Compute velocity magnitudes for filtering (using consecutive position differences)
        position_diff = np.diff(position, axis=0)
        time_diff = np.diff(df['timestamp'].values).reshape(-1, 1)
        # Avoid division by zero
        time_diff = np.where(time_diff == 0, 1e-6, time_diff)
        velocity = position_diff / time_diff
        vel_magnitudes = np.sqrt(np.sum(velocity ** 2, axis=1))

        # Pad velocity magnitudes to match state length (first element gets zero velocity)
        vel_magnitudes = np.concatenate([[0], vel_magnitudes])

        # Filter out sequences with low velocity (< 0.1 m/s for reachtruck)
        valid_indices = vel_magnitudes >= 0.0
        print(vel_magnitudes)

        if np.sum(valid_indices) < sequence_length:
            print(f"Skipping {csv_file}: insufficient high-velocity timesteps")
            return None

        state_filtered = state[valid_indices]

        # Create sequences of length (timesteps + past_history_input)
        sequences = arrange_data_sample(state_filtered, sequence_length) # Shape: (num_sequences, sequence_length, 8)

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


def filter_csv_files_by_pattern(csv_files, pattern):
    """
    Filter CSV files based on a shell-style wildcard pattern.

    Args:
        csv_files: List of file paths
        pattern: Shell-style wildcard pattern (e.g., '*', '*_9.csv', 'reachtruck_data_bag_[1-3].csv')

    Returns:
        Filtered list of file paths matching the pattern
    """
    filtered_files = []
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        if fnmatch.fnmatch(basename, pattern):
            filtered_files.append(csv_file)
    return filtered_files


def process_all_csvs_to_tensor(csv_dir, output_path, timesteps=20, past_history_input=2, include_pattern='*'):
    """
    Process selected CSV files in directory and create a single tensor.
    Final tensor shape: (data_size, timesteps + past_history_input, 8)

    Args:
        csv_dir: Directory containing CSV files
        output_path: Path to save the output tensor
        timesteps: Number of timesteps to predict
        past_history_input: Number of past history inputs
        include_pattern: Shell-style pattern to filter CSV files (e.g., '*', '*_9.csv')
    """
    sequence_length = timesteps + past_history_input

    # Get all CSV files
    all_csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    # Filter based on pattern
    csv_files = filter_csv_files_by_pattern(all_csv_files, include_pattern)

    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern '{include_pattern}' in {csv_dir}")

    print(f"Found {len(all_csv_files)} total CSV files in {csv_dir}")
    print(f"Selected {len(csv_files)} CSV files matching pattern '{include_pattern}'")
    print(f"Files to process: {[os.path.basename(f) for f in sorted(csv_files)]}")

    all_sequences = []
    total_sequences = 0

    for i, csv_file in enumerate(sorted(csv_files)):
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")

        sequences = process_csv_to_state_tensor(csv_file, timesteps, past_history_input)
        if sequences is not None:
            print(f"  Generated {sequences.shape[0]} sequences from {os.path.basename(csv_file)}")
            all_sequences.append(sequences)
            total_sequences += sequences.shape[0]
        else:
            print(f"  Skipped {os.path.basename(csv_file)}")

    if not all_sequences:
        raise ValueError("No valid sequences generated from any CSV files")

    # Concatenate all sequences along the sequence dimension
    combined_sequences = torch.cat(all_sequences, dim=0)  # Shape: (data_size, timesteps + past_history_input, 8)

    print(f"\nFinal tensor shape: {combined_sequences.shape}")
    print(f"Total sequences (data_size): {total_sequences}")
    print(f"Expected format: [data_size={total_sequences}, num_steps={sequence_length}, state_dim=7]")
    print(f"  where num_steps = timesteps({timesteps}) + past_history_input({past_history_input}) = {sequence_length}")

    # Save the tensor
    torch.save(combined_sequences, output_path)
    print(f"Tensor saved to: {output_path}")

    # Assert no NaN values
    assert not torch.isnan(combined_sequences).any(), "Final tensor contains NaN values"

    return combined_sequences


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert reachtruck CSV files to position-only tensor format (7 features)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSV files with default settings
  python reachtruck_csv_to_tensor.py --csv_dir ./data/reachtruck_csvs/

  # Process only specific files using pattern
  python reachtruck_csv_to_tensor.py --csv_dir ./data/reachtruck_csvs/ --include_csv_files '*_9.csv'

  # Process files matching a range
  python reachtruck_csv_to_tensor.py --csv_dir ./data/reachtruck_csvs/ --include_csv_files 'reachtruck_data_bag_[1-5].csv'

  # Process with custom timesteps and past history
  python reachtruck_csv_to_tensor.py --csv_dir ./data/reachtruck_csvs/ \\
      --timesteps 30 --past_history_input 3 --output ./data/reachtruck_train.pt
        """
    )
    parser.add_argument('--timesteps', type=int, default=20,
                        help='Number of prediction timesteps (default: 20)')
    parser.add_argument('--past_history_input', type=int, default=2,
                        help='Number of past history inputs (default: 2)')
    parser.add_argument('--csv_dir', type=str, default="./data/reachtruck_csvs/",
                        help='Directory containing reachtruck CSV files (default: ./data/reachtruck_csvs/)')
    parser.add_argument('--output', type=str, default="./data/reachtruck_position_only.pt",
                        help='Output tensor file path (default: ./data/reachtruck_position_only.pt)')
    parser.add_argument('--include_csv_files', type=str, default='*',
                        help='Shell-style pattern to filter CSV files (default: * for all files). '
                             'Examples: "*_9.csv", "reachtruck_data_bag_[1-3].csv"')
    args = parser.parse_args()

    # Configuration
    csv_directory = args.csv_dir
    output_file = args.output
    timesteps = args.timesteps
    past_history_input = args.past_history_input
    include_pattern = args.include_csv_files
    sequence_length = timesteps + past_history_input

    print("=" * 70)
    print("Starting CSV to Position-Only Tensor Conversion for Reachtruck Data")
    print("=" * 70)
    print(f"Input directory: {csv_directory}")
    print(f"Output file: {output_file}")
    print(f"Timesteps to predict: {timesteps}")
    print(f"Past history input: {past_history_input}")
    print(f"Total sequence length: {sequence_length}")
    print(f"CSV file pattern: {include_pattern}")
    print("-" * 70)

    # Process CSV files
    tensor = process_all_csvs_to_tensor(csv_directory, output_file, timesteps, past_history_input, include_pattern)

    print("-" * 70)
    print("Conversion completed successfully!")
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Expected format: [data_size, num_steps, state_dim]")
    print(f"  data_size: {tensor.shape[0]}")
    print(f"  num_steps: {tensor.shape[1]} (timesteps={timesteps} + past_history={past_history_input})")
    print(f"  state_dim: {tensor.shape[2]} = [x(1), y(1), cos_theta(1), sin_theta(1), feedback_speed(1), feedback_steer(1), cmd_speed(1), cmd_steer(1)]")

    # Verify the structure
    print("\nTensor verification (first trajectory, first timestep):")
    print(f"  Position (x, y): {tensor[0, 0, :2]}")
    print(f"  Orientation (cos, sin): {tensor[0, 0, 2:4]}")
    print(f"  Feedback (speed, steer): {tensor[0, 0, 4:6]}")
    print(f"  Control (cmd_speed, cmd_steer): {tensor[0, 0, 6:8]}")

    print("\n" + "=" * 70)
    print("To use this tensor with position_only_train.py:")
    print(f"  python position_only_train.py --custom_data_path {output_file} \\")
    print(f"         --timesteps {timesteps} --past_history_input {past_history_input}")
    print("=" * 70)


if __name__ == "__main__":
    main()
