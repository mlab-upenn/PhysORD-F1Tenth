#!/usr/bin/env python3
"""
Filter reachtruck CSV files by splitting them based on two criteria:
1. Invalid speed conditions: When measured_speed is 0.0 but cmd_speed is not 0.0
   (removes those datapoints plus 10 before and 10 after)
2. Timestamp gaps exceeding 0.25 seconds

Usage:
    python filter_reachtruck_csvs.py <input_folder> <output_folder>

Args:
    input_folder: Directory containing unfiltered CSV files
    output_folder: Directory where filtered CSV files will be created

The script will:
1. First remove continuous intervals where measured_speed=0.0 but cmd_speed!=0.0,
   along with 10 datapoints before and after each such interval
2. Then split the resulting segments when consecutive timestamps differ by more than 0.25 seconds
3. Skip segments with less than 45 rows

This can produce multiple output files from a single input CSV.
"""

import argparse
import os
from pathlib import Path
import pandas as pd


def identify_invalid_speed_indices(df: pd.DataFrame, context_window: int = 20):
    """
    Identify indices to remove where measured_speed is 0.0 but cmd_speed is not 0.0.
    Also marks context_window datapoints before and after each such occurrence.

    Args:
        df: DataFrame with 'measured_speed' and 'cmd_speed' columns
        context_window: Number of datapoints to remove before and after (default: 10)

    Returns:
        Set of indices to remove
    """
    indices_to_remove = set()

    # Check if required columns exist
    if 'measured_speed' not in df.columns or 'cmd_speed' not in df.columns:
        return indices_to_remove

    # Find rows where measured_speed is 0.0 but cmd_speed is not 0.0
    invalid_mask = (df['measured_speed'] == 0.0) & (df['cmd_speed'] != 0.0)
    invalid_indices = df.index[invalid_mask].tolist()

    # For each invalid index, mark it and surrounding context
    for idx in invalid_indices:
        # Add the index itself
        indices_to_remove.add(idx)

        # Add previous context_window indices
        for i in range(max(0, idx - context_window), idx):
            indices_to_remove.add(i)

        # Add next context_window indices
        for i in range(idx + 1, min(len(df), idx + context_window + 1)):
            indices_to_remove.add(i)

    return indices_to_remove


def split_dataframe_by_valid_indices(df: pd.DataFrame, indices_to_remove: set):
    """
    Split dataframe into segments by removing specified indices.

    Args:
        df: Input dataframe
        indices_to_remove: Set of indices to remove

    Returns:
        List of dataframe segments
    """
    if not indices_to_remove:
        return [df]

    # Create a mask for valid rows
    valid_mask = ~df.index.isin(indices_to_remove)

    # Find groups of consecutive True values
    segments = []
    start_idx = None

    for i, is_valid in enumerate(valid_mask):
        if is_valid and start_idx is None:
            # Start of a new segment
            start_idx = i
        elif not is_valid and start_idx is not None:
            # End of a segment
            segments.append(df.iloc[start_idx:i])
            start_idx = None

    # Add the last segment if it exists
    if start_idx is not None:
        segments.append(df.iloc[start_idx:])

    return segments if segments else []


def filter_csv(input_file: Path, output_folder: Path, max_time_gap: float = 0.25):
    """
    Filter a single CSV file by splitting it based on:
    1. Invalid speed intervals (measured_speed=0 but cmd_speed!=0)
    2. Timestamp gaps

    Args:
        input_file: Path to the input CSV file
        output_folder: Path to the output directory
        max_time_gap: Maximum allowed time gap in seconds (default: 0.25)

    Returns:
        Number of output files created
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    if len(df) == 0:
        print(f"Warning: {input_file.name} is empty, skipping.")
        return 0

    if 'timestamp' not in df.columns:
        print(f"Warning: {input_file.name} does not have a 'timestamp' column, skipping.")
        return 0

    # Step 1: Filter out invalid speed intervals
    indices_to_remove = identify_invalid_speed_indices(df, context_window=20)

    if indices_to_remove:
        print(f"  Found {len(indices_to_remove)} datapoints to remove due to invalid speed conditions")
        initial_segments = split_dataframe_by_valid_indices(df, indices_to_remove)
    else:
        initial_segments = [df]

    # Step 2: Apply timestamp gap filtering to each segment
    segments = []
    for segment in initial_segments:
        if len(segment) == 0:
            continue

        # Reset index for proper indexing
        segment = segment.reset_index(drop=True)

        # Calculate time differences between consecutive rows
        time_diffs = segment['timestamp'].diff()

        # Find indices where the time gap exceeds the threshold
        split_indices = time_diffs[time_diffs > max_time_gap].index.tolist()

        # Create sub-segments based on split points
        start_idx = 0
        for split_idx in split_indices:
            # Add segment from start_idx to split_idx (exclusive)
            if start_idx < split_idx:
                segments.append(segment.iloc[start_idx:split_idx])
            start_idx = split_idx

        # Add the last sub-segment
        if start_idx < len(segment):
            segments.append(segment.iloc[start_idx:])

    # If no segments were created, something went wrong
    if not segments:
        print(f"Warning: No valid segments after filtering {input_file.name}")
        return 0

    # Write each segment to a separate file
    base_name = input_file.stem  # filename without extension
    files_created = 0

    for i, segment in enumerate(segments):
        if len(segment) == 0:
            continue

        # Skip segments with less than 45 rows
        if len(segment) < 45:
            time_start = segment['timestamp'].iloc[0]
            time_end = segment['timestamp'].iloc[-1]
            duration = time_end - time_start
            print(f"  Skipping segment {i+1}: only {len(segment)} rows (< 45), duration: {duration:.2f}s")
            continue

        # Create output filename
        if len(segments) == 1:
            # No splits, keep original name
            output_file = output_folder / f"{base_name}.csv"
        else:
            # Multiple segments, add suffix
            output_file = output_folder / f"{base_name}_part{i+1}.csv"

        # Write segment to CSV
        segment.to_csv(output_file, index=False)
        files_created += 1

        # Calculate time range for this segment
        time_start = segment['timestamp'].iloc[0]
        time_end = segment['timestamp'].iloc[-1]
        duration = time_end - time_start

        print(f"  Created {output_file.name}: {len(segment)} rows, duration: {duration:.2f}s")

    return files_created


def main():
    """Main function to process all CSV files in the input folder."""
    parser = argparse.ArgumentParser(
        description='Filter reachtruck CSV files by removing invalid speed intervals and splitting on timestamp gaps'
    )
    parser.add_argument(
        'input_folder',
        type=str,
        help='Directory containing unfiltered CSV files'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Directory where filtered CSV files will be created'
    )
    parser.add_argument(
        '--max-gap',
        type=float,
        default=0.2,
        help='Maximum allowed time gap in seconds (default: 0.2)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    # Validate input folder
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return 1

    if not input_folder.is_dir():
        print(f"Error: Input path '{input_folder}' is not a directory.")
        return 1

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in the input folder
    csv_files = list(input_folder.glob('*.csv'))

    if not csv_files:
        print(f"Warning: No CSV files found in '{input_folder}'")
        return 0

    print(f"Processing {len(csv_files)} CSV files from '{input_folder}'")
    print(f"Output directory: '{output_folder}'")
    print(f"Maximum time gap: {args.max_gap}s")
    print("-" * 70)

    # Process each CSV file
    total_input_files = 0
    total_output_files = 0

    for csv_file in sorted(csv_files):
        print(f"\nProcessing {csv_file.name}...")
        total_input_files += 1

        try:
            files_created = filter_csv(csv_file, output_folder, args.max_gap)
            total_output_files += files_created
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")

    print("\n" + "=" * 70)
    print(f"Summary: Processed {total_input_files} input files")
    print(f"         Created {total_output_files} output files")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
