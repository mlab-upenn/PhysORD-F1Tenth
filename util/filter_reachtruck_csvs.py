#!/usr/bin/env python3
"""
Filter reachtruck CSV files by splitting them when timestamp gaps exceed 0.25 seconds.

Usage:
    python filter_reachtruck_csvs.py <input_folder> <output_folder>

Args:
    input_folder: Directory containing unfiltered CSV files
    output_folder: Directory where filtered CSV files will be created

The script will split CSV files at points where consecutive timestamps differ by more than 0.25 seconds.
If a CSV has n splits, it will produce n+1 output files.
"""

import argparse
import os
from pathlib import Path
import pandas as pd


def filter_csv(input_file: Path, output_folder: Path, max_time_gap: float = 0.25):
    """
    Filter a single CSV file by splitting it based on timestamp gaps.

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

    # Calculate time differences between consecutive rows
    time_diffs = df['timestamp'].diff()

    # Find indices where the time gap exceeds the threshold
    split_indices = time_diffs[time_diffs > max_time_gap].index.tolist()

    # Create segments based on split points
    segments = []
    start_idx = 0

    for split_idx in split_indices:
        # Add segment from start_idx to split_idx (exclusive)
        if start_idx < split_idx:
            segments.append(df.iloc[start_idx:split_idx])
        start_idx = split_idx

    # Add the last segment
    if start_idx < len(df):
        segments.append(df.iloc[start_idx:])

    # If no splits were found, add the entire dataframe as one segment
    if not segments:
        segments.append(df)

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
        description='Filter reachtruck CSV files by splitting on timestamp gaps > 0.25s'
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
        default=0.25,
        help='Maximum allowed time gap in seconds (default: 0.25)'
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
