"""
Reachtruck Plotter for Planar Position-Only PhysORD Model

This script plots the results of planar_physord/planar_position_only_model.py
trained using position_only_train.py on reachtruck datasets.

Displays 6 representative trajectory comparisons:
- Top row: 3 sequences with maximum commanded speed
- Bottom row: 3 sequences with worst prediction error/loss
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
import sys
from typing import Tuple, Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planar_physord.planar_position_only_model import PlanarPositionOnlyPhysORD
from util.utils import normalize_theta


def compute_trajectory_errors(gt_data: torch.Tensor, pred_data: torch.Tensor) -> np.ndarray:
    """
    Compute position and angle errors for each trajectory.

    Args:
        gt_data: Ground truth [batch_size, timesteps, state_dim]
        pred_data: Predictions [batch_size, timesteps, state_dim]

    Returns:
        errors: [batch_size] array of total trajectory errors
    """
    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data
    pred_np = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data

    batch_size = gt_np.shape[0]
    errors = np.zeros(batch_size)

    for i in range(batch_size):
        # Position error (x, y)
        gt_pos = gt_np[i, :, :2]
        pred_pos = pred_np[i, :, :2]
        pos_error = np.mean(np.linalg.norm(gt_pos - pred_pos, axis=1))

        # Angle error (theta)
        gt_theta = gt_np[i, :, 2:3]
        pred_theta = pred_np[i, :, 2:3]
        angle_diff = gt_theta - pred_theta
        # Normalize angle difference to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        angle_error = np.mean(np.abs(angle_diff))

        # Combined error (weighted sum)
        errors[i] = pos_error + 0.5 * angle_error

    return errors


def get_commanded_speeds(data: torch.Tensor) -> np.ndarray:
    """
    Extract commanded speeds from trajectories.

    Args:
        data: [batch_size, timesteps, state_dim] trajectories
              state_dim = [x, y, theta, feedback_speed, feedback_steer, u_speed, u_steer]

    Returns:
        speeds: [batch_size] array of mean commanded speeds
    """
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

    # u_speed is at index 5
    u_speeds = data_np[:, :, 5]
    mean_speeds = np.mean(np.abs(u_speeds), axis=1)

    return mean_speeds


def select_representative_trajectories(
    gt_data: torch.Tensor,
    pred_data: torch.Tensor,
    num_max_speed: int = 3,
    num_worst_error: int = 3
) -> Tuple[List[int], List[int]]:
    """
    Select representative trajectories based on:
    1. Maximum commanded speed
    2. Worst prediction error

    Args:
        gt_data: Ground truth [batch_size, timesteps, state_dim]
        pred_data: Predictions [batch_size, timesteps, state_dim]
        num_max_speed: Number of max speed trajectories to select
        num_worst_error: Number of worst error trajectories to select

    Returns:
        max_speed_indices: Indices of max speed trajectories
        worst_error_indices: Indices of worst error trajectories
    """
    # Compute errors and speeds
    errors = compute_trajectory_errors(gt_data, pred_data)
    speeds = get_commanded_speeds(gt_data)

    # Get max speed trajectories
    max_speed_indices = np.argsort(speeds)[-num_max_speed:][::-1].tolist()

    # Get worst error trajectories
    worst_error_indices = np.argsort(errors)[-num_worst_error:][::-1].tolist()

    return max_speed_indices, worst_error_indices


def calculate_trajectory_stats(trajectory: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a trajectory.

    Args:
        trajectory: [timesteps, state_dim] trajectory

    Returns:
        Dictionary with trajectory statistics
    """
    # Extract commanded speed and steering
    u_speed = trajectory[:, 5]
    u_steer = trajectory[:, 6]

    # Calculate statistics
    mean_cmd_speed = np.mean(np.abs(u_speed))
    max_cmd_speed = np.max(np.abs(u_speed))
    mean_cmd_steer = np.mean(np.abs(u_steer))

    return {
        'mean_cmd_speed': mean_cmd_speed,
        'max_cmd_speed': max_cmd_speed,
        'mean_cmd_steer': mean_cmd_steer
    }


def plot_trajectory_comparison(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    ax: plt.Axes,
    title: str,
    stats: Dict[str, float],
    error: float
) -> None:
    """
    Plot comparison between ground truth and predicted trajectory.

    Args:
        gt_traj: Ground truth trajectory [timesteps, state_dim]
        pred_traj: Predicted trajectory [timesteps, state_dim]
        ax: Matplotlib axes to plot on
        title: Plot title
        stats: Trajectory statistics
        error: Prediction error for this trajectory
    """
    # Extract positions (x, y)
    gt_pos = gt_traj[:, :2]
    pred_pos = pred_traj[:, :2]

    # Plot trajectories
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], 'o-', color='green', linewidth=2.5,
            markersize=4, label='Ground Truth', alpha=0.85)
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], 's-', color='orange', linewidth=2.5,
            markersize=4, label='PhysORD Prediction', alpha=0.85)

    # Mark start and end points
    ax.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', markersize=10,
            label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(gt_pos[-1, 0], gt_pos[-1, 1], 'rs', markersize=10,
            label='End', markeredgecolor='darkred', markeredgewidth=2)

    # Calculate trajectory bounds
    all_pos = np.vstack([gt_pos, pred_pos])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()

    # Add padding and ensure same width/height
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range, 0.1)  # Minimum range of 0.1

    # Add 15% padding
    padding = max_range * 0.15
    max_range_padded = max_range + 2 * padding

    # Center the plot
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    ax.set_xlim(x_center - max_range_padded/2, x_center + max_range_padded/2)
    ax.set_ylim(y_center - max_range_padded/2, y_center + max_range_padded/2)

    # Add statistics text
    stats_text = (f"Cmd Speed: {stats['mean_cmd_speed']:.2f} (max: {stats['max_cmd_speed']:.2f})\n"
                  f"Cmd Steer: {stats['mean_cmd_steer']:.3f}\n"
                  f"Error: {error:.4f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontsize=10)
    ax.set_ylabel('Y Position (m)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)


def create_comprehensive_plot(
    gt_data: torch.Tensor,
    pred_data: torch.Tensor,
    save_path: str = 'reachtruck_comparison.png',
    figsize: Tuple[int, int] = (18, 10)
) -> None:
    """
    Create comprehensive plot with 6 representative trajectories.

    Args:
        gt_data: Ground truth [batch_size, timesteps, state_dim]
        pred_data: Predictions [batch_size, timesteps, state_dim]
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Select representative trajectories
    max_speed_indices, worst_error_indices = select_representative_trajectories(
        gt_data, pred_data, num_max_speed=3, num_worst_error=3
    )

    # Compute errors for all trajectories
    errors = compute_trajectory_errors(gt_data, pred_data)

    # Convert to numpy
    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data
    pred_np = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data

    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Reachtruck Position-Only PhysORD: Trajectory Predictions',
                 fontsize=16, fontweight='bold', y=0.98)

    # Top row: Max commanded speed trajectories
    for j, idx in enumerate(max_speed_indices):
        gt_traj = gt_np[idx, :, :]
        pred_traj = pred_np[idx, :, :]
        stats = calculate_trajectory_stats(gt_traj)
        error = errors[idx]

        plot_trajectory_comparison(
            gt_traj, pred_traj, axes[0, j],
            f'Max Speed #{j+1} (Traj {idx})',
            stats, error
        )

    # Bottom row: Worst error trajectories
    for j, idx in enumerate(worst_error_indices):
        gt_traj = gt_np[idx, :, :]
        pred_traj = pred_np[idx, :, :]
        stats = calculate_trajectory_stats(gt_traj)
        error = errors[idx]

        plot_trajectory_comparison(
            gt_traj, pred_traj, axes[1, j],
            f'Worst Error #{j+1} (Traj {idx})',
            stats, error
        )

    # Add row labels
    fig.text(0.02, 0.75, 'Max Commanded\nSpeed', fontsize=13, fontweight='bold',
             va='center', ha='center', rotation=90)
    fig.text(0.02, 0.28, 'Worst\nPrediction Error', fontsize=13, fontweight='bold',
             va='center', ha='center', rotation=90)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(labels), fontsize=11, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot reachtruck trajectory predictions from PlanarPositionOnlyPhysORD'
    )

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.tar file)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data file (.pt file)')

    # Model configuration
    parser.add_argument('--timesteps', type=int, default=20,
                       help='Number of timesteps to predict (default: 20)')
    parser.add_argument('--past_history_input', type=int, default=2,
                       help='Number of past history inputs (default: 2)')
    parser.add_argument('--control_dim', type=int, default=2,
                       help='Control input dimension (default: 2)')
    parser.add_argument('--time_step', type=float, default=0.1,
                       help='Time step for model integration (default: 0.1)')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden layer size for force model (default: 64)')
    parser.add_argument('--use_feedback', action='store_true', default=True,
                       help='Use feedback measurements (default: True)')
    parser.add_argument('--no_feedback', action='store_false', dest='use_feedback',
                       help='Disable feedback measurements')

    # Output configuration
    parser.add_argument('--save_dir', type=str, default='./plots/',
                       help='Directory to save plots (default: ./plots/)')
    parser.add_argument('--save_name', type=str, default='reachtruck_comparison.png',
                       help='Output filename (default: reachtruck_comparison.png)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[18, 10],
                       help='Figure size in inches [width height] (default: 18 10)')

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run evaluation on (default: cuda:0)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plot, only save it')

    # Data selection
    parser.add_argument('--use_val_split', action='store_true',
                       help='Use validation split of data (last 20%% of trajectories)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Maximum number of trajectories to evaluate (default: all)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = PlanarPositionOnlyPhysORD(
        device=device,
        udim=args.control_dim,
        time_step=args.time_step,
        past_history_input=args.past_history_input,
        hidden_size=args.hidden_size,
        use_feedback=args.use_feedback
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")

    # Load test data
    print(f"Loading test data from: {args.test_data}")
    test_data = torch.load(args.test_data, map_location=device)
    test_data = test_data[3000:4000:, :, :]

    # Handle different data formats
    if isinstance(test_data, dict):
        if 'val_data' in test_data:
            test_data = test_data['val_data']
        elif 'test_data' in test_data:
            test_data = test_data['test_data']
        elif 'data' in test_data:
            test_data = test_data['data']
        else:
            test_data = list(test_data.values())[0]

    print(f"Test data shape: {test_data.shape}")

    # Use validation split if requested
    if args.use_val_split:
        val_split_idx = int(0.8 * test_data.shape[0])
        test_data = test_data[val_split_idx:]
        print(f"Using validation split: {test_data.shape}")

    # Limit number of trajectories if requested
    if args.max_trajectories is not None and args.max_trajectories < test_data.shape[0]:
        test_data = test_data[:args.max_trajectories]
        print(f"Limited to {args.max_trajectories} trajectories")

    # Ensure data is in float64 format
    test_data = test_data.to(dtype=torch.float64, device=device)

    # Transpose data from [batch_size, timesteps, state_dim] to model format
    # Model expects [batch_size, timesteps, state_dim]
    print(f"Data shape before evaluation: {test_data.shape}")

    # Override plt.show if no_show is set
    if args.no_show:
        import matplotlib
        matplotlib.pyplot.show = lambda: None

    # Evaluate model
    print("Evaluating model...")
    with torch.no_grad():
        # Generate predictions
        # pred_data_transposed = model.evaluation(args.timesteps, test_data_transposed)
        pred_data = model.evaluation(args.timesteps, test_data)

        # Get ground truth (excluding past history)
        gt_data = test_data[:, args.past_history_input:, :]
        pred_data = pred_data[:, args.past_history_input:, :]

    print(f"Ground truth shape: {gt_data.shape}")
    print(f"Predictions shape: {pred_data.shape}")

    # Compute overall statistics
    errors = compute_trajectory_errors(gt_data, pred_data)
    speeds = get_commanded_speeds(test_data)

    print(f"\nDataset Statistics:")
    print(f"  Mean prediction error: {np.mean(errors):.4f}")
    print(f"  Std prediction error: {np.std(errors):.4f}")
    print(f"  Max prediction error: {np.max(errors):.4f}")
    print(f"  Mean commanded speed: {np.mean(speeds):.4f}")
    print(f"  Max commanded speed: {np.max(speeds):.4f}")

    # Create plot
    print("\nCreating comparison plot...")
    save_path = os.path.join(args.save_dir, args.save_name)
    create_comprehensive_plot(
        gt_data, pred_data,
        save_path=save_path,
        figsize=tuple(args.figsize)
    )

    print(f"\nDone! Plot saved to: {save_path}")
