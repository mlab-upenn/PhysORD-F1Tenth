import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple

def detect_motion_cases(velocities: np.ndarray, threshold: float = 0.1) -> str:
    """
    Detect motion case based on velocity changes.

    Args:
        velocities: Array of velocity magnitudes over time [timesteps]
        threshold: Threshold for detecting acceleration/deceleration

    Returns:
        Motion case: 'accelerating', 'decelerating', or 'uniform'
    """
    if len(velocities) < 2:
        return 'uniform'

    # Calculate velocity changes
    vel_changes = np.diff(velocities)
    mean_change = np.mean(vel_changes)

    dt = 0.1  # time step
    if mean_change/dt > threshold:
        return 'accelerating'
    elif mean_change/dt < -threshold:
        return 'decelerating'
    else:
        return 'uniform'

def detect_path_cases(trajectory: np.ndarray, threshold_straight: float = 0.05) -> str:
    """
    Detect path case based on angular velocities.

    Args:
        trajectory: Array of trajectory states over time [timesteps, features]
        threshold_straight: Threshold for detecting straight paths

    Returns:
        Path case: 'straight', 'slight_turn', or 'continuous_turn'
    """
    if len(trajectory) < 3:
        return 'straight'

    # Determine if 2D or 3D based on trajectory features
    if trajectory.shape[1] >= 18:  # 3D case
        angular_velocities = trajectory[:, 15:18]  # omega_x, omega_y, omega_z
        angular_speeds = np.linalg.norm(angular_velocities, axis=1)
    else:  # 2D case
        angular_speeds = np.abs(trajectory[:, 6])  # omega_z only

    mean_angular_speed = np.mean(angular_speeds)

    # Check angular speed magnitude
    if mean_angular_speed < threshold_straight:
        return 'straight'
    elif mean_angular_speed < 0.5:  # Moderate turning
        return 'slight_turn'
    else:
        return 'continuous_turn'

def calculate_trajectory_stats(trajectory: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a trajectory.

    Args:
        trajectory: Array of positions/states over time [timesteps, features]

    Returns:
        Dictionary with speed and acceleration statistics
    """
    if len(trajectory.shape) == 3:
        # Handle batch dimension [timesteps, batch, features]
        trajectory = trajectory.mean(axis=1)  # Average over batch

    # Use proper velocity indices based on trajectory type
    if trajectory.shape[1] >= 15:  # 3D case
        velocities = trajectory[:, 12:15]  # vx, vy, vz
    else:  # 2D case
        velocities = trajectory[:, 4:6]  # vx, vy

    speeds = np.linalg.norm(velocities, axis=1)

    # Calculate accelerations from velocity changes
    dt = 0.1  # time step
    accelerations = np.diff(velocities, axis=0) / dt
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)

    # Determine motion case and calculate signed acceleration for decelerating cases
    motion_case = detect_motion_cases(speeds)
    mean_acceleration = np.mean(accel_magnitudes)

    if motion_case == 'decelerating':
        # For decelerating trajectories, show negative acceleration
        mean_acceleration = -mean_acceleration

    return {
        'mean_speed': np.mean(speeds),
        'mean_acceleration': mean_acceleration
    }

def classify_trajectory(trajectory: np.ndarray) -> Tuple[str, str]:
    """
    Classify a trajectory into motion and path cases.

    Args:
        trajectory: Array of positions/states over time [timesteps, features] or [timesteps, batch, features]

    Returns:
        Tuple of (motion_case, path_case)
    """
    if len(trajectory.shape) == 3:
        # Handle batch dimension - use first trajectory for classification
        traj = trajectory[:, 0, :]
    else:
        traj = trajectory

    # Use proper velocity indices based on trajectory type
    if traj.shape[1] >= 15:  # 3D case
        velocities = traj[:, 12:15]  # vx, vy, vz
    else:  # 2D case
        velocities = traj[:, 4:6]  # vx, vy

    speeds = np.linalg.norm(velocities, axis=1)

    motion_case = detect_motion_cases(speeds)
    path_case = detect_path_cases(traj)

    return motion_case, path_case

def find_representative_trajectories(gt_data: torch.Tensor, pred_data: torch.Tensor,
                                   num_samples: int = 1000) -> Dict[str, Dict[str, list]]:
    """
    Find representative trajectories for each motion and path case combination.

    Args:
        gt_data: Ground truth trajectories [timesteps, batch, features]
        pred_data: Predicted trajectories [timesteps, batch, features]
        num_samples: Number of trajectories to sample for classification

    Returns:
        Dictionary mapping case combinations to trajectory indices
    """
    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data

    # Sample trajectories for efficiency
    batch_size = gt_np.shape[1]
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)

    case_trajectories = {
        'accelerating': {'straight': [], 'slight_turn': [], 'continuous_turn': []},
        'uniform': {'straight': [], 'slight_turn': [], 'continuous_turn': []},
        'decelerating': {'straight': [], 'slight_turn': [], 'continuous_turn': []}
    }

    for idx in sample_indices:
        traj = gt_np[:, idx, :]

        # Calculate mean speed and filter out slow trajectories
        if traj.shape[1] >= 15:  # 3D case
            velocities = traj[:, 12:15]  # vx, vy, vz
        else:  # 2D case
            velocities = traj[:, 4:6]  # vx, vy

        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speeds)

        # Skip trajectories with mean speed less than 0.9 m/s
        if mean_speed < 0.9:
            continue

        motion_case, path_case = classify_trajectory(traj)

        if len(case_trajectories[motion_case][path_case]) < 3:  # Limit to 3 examples per case
            case_trajectories[motion_case][path_case].append(idx)

    return case_trajectories

def plot_trajectory_comparison(gt_traj: np.ndarray, pred_traj: np.ndarray,
                             motion_case: str, path_case: str,
                             ax: plt.Axes, stats: Dict[str, float]) -> None:
    """
    Plot comparison between ground truth and predicted trajectory.

    Args:
        gt_traj: Ground truth trajectory [timesteps, features]
        pred_traj: Predicted trajectory [timesteps, features]
        motion_case: Motion classification
        path_case: Path classification
        ax: Matplotlib axes to plot on
        stats: Trajectory statistics
    """
    # Extract positions (x, y for 2D visualization)
    gt_pos = gt_traj[:, :2]
    pred_pos = pred_traj[:, :2]

    # Plot trajectories
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], 'o-', color='green', linewidth=2,
            markersize=3, label='Ground Truth', alpha=0.8)
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], 's-', color='orange', linewidth=2,
            markersize=3, label='PhysORD', alpha=0.8)

    # Mark start and end points
    ax.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', markersize=8, label='Start')
    ax.plot(gt_pos[-1, 0], gt_pos[-1, 1], 'rs', markersize=8, label='End')

    # Calculate trajectory bounds to scale appropriately
    all_pos = np.vstack([gt_pos, pred_pos])
    x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
    y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()

    # Add padding and ensure same width/height by using the larger range
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    # Add 10% padding
    padding = max_range * 0.1
    max_range_padded = max_range + 2 * padding

    # Center the plot and use same width/height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    ax.set_xlim(x_center - max_range_padded/2, x_center + max_range_padded/2)
    ax.set_ylim(y_center - max_range_padded/2, y_center + max_range_padded/2)

    # Add statistics text
    stats_text = f"Speed={stats['mean_speed']:.2f}m/s\nAccel={stats['mean_acceleration']:.2f}m/s²"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

def create_comprehensive_plot(gt_data: torch.Tensor, pred_data: torch.Tensor,
                            save_path: str = 'trajectory_comparison.png',
                            figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Create comprehensive plot comparing ground truth and predicted trajectories
    across different motion and path cases.

    Args:
        gt_data: Ground truth trajectories [timesteps, batch, features]
        pred_data: Predicted trajectories [timesteps, batch, features]
        save_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Find representative trajectories
    case_trajectories = find_representative_trajectories(gt_data, pred_data)

    # Create subplot grid
    motion_cases = ['accelerating', 'uniform', 'decelerating']
    path_cases = ['straight', 'slight_turn', 'continuous_turn']

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('Ground Truth vs PhysORD Predictions: Motion and Path Analysis', fontsize=16)

    # Column titles
    for j, path_case in enumerate(path_cases):
        title = path_case.replace('_', ' ').title()
        axes[0, j].set_title(title, fontsize=14, fontweight='bold')

    # Row labels
    for i, motion_case in enumerate(motion_cases):
        title = motion_case.title()
        axes[i, 0].set_ylabel(title, fontsize=14, fontweight='bold', rotation=90)

    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data
    pred_np = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data

    for i, motion_case in enumerate(motion_cases):
        for j, path_case in enumerate(path_cases):
            ax = axes[i, j]

            # Get representative trajectory for this case
            traj_indices = case_trajectories[motion_case][path_case]

            if traj_indices:
                # Use first available trajectory
                idx = traj_indices[0]
                gt_traj = gt_np[:, idx, :]
                pred_traj = pred_np[:, idx, :]

                # Calculate stats
                stats = calculate_trajectory_stats(gt_traj)

                # Plot comparison
                plot_trajectory_comparison(gt_traj, pred_traj, motion_case, path_case, ax, stats)

            else:
                # No trajectory found for this case
                ax.text(0.5, 0.5, 'No data\navailable', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, alpha=0.6)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(labels), fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from physord.model import PhysORD
    from planar_physord.planar_model import PlanarPhysORD

    parser = argparse.ArgumentParser(description='Evaluate model and plot ground truth vs predicted trajectories')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.tar)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data file (.pt)')
    parser.add_argument('--model_type', type=str, choices=['2d', '3d'], default='3d', help='Model type: 2d (planar) or 3d')
    parser.add_argument('--timesteps', type=int, default=20, help='Number of timesteps to predict')
    parser.add_argument('--save_dir', type=str, default='./plots/', help='Directory to save plots')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run evaluation on')
    parser.add_argument('--use_v_gap', action='store_true', help='Whether model uses v_gap (RPM difference)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[16, 12], help='Figure size (width height)')
    parser.add_argument('--no_show', action='store_true', help='Do not display plots, only save them')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model_type} model from: {args.model_path}")
    if args.model_type == '2d':
        model = PlanarPhysORD(device=device, time_step=0.1, udim=3, use_v_gap=args.use_v_gap).to(device)
    else:
        model = PhysORD(device=device, use_dVNet=True, time_step=0.1, udim=3, use_v_gap=args.use_v_gap).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load test data
    print(f"Loading test data from: {args.test_data}")
    test_data = torch.load(args.test_data, map_location=device)

    # Handle different data formats
    if isinstance(test_data, dict):
        if 'val_data' in test_data:
            test_data = test_data['val_data']
        elif 'test_data' in test_data:
            test_data = test_data['test_data']
        elif 'data' in test_data:
            test_data = test_data['data']
        else:
            # Take the first tensor value if it's a dict
            test_data = list(test_data.values())[0]

    print(f"Test data shape: {test_data.shape}")

    # Ensure data is in float64 format for model compatibility
    test_data = test_data.to(dtype=torch.float64, device=device)

    # Override plt.show if no_show is set
    if args.no_show:
        import matplotlib
        matplotlib.pyplot.show = lambda: None

    # Evaluate model and generate plots
    with torch.no_grad():
        # Get ground truth
        gt_data = test_data[:args.timesteps+1]  # Include initial condition

        # Generate predictions using model evaluation
        print("Evaluating model...")
        pred_data = model.evaluation(args.timesteps, test_data)

    print(f"Ground truth shape: {gt_data.shape}")
    print(f"Predictions shape: {pred_data.shape}")

    # Generate comprehensive plot
    print("Creating comprehensive analysis plot...")
    save_path = os.path.join(args.save_dir, 'comprehensive_analysis.png')
    create_comprehensive_plot(gt_data, pred_data, save_path=save_path, figsize=tuple(args.figsize))
    print(f"Comprehensive plot saved to: {save_path}")
    print(f"All plots saved to: {args.save_dir}")