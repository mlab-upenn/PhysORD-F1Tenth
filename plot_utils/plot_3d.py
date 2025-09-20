import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Any

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

    if mean_change > threshold:
        return 'accelerating'
    elif mean_change < -threshold:
        return 'decelerating'
    else:
        return 'uniform'

def detect_path_cases(positions: np.ndarray, threshold_straight: float = 0.05,
                     threshold_oscillation: float = 0.3) -> str:
    """
    Detect path case based on trajectory curvature and patterns.

    Args:
        positions: Array of positions over time [timesteps, 2 or 3]
        threshold_straight: Threshold for detecting straight paths
        threshold_oscillation: Threshold for detecting oscillations

    Returns:
        Path case: 'straight', 'slight_turn', 'continuous_turn', or 'oscillating'
    """
    if len(positions) < 3:
        return 'straight'

    # Calculate path curvature using consecutive points
    directions = np.diff(positions, axis=0)
    if len(directions) < 2:
        return 'straight'

    # Normalize directions
    direction_norms = np.linalg.norm(directions, axis=1)
    direction_norms[direction_norms == 0] = 1  # Avoid division by zero
    normalized_directions = directions / direction_norms[:, np.newaxis]

    # Calculate angles between consecutive direction vectors
    angles = []
    for i in range(len(normalized_directions) - 1):
        dot_product = np.clip(np.dot(normalized_directions[i], normalized_directions[i+1]), -1, 1)
        angle = np.arccos(dot_product)
        angles.append(angle)

    angles = np.array(angles)
    mean_curvature = np.mean(angles)
    curvature_std = np.std(angles)

    # Check for oscillations (high variance in angles)
    if curvature_std > threshold_oscillation:
        return 'oscillating'

    # Check curvature magnitude
    if mean_curvature < threshold_straight:
        return 'straight'
    elif mean_curvature < 0.2:  # Moderate turning
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

    positions = trajectory[:, :3]  # x, y, z positions

    # Calculate velocities
    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)

    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0)
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)

    return {
        'mean_speed': np.mean(speeds),
        'mean_acceleration': np.mean(accel_magnitudes)
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

    positions = traj[:, :3]  # x, y, z positions

    # Calculate velocities for motion classification
    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)

    motion_case = detect_motion_cases(speeds)
    path_case = detect_path_cases(positions)

    return motion_case, path_case

def find_representative_trajectories(gt_data: torch.Tensor, pred_data: torch.Tensor,
                                   num_samples: int = 100) -> Dict[str, Dict[str, List[int]]]:
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
    pred_np = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data

    # Sample trajectories for efficiency
    batch_size = gt_np.shape[1]
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)

    case_trajectories = {
        'accelerating': {'straight': [], 'slight_turn': [], 'continuous_turn': [], 'oscillating': []},
        'uniform': {'straight': [], 'slight_turn': [], 'continuous_turn': [], 'oscillating': []},
        'decelerating': {'straight': [], 'slight_turn': [], 'continuous_turn': [], 'oscillating': []}
    }

    for idx in sample_indices:
        traj = gt_np[:, idx, :]
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

    # Add statistics text
    stats_text = f"Speed={stats['mean_speed']:.2f}m/s\nAccel={stats['mean_acceleration']:.2f}m/s�"
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
    path_cases = ['straight', 'slight_turn', 'continuous_turn', 'oscillating']

    fig, axes = plt.subplots(3, 4, figsize=figsize)
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

    print(case_trajectories)

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

def plot_single_case(gt_data: torch.Tensor, pred_data: torch.Tensor,
                    motion_case: str, path_case: str,
                    save_path: str = None) -> None:
    """
    Plot a single motion/path case comparison.

    Args:
        gt_data: Ground truth trajectories [timesteps, batch, features]
        pred_data: Predicted trajectories [timesteps, batch, features]
        motion_case: Motion case to plot ('accelerating', 'uniform', 'decelerating')
        path_case: Path case to plot ('straight', 'slight_turn', 'continuous_turn', 'oscillating')
        save_path: Optional path to save the plot
    """
    case_trajectories = find_representative_trajectories(gt_data, pred_data)

    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data
    pred_np = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data

    traj_indices = case_trajectories[motion_case][path_case]

    if not traj_indices:
        print(f"No trajectories found for {motion_case} + {path_case} case")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot multiple examples if available
    for i, idx in enumerate(traj_indices[:3]):  # Max 3 examples
        gt_traj = gt_np[:, idx, :]
        pred_traj = pred_np[:, idx, :]
        stats = calculate_trajectory_stats(gt_traj)

        plot_trajectory_comparison(gt_traj, pred_traj, motion_case, path_case, ax, stats)

    title = f"{motion_case.title()} + {path_case.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=14)
    ax.legend()

    if save_path:
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
    parser.add_argument('--plot_type', type=str, choices=['comprehensive', 'individual', 'both'], default='comprehensive',
                       help='Type of plots to generate')

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

    # Generate plots based on type
    if args.plot_type in ['comprehensive', 'both']:
        print("Creating comprehensive analysis plot...")
        save_path = os.path.join(args.save_dir, 'comprehensive_analysis.png')
        create_comprehensive_plot(gt_data, pred_data, save_path=save_path, figsize=tuple(args.figsize))
        print(f"Comprehensive plot saved to: {save_path}")

    if args.plot_type in ['individual', 'both']:
        print("Creating individual plots for all cases...")
        motion_cases = ['accelerating', 'uniform', 'decelerating']
        path_cases = ['straight', 'slight_turn', 'continuous_turn', 'oscillating']

        for motion in motion_cases:
            for path in path_cases:
                filename = f'{motion}_{path}.png'
                save_path = os.path.join(args.save_dir, filename)
                plot_single_case(gt_data, pred_data, motion, path, save_path=save_path)
                print(f"Individual plot saved to: {save_path}")

    print(f"All plots saved to: {args.save_dir}")