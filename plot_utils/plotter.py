import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple

def detect_motion_cases(velocities: np.ndarray, threshold: float = 2.5, dt: float = 0.1) -> str:
    """
    Detect motion case based on velocity changes.

    Args:
        velocities: Array of velocity vectors over time [timesteps, (vx, vy)]
        threshold: Threshold for detecting acceleration.
        dt: Time step for calculating acceleration

    Returns:
        Motion case: 'accelerating', or 'uniform'
    """
    if len(velocities) < 2:
        return 'uniform'

    # Calculate accelerations from velocity changes
    accelerations = np.diff(velocities, axis=0) / dt
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)

    # Determine motion case based on mean acceleration
    mean_acceleration = np.mean(accel_magnitudes)
    print(f"Mean acceleration: {mean_acceleration:.4f} m/s²")

    if mean_acceleration < threshold:
        return 'uniform'
    else:
        return "accelerating"


def detect_path_cases(trajectory: np.ndarray, threshold_straight: float = 0.2) -> str:
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
        angular_velocities = trajectory[:, 17]  # omega_z
        angular_speeds = np.abs(angular_velocities)
    else:  # 2D case
        angular_speeds = np.abs(trajectory[:, 6])  # omega_z only

    mean_angular_speed = np.mean(angular_speeds)
    print(f"Mean angular speed: {mean_angular_speed:.4f} rad/s")

    # Check angular speed magnitude
    if mean_angular_speed < threshold_straight:
        return 'straight'
    elif mean_angular_speed < 0.35:  # Moderate turning
        return 'slight_turn'
    else:
        return 'continuous_turn'

def calculate_trajectory_stats(trajectory: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """
    Calculate statistics for a trajectory.

    Args:
        trajectory: Array of positions/states over time [timesteps, features]
        dt: Time step for calculating acceleration

    Returns:
        Dictionary with speed and acceleration statistics
    """
    if len(trajectory.shape) == 3:
        # Handle batch dimension [timesteps, batch, features]
        trajectory = trajectory.mean(axis=1)  # Average over batch

    # Use proper velocity indices based on trajectory type
    if trajectory.shape[1] >= 15:  # 3D case
        velocities = trajectory[:, 12:14]  # vx, vy, vz
    else:  # 2D case
        velocities = trajectory[:, 4:6]  # vx, vy

    # Calculate accelerations from velocity changes
    accelerations = np.diff(velocities, axis=0) / dt
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)

    # Determine motion case and calculate signed acceleration for decelerating cases
    mean_acceleration = np.mean(accel_magnitudes)

    speeds = np.linalg.norm(velocities, axis=1)
    return {
        'mean_speed': np.mean(speeds),
        'mean_acceleration': mean_acceleration
    }

def classify_trajectory(trajectory: np.ndarray, dt: float = 0.1) -> Tuple[str, str]:
    """
    Classify a trajectory into motion and path cases.

    Args:
        trajectory: Array of positions/states over time [timesteps, features] or [timesteps, batch, features]
        dt: Time step for calculating acceleration

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
        velocities = traj[:, 12:14]  # vx, vy
    else:  # 2D case
        velocities = traj[:, 4:6]  # vx, vy

    motion_case = detect_motion_cases(velocities, dt=dt)
    path_case = detect_path_cases(traj)

    return motion_case, path_case

def find_representative_trajectories(gt_data: torch.Tensor, pred_data: torch.Tensor,
                                   num_samples: int = 1000, dt: float = 0.1,
                                   max_error: float = 2.0) -> Dict[str, Dict[str, list]]:
    """
    Find representative trajectories for each motion and path case combination.

    Args:
        gt_data: Ground truth trajectories [timesteps, batch, features]
        pred_data: Predicted trajectories [timesteps, batch, features]
        num_samples: Number of trajectories to sample for classification
        dt: Time step for calculating acceleration
        max_error: Maximum allowed position error in meters (default: 2.0)

    Returns:
        Dictionary mapping case combinations to trajectory indices
    """
    gt_np = gt_data.cpu().numpy() if isinstance(gt_data, torch.Tensor) else gt_data
    pred_np = pred_data.cpu().numpy() if isinstance(pred_data, torch.Tensor) else pred_data

    # Sample trajectories for efficiency
    batch_size = gt_np.shape[1]
    sample_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)

    case_trajectories = {
        'accelerating': {'straight': [], 'slight_turn': [], 'continuous_turn': []},
        'uniform': {'straight': [], 'slight_turn': [], 'continuous_turn': []}
    }

    for idx in sample_indices:
        traj_gt = gt_np[:, idx, :]
        traj_pred = pred_np[:, idx, :]

        # Calculate position error (Euclidean distance)
        gt_pos = traj_gt[:, :2]  # x, y positions
        pred_pos = traj_pred[:, :2]  # x, y positions
        position_errors = np.linalg.norm(gt_pos - pred_pos, axis=1)
        max_position_error = np.max(position_errors)

        # Skip trajectories with error exceeding threshold
        if max_position_error > max_error:
            continue

        # Calculate mean speed and filter out slow trajectories
        if traj_gt.shape[1] >= 15:  # 3D case
            velocities = traj_gt[:, 12:15]  # vx, vy, vz
        else:  # 2D case
            velocities = traj_gt[:, 4:6]  # vx, vy

        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = np.mean(speeds)

        # Skip trajectories with mean speed less than 0.9 m/s
        if mean_speed < 0.9:
            continue

        motion_case, path_case = classify_trajectory(traj_gt, dt=dt)

        if len(case_trajectories[motion_case][path_case]) < 3:  # Limit to 3 examples per case
            case_trajectories[motion_case][path_case].append(idx)

    return case_trajectories

def kinematic_bicycle_model(state: np.ndarray, u: np.ndarray, dt: float, wheelbase: float = 0.5) -> np.ndarray:
    """
    Kinematic bicycle model dynamics: x_dot = f(x, u)

    Args:
        state: Current state [x, y, theta, v] where:
               x, y: position
               theta: heading angle
               v: velocity
        u: Control input [throttle, steering_angle] where:
           throttle: acceleration command
           steering_angle: steering angle (delta)
        dt: Time step (not used here, kept for API consistency)
        wheelbase: Distance between front and rear axles

    Returns:
        x_dot: State derivative [x_dot, y_dot, theta_dot, v_dot]
    """
    _ = dt  # Unused parameter, kept for consistency with integration methods
    _, _, theta, v = state  # x, y not needed for dynamics computation
    throttle, delta = u

    # Kinematic bicycle model equations
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = (v / wheelbase) * np.tan(delta)
    v_dot = throttle  # Assuming throttle directly controls acceleration

    return np.array([x_dot, y_dot, theta_dot, v_dot])


def euler_integration(state: np.ndarray, u: np.ndarray, dt: float, wheelbase: float = 0.5) -> np.ndarray:
    """
    Euler integration: x_{k+1} = x_k + dt * f(x_k, u_k)

    Args:
        state: Current state [x, y, theta, v]
        u: Control input [throttle, steering_angle]
        dt: Time step
        wheelbase: Distance between front and rear axles

    Returns:
        next_state: Next state [x, y, theta, v]
    """
    x_dot = kinematic_bicycle_model(state, u, dt, wheelbase)
    next_state = state + dt * x_dot
    return next_state


def rk4_integration(state: np.ndarray, u: np.ndarray, dt: float, wheelbase: float = 0.5) -> np.ndarray:
    """
    Runge-Kutta 4th order integration

    Args:
        state: Current state [x, y, theta, v]
        u: Control input [throttle, steering_angle]
        dt: Time step
        wheelbase: Distance between front and rear axles

    Returns:
        next_state: Next state [x, y, theta, v]
    """
    k1 = kinematic_bicycle_model(state, u, dt, wheelbase)
    k2 = kinematic_bicycle_model(state + 0.5 * dt * k1, u, dt, wheelbase)
    k3 = kinematic_bicycle_model(state + 0.5 * dt * k2, u, dt, wheelbase)
    k4 = kinematic_bicycle_model(state + dt * k3, u, dt, wheelbase)

    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state


def kinematic_prediction(gt_traj: np.ndarray, timesteps: int, dt: float = 0.1,
                        method: str = 'euler', wheelbase: float = 0.5) -> np.ndarray:
    """
    Generate trajectory prediction using kinematic bicycle model with proper control inputs.

    Args:
        gt_traj: Ground truth trajectory [timesteps+1, features] containing states and controls
        timesteps: Number of timesteps to predict
        dt: Time step
        method: Integration method ('euler' or 'rk4')
        wheelbase: Distance between front and rear axles

    Returns:
        Predicted trajectory [timesteps+1, features] with kinematic prediction
    """
    traj = np.zeros((timesteps + 1, gt_traj.shape[1]))
    traj[0] = gt_traj[0].copy()

    # Determine if 2D or 3D case based on state dimension
    if gt_traj.shape[1] >= 15:  # 3D case
        # Extract initial state: [x, y, theta, v]
        # Format: [x, y, z, R(9), vx, vy, vz, omega(3), s(4), rpm(4), u(3)]
        for t in range(timesteps):
            x = traj[t, 0]  # x position
            y = traj[t, 1]  # y position

            # Get heading from rotation matrix or use simplified approach
            vx = traj[t, 12]
            vy = traj[t, 13]
            theta = np.arctan2(vy, vx)  # Heading from velocity direction
            v = np.sqrt(vx**2 + vy**2)  # Speed magnitude

            # Extract control inputs from ground truth
            # Assuming u contains [throttle, steering_angle, ...]
            u_idx = 26  # Start of control input in state vector
            if u_idx < gt_traj.shape[1]:
                throttle = gt_traj[t, u_idx]  # First control input (throttle/acceleration)
                steering = -gt_traj[t, u_idx + 1] if u_idx + 1 < gt_traj.shape[1] else 0.0
            else:
                throttle = 0.0
                steering = 0.0

            # Create simplified state for kinematic model
            state = np.array([x, y, theta, v])
            u = np.array([throttle, steering])

            # Integrate using selected method
            if method == 'rk4':
                next_state = rk4_integration(state, u, dt, wheelbase)
            else:  # euler
                next_state = euler_integration(state, u, dt, wheelbase)

            # Update trajectory with next state
            traj[t+1] = traj[t].copy()
            traj[t+1, 0] = next_state[0]  # x
            traj[t+1, 1] = next_state[1]  # y
            traj[t+1, 12] = next_state[3] * np.cos(next_state[2])  # vx
            traj[t+1, 13] = next_state[3] * np.sin(next_state[2])  # vy

    else:  # 2D case
        # Format for 2D: [x, y, theta, feedback_speed, feedback_steer, u_speed, u_steer]
        for t in range(timesteps):
            x = traj[t, 0]  # x position
            y = traj[t, 1]  # y position
            theta = traj[t, 2]  # heading angle

            # Get velocity from feedback or estimate
            if traj.shape[1] >= 4:
                v = traj[t, 3]  # feedback_speed
            else:
                v = 0.0

            # Extract control inputs
            # Assuming controls are at the end: [u_speed, u_steer]
            if traj.shape[1] >= 7:
                throttle = traj[t, 5]  # u_speed (assuming this is acceleration)
                steering = traj[t, 6]  # u_steer
            elif traj.shape[1] >= 6:
                throttle = 0.0
                steering = traj[t, 5] if traj.shape[1] >= 6 else 0.0
            else:
                throttle = 0.0
                steering = 0.0

            # Create state for kinematic model
            state = np.array([x, y, theta, v])
            u = np.array([throttle, steering])

            # Integrate using selected method
            if method == 'rk4':
                next_state = rk4_integration(state, u, dt, wheelbase)
            else:  # euler
                next_state = euler_integration(state, u, dt, wheelbase)

            # Update trajectory with next state
            traj[t+1] = traj[t].copy()
            traj[t+1, 0] = next_state[0]  # x
            traj[t+1, 1] = next_state[1]  # y
            traj[t+1, 2] = next_state[2]  # theta
            if traj.shape[1] >= 4:
                traj[t+1, 3] = next_state[3]  # update feedback_speed if available

    return traj

def plot_trajectory_comparison(gt_traj: np.ndarray, pred_traj: np.ndarray,
                             motion_case: str, path_case: str,
                             ax: plt.Axes, stats: Dict[str, float], dt: float = 0.1,
                             wheelbase: float = 0.5) -> None:
    """
    Plot comparison between ground truth, predicted trajectory, and kinematic models (Euler & RK4).

    Args:
        gt_traj: Ground truth trajectory [timesteps, features]
        pred_traj: Predicted trajectory [timesteps, features]
        motion_case: Motion classification (not used, for compatibility)
        path_case: Path classification (not used, for compatibility)
        ax: Matplotlib axes to plot on
        stats: Trajectory statistics
        dt: Time step for kinematic model
        wheelbase: Distance between front and rear axles for kinematic model
    """
    # Extract positions (x, y for 2D visualization)
    gt_pos = gt_traj[:, :2]
    pred_pos = pred_traj[:, :2]

    # Generate kinematic model predictions using both Euler and RK4
    kinematic_euler_traj = kinematic_prediction(gt_traj, len(gt_traj) - 1, dt, method='euler', wheelbase=wheelbase)
    kinematic_euler_pos = kinematic_euler_traj[:, :2]

    kinematic_rk4_traj = kinematic_prediction(gt_traj, len(gt_traj) - 1, dt, method='rk4', wheelbase=wheelbase)
    kinematic_rk4_pos = kinematic_rk4_traj[:, :2]

    # Plot trajectories
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], 'o-', color='green', linewidth=2,
            markersize=3, label='Ground Truth', alpha=0.8)
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], 's-', color='orange', linewidth=2,
            markersize=3, label='PhysORD', alpha=0.8)
    ax.plot(kinematic_euler_pos[:, 0], kinematic_euler_pos[:, 1], '^-', color='blue', linewidth=2,
            markersize=3, label='Kinematic (Euler)', alpha=0.8)
    ax.plot(kinematic_rk4_pos[:, 0], kinematic_rk4_pos[:, 1], 'v-', color='purple', linewidth=2,
            markersize=3, label='Kinematic (RK4)', alpha=0.8)

    # Mark start and end points
    ax.plot(gt_pos[0, 0], gt_pos[0, 1], 'go', markersize=8, label='Start')
    ax.plot(gt_pos[-1, 0], gt_pos[-1, 1], 'rs', markersize=8, label='End')

    # Calculate trajectory bounds to scale appropriately
    all_pos = np.vstack([gt_pos, pred_pos, kinematic_euler_pos, kinematic_rk4_pos])
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
                            figsize: Tuple[int, int] = (16, 12), dt: float = 0.1,
                            max_error: float = 2.0, wheelbase: float = 0.5) -> None:
    """
    Create comprehensive plot comparing ground truth and predicted trajectories
    across different motion and path cases.

    Args:
        gt_data: Ground truth trajectories [timesteps, batch, features]
        pred_data: Predicted trajectories [timesteps, batch, features]
        save_path: Path to save the plot
        figsize: Figure size (width, height)
        dt: Time step for calculating acceleration
        max_error: Maximum allowed position error in meters (default: 2.0)
        wheelbase: Distance between front and rear axles for kinematic model (default: 0.5)
    """
    # Find representative trajectories
    case_trajectories = find_representative_trajectories(gt_data, pred_data, dt=dt, max_error=max_error)

    # Create subplot grid
    motion_cases = ['accelerating', 'uniform']
    path_cases = ['straight', 'slight_turn', 'continuous_turn']

    fig, axes = plt.subplots(len(motion_cases),len(path_cases), figsize=figsize)
    fig.suptitle('Ground Truth vs PhysORD vs Kinematic Models (Euler & RK4): Motion and Path Analysis', fontsize=16)

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
                stats = calculate_trajectory_stats(gt_traj, dt=dt)

                # Plot comparison
                plot_trajectory_comparison(gt_traj, pred_traj, motion_case, path_case, ax, stats, dt=dt, wheelbase=wheelbase)

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
    parser.add_argument('--dt', '--time_step', type=float, default=0.1, help='Time step/dt for trajectory calculations')
    parser.add_argument('--control_dim', type=int, choices=[2, 3], default=3, help='Number of control input dimensions (2 or 3)')
    parser.add_argument('--max_error', type=float, default=2.0, help='Maximum allowed position error in meters for trajectory filtering (default: 2.0)')
    parser.add_argument('--wheelbase', type=float, default=0.5, help='Wheelbase distance for kinematic bicycle model (default: 0.5)')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model_type} model from: {args.model_path}")
    if args.model_type == '2d':
        model = PlanarPhysORD(device=device, time_step=args.dt, udim=2, use_v_gap=args.use_v_gap).to(device)
    else:
        model = PhysORD(device=device, use_dVNet=True, time_step=args.dt, udim=args.control_dim, use_v_gap=args.use_v_gap).to(device)

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
    print(f"Filtering trajectories with max error > {args.max_error}m")
    print(f"Using wheelbase = {args.wheelbase}m for kinematic model")
    save_path = os.path.join(args.save_dir, 'comprehensive_analysis.png')
    create_comprehensive_plot(gt_data, pred_data, save_path=save_path, figsize=tuple(args.figsize),
                            dt=args.dt, max_error=args.max_error, wheelbase=args.wheelbase)
    print(f"Comprehensive plot saved to: {save_path}")
    print(f"All plots saved to: {args.save_dir}")