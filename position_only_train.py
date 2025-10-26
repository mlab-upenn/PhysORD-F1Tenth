"""
Training script for PlanarPositionOnlyPhysORD model.

This script trains a physics-informed neural network for planar motion prediction
using only position updates (no velocity states). The model uses a variational
integrator approach to learn dynamics from position trajectories.

State representation: [x, y, θ, feedback_speed, feedback_steer, u_speed, u_steer]
- Position: [x, y, θ] (2D position + rotation angle)
- Feedback: [feedback_speed, feedback_steer] (sensor measurements)
- Control: [u_speed, u_steer] (commanded inputs)

The model learns to predict external forces and torques that drive the system dynamics.
"""

import torch
import argparse
import pickle
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from planar_physord.planar_position_only_model import PlanarPositionOnlyPhysORD
from util.data_process import get_model_parm_nums
from util.utils import normalize_theta


def position_only_loss(target, target_hat, split):
    """
    Custom loss function for position-only model.

    The state vector contains: [x, y, θ, feedback_speed, feedback_steer, u_speed, u_steer]
    We care about position (x, y) and orientation (θ) accuracy.

    Args:
        target: Ground truth state [batch, timesteps, state_dim]
        target_hat: Predicted state [batch, timesteps, state_dim]
        split: List of dimensions for splitting state vector
               [xdim=2, thetadim=1, feedback_speed_dim=1, feedback_steer_dim=1, udim=2]

    Returns:
        loss: Combined loss value
    """
    # Split states into components
    x_hat, theta_hat, _, _, _ = torch.split(target_hat, split, dim=2)
    x, theta, _, _, _ = torch.split(target, split, dim=2)

    # Flatten for loss computation
    x = x.flatten(start_dim=0, end_dim=1)
    x_hat = x_hat.flatten(start_dim=0, end_dim=1)
    x_loss = (x - x_hat).pow(2).mean()

    # Angle loss with wrapping to [-π, π] using utils.normalize_theta
    theta = theta.flatten(start_dim=0, end_dim=1)
    theta_hat = theta_hat.flatten(start_dim=0, end_dim=1)
    angle_diff = normalize_theta(theta - theta_hat)
    theta_loss = angle_diff.pow(2).mean()

    # Combined loss
    total_loss = x_loss + theta_loss

    return total_loss


def data_load(args):
    """
    Load and prepare training and validation data.

    Expected data format:
    - Tensor shape: [data_size, num_steps + past_history, state_dim]
    - state_dim = 7 for position-only model: [x, y, θ, feedback_speed, feedback_steer, u_speed, u_steer]

    Args:
        args: Command line arguments containing data paths and parameters

    Returns:
        train_data: Training trajectories tensor
        val_data: Validation trajectories tensor
    """
    print("Loading data ...")

    if args.custom_data_path:
        print(f"Loading custom data: {args.custom_data_path}")
        custom_data = torch.load(args.custom_data_path)[:4500, :, :]

        # Validate data shape
        print(f"Data shape: {custom_data.shape}")
        expected_state_dim = 7  # [x, y, θ, feedback_speed, feedback_steer, u_speed, u_steer]
        if custom_data.shape[2] != expected_state_dim:
            print(f"Warning: Expected state_dim={expected_state_dim}, got {custom_data.shape[2]}")

        # Data format: [data_size, num_steps + past_history, state_dim]
        data_size = custom_data.shape[0]
        num_steps = custom_data.shape[1] - args.past_history_input

        print(f"Data size: {data_size}, Num steps: {num_steps}, State dim: {custom_data.shape[2]}")

        # Split data into train/val (80/20 split)
        train_size = int(0.8 * data_size)

        # Shuffle indices for random split
        torch.manual_seed(args.seed)
        indices = torch.randperm(data_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_data = custom_data[train_indices, :, :]
        val_data = custom_data[val_indices, :, :]

        # Update timesteps to match the data if needed
        if args.timesteps != num_steps:
            print(f"Note: Data has {num_steps} timesteps, adjusting args.timesteps from {args.timesteps} to {num_steps}")
            args.timesteps = num_steps

    else:
        raise ValueError("custom_data_path is required for position-only model training")

    # Convert to appropriate dtype and device
    train_data = train_data.clone().detach().to(dtype=torch.float64, device=device).requires_grad_(True)
    val_data = val_data.clone().detach().to(dtype=torch.float64, device=device).requires_grad_(False)

    print(f"Train data: {train_data.shape}, Val data: {val_data.shape}")
    return train_data, val_data


def train(args, train_data, val_data):
    """
    Main training loop for the position-only model.

    Args:
        args: Command line arguments
        train_data: Training trajectories [data_size, num_steps + past_history, state_dim]
        val_data: Validation trajectories [data_size, num_steps + past_history, state_dim]

    Returns:
        model: Trained model
        stats: Dictionary containing training statistics
    """
    # Training settings
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(save_fp, 'tensorboard_logs')
    writer = SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")

    # Initialize model
    print("Creating PlanarPositionOnlyPhysORD model ...")
    model = PlanarPositionOnlyPhysORD(
        device=device,
        udim=args.control_dim,
        time_step=args.time_step,
        past_history_input=args.past_history_input,
        hidden_size=args.hidden_size,
        use_feedback=args.use_feedback
    ).to(device)

    if args.pretrained is not None:
        print(f"Loading pretrained model from: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    num_parm = get_model_parm_nums(model)
    print(f'Model contains {num_parm} parameters')

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)

    print(f"{args.exp_name}: Start training")
    print(f"  Trajectories: {train_data.shape[0]}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Learning rate: {args.learn_rate}")
    print(f"  Past history input: {args.past_history_input}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Use feedback: {args.use_feedback}")

    # Training statistics
    stats = {
        'loss': [],
        'val_error': [],
        'best_error': [],
        'train_time': [],
        'eval_time': [],
        'save_time': [],
        'epoch_time': [],
        'position_error': [],
        'angle_error': []
    }

    best_error = float('inf')
    best_step = -1
    counter = 0
    terminate = False
    patience = 150 if args.early_stopping else args.num_epochs

    batch_size = args.batch_size
    steps_total = train_data.shape[0] // batch_size + (1 if train_data.shape[0] % batch_size != 0 else 0)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        shuffled_indices = torch.randperm(train_data.shape[0])
        t_epoch = time.time()

        for step in range(steps_total):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(shuffled_indices))
            batch_indices = shuffled_indices[start_idx:end_idx]
            x = train_data[batch_indices, :, :]  # [batch_size, num_steps + past_history, state_dim]

            optimizer.zero_grad()

            # Forward pass
            x_hat = model(args.timesteps, x)
            target = x[:, args.past_history_input:, :]  # Ground truth future states 
            target_hat = x_hat[:, args.past_history_input:, :]  # Predicted future states
            assert target.shape == target_hat.shape, "Mismatch in target and prediction shapes"

            # Compute loss
            # Split: [xdim=2, thetadim=1, feedback_speed_dim=1, feedback_steer_dim=1, udim=2]
            train_loss_mini = position_only_loss(
                target,
                target_hat,
                split=[model.xdim, model.thetadim, model.feedback_speed_dim,
                       model.feedback_steer_dim, model.udim]
            )

            epoch_loss += train_loss_mini.item()

            # Backward pass
            train_loss_mini.backward()
            optimizer.step()

        train_time = time.time() - t_epoch

        # Evaluation
        t_eval = time.time()
        model.eval()
        with torch.no_grad():
            val_hat = model.evaluation(args.timesteps, val_data)

            # Compute full trajectory loss (same as training)
            val_target = val_data[:, args.past_history_input:, :]  # Ground truth future states
            val_target_hat = val_hat[:, args.past_history_input:, :]  # Predicted future states

            # Use the same loss function as training for consistency
            val_loss = position_only_loss(
                val_target,
                val_target_hat,
                split=[model.xdim, model.thetadim, model.feedback_speed_dim,
                       model.feedback_steer_dim, model.udim]
            )

            # Also compute final timestep metrics for monitoring
            val_pos = val_hat[:, -1, :2]  # [batch, 2] - final (x, y)
            gt_pos = val_data[:, -1, :2]
            val_angle = val_hat[:, -1, 2:3]  # [batch, 1] - final θ
            gt_angle = val_data[:, -1, 2:3]

            # Position RMSE (final timestep only, for monitoring)
            pos_rmse = (val_pos - gt_pos).pow(2).sum(dim=1).mean().sqrt()

            # Angle error with wrapping using utils.normalize_theta
            angle_diff = normalize_theta(val_angle - gt_angle).squeeze()
            angle_rmse = angle_diff.pow(2).mean().sqrt()

            # Use full trajectory loss as validation error
            val_error = val_loss

            # Save best model
            if val_error < best_error:
                counter = 0
                best_error = val_error
                best_step = epoch
                best_dir = os.path.join(save_fp, 'best')
                os.makedirs(best_dir, exist_ok=True)
                path = f'{best_dir}/best-data{train_data.shape[0]}-timestep{args.timesteps}.tar'
                torch.save(model.state_dict(), path)
            else:
                counter += 1
                if counter >= patience:
                    terminate = True

        eval_time = time.time() - t_eval

        # Save periodic checkpoints
        t_save = time.time()
        if epoch % args.save_every == 0:
            path = f'{save_fp}/data{train_data.shape[0]}-timestep{args.timesteps}-epoch{epoch}.tar'
            torch.save(model.state_dict(), path)
        save_time = time.time() - t_save

        # Update statistics
        stats['loss'].append(epoch_loss)
        stats['val_error'].append(val_error.item())
        stats['best_error'].append(best_error.item())
        stats['train_time'].append(train_time)
        stats['eval_time'].append(eval_time)
        stats['save_time'].append(save_time)
        stats['position_error'].append(pos_rmse.item())
        stats['angle_error'].append(angle_rmse.item())

        # TensorBoard logging
        writer.add_scalar('Loss/Training_Loss', epoch_loss, epoch)
        writer.add_scalar('Error/Validation_Error', val_error.item(), epoch)
        writer.add_scalar('Error/Position_RMSE', pos_rmse.item(), epoch)
        writer.add_scalar('Error/Angle_RMSE', angle_rmse.item(), epoch)
        writer.add_scalar('Error/Best_Error', best_error.item(), epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Log gradient and weight norms
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_norm = param.data.norm(2).item()
                writer.add_scalar(f'Weight_Norms/{name}', weight_norm, epoch)

                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, epoch)

        # Step scheduler
        scheduler.step()

        # Print progress
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch:4d} | Loss: {epoch_loss:.4e} | "
                  f"Val Error: {val_error:.4e} | Pos RMSE: {pos_rmse:.4e} | "
                  f"Angle RMSE: {angle_rmse:.4e} | Best: {best_error:.4e}")

        epoch_time = time.time() - t_epoch
        stats['epoch_time'].append(epoch_time)

        # Early stopping
        if terminate:
            print(f"Early stopping at epoch {epoch}")
            break

    # Cleanup
    writer.close()
    print(f"\nTraining complete!")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print(f"To view logs, run: tensorboard --logdir={tensorboard_dir}")
    print(f"Best model saved at epoch {best_step} with error {best_error:.4e}")

    stats['best_step'] = best_step
    return model, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PlanarPositionOnlyPhysORD model for planar motion prediction"
    )

    # Experiment settings
    parser.add_argument('--exp_name', default='position_only_physord', type=str,
                       help='Experiment name for saving results')
    parser.add_argument('--save_dir', default="./result/", type=str,
                       help='Directory to save trained models and logs')

    # Data settings
    parser.add_argument('--custom_data_path', required=True, type=str,
                       help='Path to custom tensor file (e.g., ./data/planar_data.pt)')
    parser.add_argument('--timesteps', type=int, default=20,
                       help='Number of prediction steps')
    parser.add_argument('--past_history_input', default=2, type=int,
                       help='Number of past history inputs for force model')

    # Model settings
    parser.add_argument('--control_dim', default=2, type=int,
                       help='Number of control input dimensions (default: 2 for speed and steer)')
    parser.add_argument('--time_step', default=0.1, type=float,
                       help='Time step for model integration (seconds)')
    parser.add_argument('--hidden_size', default=64, type=int,
                       help='Hidden layer size for the force model MLP (default: 64)')
    parser.add_argument('--use_feedback', action='store_true', default=True,
                       help='Use feedback measurements in the force model (default: True)')
    parser.add_argument('--no_feedback', action='store_false', dest='use_feedback',
                       help='Disable feedback measurements in the force model')
    parser.add_argument('--pretrained', default=None, type=str,
                       help='Path to pretrained model weights')

    # Training settings
    parser.add_argument('--learn_rate', default=1e-2, type=float,
                       help='Learning rate for Adam optimizer')
    parser.add_argument('--num_epochs', default=5000, type=int,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', default=65, type=int,
                       help='Training batch size (number of trajectories)')
    parser.add_argument('--seed', default=0, type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')

    # Logging and checkpointing
    parser.add_argument('--print_every', default=50, type=int,
                       help='Print training progress every N epochs')
    parser.add_argument('--save_every', default=1000, type=int,
                       help='Save model checkpoint every N epochs')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping with patience=150')

    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_fp = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_fp, exist_ok=True)

    # Save arguments
    args_dict = vars(args)
    with open(os.path.join(save_fp, 'args.pkl'), 'wb') as f:
        pickle.dump(args_dict, f)
    print(f"Arguments saved to: {os.path.join(save_fp, 'args.pkl')}")

    # Load data
    train_data, val_data = data_load(args)

    # Train model
    model, stats = train(args, train_data, val_data)

    # Save training statistics
    stats_path = f'{save_fp}/stats-timestep{args.timesteps}.pkl'
    with open(stats_path, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Training statistics saved to: {stats_path}")
