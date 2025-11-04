# Planar PhysORD: 2D Physics-Informed Ordinary Differential Equations for Robotics
# Simplified version of 3D PhysORD for planar motion (x-y plane, z-axis rotation only)

import torch
from planar_physord.planar_nn_models import ForceMLP
import numpy as np
import util.utils as utils


class PositionOnlyForceModelHelper(torch.nn.Module):
    """
    This helper class constructs the external force model for the planar position-only PhysORD.
    It builds a neural network that predicts external forces and torque based on the current state,
    control inputs, and optionally feedback measurements.
    This class will be helpful for experimenting with different force model architectures and input features.
    """
    def __init__(
            self,
            use_feedback=True,
            udim=2,
            past_history_input=2,
            pose_dim=4,
            feedback_speed_dim=1,
            feedback_steer_dim=1,
            hidden_size=128,
            relative_coords=True,
        ):
        """
        Args:
            use_feedback: Whether to use feedback measurements in the force model
            udim: Control input dimension
            past_history_input: Number of past history inputs to consider
            pose_dim: Dimension of the pose (default 4 for planar: x, y, cos(θ), sin(θ))
            feedback_speed_dim: Dimension of feedback speed measurement (default 1 for planar)
            feedback_steer_dim: Dimension of feedback steering measurement (default 1 for planar)
            hidden_size: Hidden layer size for the force model MLP
            relative_coords: Whether to convert x, y, and theta to relative coordinates (default True)
        """
        super(PositionOnlyForceModelHelper, self).__init__()
        self.use_feedback = use_feedback
        self.udim = udim
        self.past_history_input = past_history_input
        self.posedim = pose_dim
        self.feedback_speed_dim = feedback_speed_dim
        self.feedback_steer_dim = feedback_steer_dim
        self.relative_coords = relative_coords

        input_dim = (
            pose_dim
            + (feedback_speed_dim if use_feedback else 0)
            + (feedback_steer_dim if use_feedback else 0)
            + udim
        ) * past_history_input
        self.input_dim = input_dim
        print(f"Force model input dimension: {self.input_dim}")

        self.force_model = ForceMLP(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            output_size=3,  # [fx, fy, tau_z]
        )

    def evaluate_force_model(self, x):
        """
        Evaluate the external force model for the given input state.

        Args:
            x: [batch_size, past_history_input_size, state_dim] input state tensor
               state_dim = posedim + feedback_speed_dim + feedback_steer_dim + udim

        Returns:
            force_output: [batch_size, 3] predicted external forces and torque
        """
        bs, past_history_input_size, state_dim = x.shape

        assert state_dim == self.posedim + self.feedback_speed_dim + self.feedback_steer_dim + self.udim, \
            f"Expected state_dim {self.posedim + self.feedback_speed_dim + self.feedback_steer_dim + self.udim}, got {state_dim}"
        assert past_history_input_size == self.past_history_input, \
            f"Expected past_history_input_size {self.past_history_input}, got {past_history_input_size}"

        if self.relative_coords:
            # Don't pass absolute pose information, treat the latest pose as origin with zero orientation
            # Reposition + reorient all past poses relative to the latest pose
            initial_xy = x[:, -1, 0:2]  # [bs, 2]
            initial_cos_theta = x[:, -1, 2]  # [bs]
            initial_sin_theta = x[:, -1, 3]  # [bs]

            # Build rotation matrix R_k for current orientation (body to world frame)
            R_k = torch.zeros(bs, 2, 2, device=x.device, dtype=x.dtype)  # [bs, 2, 2]
            R_k[:, 0, 0] = initial_cos_theta
            R_k[:, 0, 1] = -initial_sin_theta
            R_k[:, 1, 0] = initial_sin_theta
            R_k[:, 1, 1] = initial_cos_theta

            # Transform positions to relative coordinates in body frame (avoid in-place)
            xy_relative = x[:, :, 0:2] - initial_xy.unsqueeze(1)  # [bs, past_history_input_size, 2]
            # Rotate xy_relative into body frame: xy_body = R_k^T * (xy_relative) in col vector form
            # but we have row vectors, so do xy_body = (xy_relative) * R_k.
            # So, in bmm form: xy_body = bmm(xy_relative, R_k)
            xy_body = torch.bmm(xy_relative, R_k)  # [bs, past_history_input_size, 2]

            # Transform orientations to relative using rotation matrices (avoid in-place)
            # For each timestep, compute R_relative = R_k^T * R_i
            cos_theta = x[:, :, 2]  # [bs, past_history_input_size]
            sin_theta = x[:, :, 3]  # [bs, past_history_input_size]

            # Compute relative cos and sin: R_relative = R_k^T * R_i
            # cos(theta_rel) = cos(theta_k)*cos(theta_i) + sin(theta_k)*sin(theta_i)
            # sin(theta_rel) = -sin(theta_k)*cos(theta_i) + cos(theta_k)*sin(theta_i)
            cos_theta_rel = initial_cos_theta.unsqueeze(1) * cos_theta + initial_sin_theta.unsqueeze(1) * sin_theta
            sin_theta_rel = -initial_sin_theta.unsqueeze(1) * cos_theta + initial_cos_theta.unsqueeze(1) * sin_theta

            # Construct x_transformed with transformed pose values (no in-place modification)
            if self.use_feedback:
                # Concatenate: xy_body, cos_theta_rel, sin_theta_rel, feedback, control
                x_transformed = torch.cat([
                    xy_body,
                    cos_theta_rel.unsqueeze(2),
                    sin_theta_rel.unsqueeze(2),
                    x[:, :, 4:6],  # feedback
                    x[:, :, 6:8]   # control
                ], dim=2)
                x_flat = x_transformed.reshape(bs, -1)  # flatten past history inputs
            else:
                # Use only pose information and control inputs
                x_pose_transformed = torch.cat([
                    xy_body,
                    cos_theta_rel.unsqueeze(2),
                    sin_theta_rel.unsqueeze(2)
                ], dim=2)  # [bs, past_history_input_size, 4]
                u = x[:, :, 6:8]  # [bs, past_history_input_size, udim]
                x_selected = torch.cat((x_pose_transformed, u), dim=2)
                x_flat = x_selected.reshape(bs, -1)  # flatten past history inputs
        else:
            # Use absolute coordinates
            if self.use_feedback:
                # Use all state information: pose, feedback, control
                x_flat = x.reshape(bs, -1)  # flatten past history inputs
            else:
                # Use only pose information and control inputs
                xy = x[:, :, 0:2]  # [bs, past_history_input_size, 2]
                cos_theta = x[:, :, 2:3]  # [bs, past_history_input_size, 1]
                sin_theta = x[:, :, 3:4]  # [bs, past_history_input_size, 1]
                u = x[:, :, 6:8]  # [bs, past_history_input_size, udim]
                x_selected = torch.cat((xy, cos_theta, sin_theta, u), dim=2)
                x_flat = x_selected.reshape(bs, -1)  # flatten past history inputs

        force_output = self.force_model(x_flat)  # [bs, 3]

        return force_output


class PlanarPositionOnlyPhysORD(torch.nn.Module):
    """
    Planar PhysORD model with position only update for 2D motion with the following constraints:
    - Translation only in x-y plane (no z motion)
    - Rotation only about z-axis (scalar angle θ)
    - No potential energy changes (constant height)
    
    State vector structure: [x, y, cos(θ), sin(θ), feedback_speed, feedback_steer, u_speed, u_steer]
    - Position: [x, y, cos(θ), sin(θ)] (2D position + rotation as cos/sin)
    - Feedback measurements: [feedback_speed, feedback_steer] (measured speed and steering angle)
    - Control inputs: [u_speed, u_steer] (commanded speed, steering angle)
    """

    def __init__(
            self,
            device=None,
            udim=2,
            time_step=0.1,
            past_history_input=2,
            hidden_size=64,
            use_feedback=True,
            relative_coords=True,
        ):
        super(PlanarPositionOnlyPhysORD, self).__init__()
        self.device = device

        # Dimensional parameters for planar motion
        self.xdim = 2           # 2D position (x, y)
        self.thetadim = 2      # 2D rotation representation (cos(θ), sin(θ))
        self.posedim = self.xdim + self.thetadim  # Total pose dimension = 4
        self.feedback_speed_dim = 1      # 1D linear velocity (vx, vy combined as speed) measurement
        self.feedback_steer_dim = 1      # 1D steering angle measurement through sensor
        self.udim = udim        # Control input dimension

        self.h = time_step  # Time step for integration
        self.past_history_input = past_history_input  # Past history input size

        # External force model helper
        self.force_model_helper = PositionOnlyForceModelHelper(
            use_feedback=use_feedback,
            udim=self.udim,
            past_history_input=past_history_input,
            pose_dim=self.posedim,
            feedback_speed_dim=self.feedback_speed_dim,
            feedback_steer_dim=self.feedback_steer_dim,
            hidden_size=hidden_size,
            relative_coords=relative_coords,
        )

    def compute_cos_sin_update(self, R_k, R_k_m_1, tau_z):
        """
        Helper function to compute cos(theta_{k+1}) and sin(theta_{k+1}) from rotation matrices.

        Args:
            R_k: [bs, 2, 2] rotation matrix at timestep k
            R_k_m_1: [bs, 2, 2] rotation matrix at timestep k-1
            tau_z: [bs, 1] torque about z-axis (scaled by moment of inertia)

        Returns:
            cos_theta_k_p_1: [bs, 1] cos(theta_{k+1})
            sin_theta_k_p_1: [bs, 1] sin(theta_{k+1})
        """
        # Compute Z_k_m_1 = R_k_m_1^T * R_k
        Z_k_m_1 = torch.bmm(R_k_m_1.transpose(1, 2), R_k)  # [bs, 2, 2]

        # Extract sin(theta_k - theta_k_m_1) from Z_k_m_1
        sin_delta_theta = Z_k_m_1[:, 1, 0].unsqueeze(1)  # [bs, 1]

        # a = 0.5 * (h^2 * tau_z) + sin(theta_k - theta_k_m_1)
        a = 0.5 * (self.h ** 2) * tau_z + sin_delta_theta  # [bs, 1]
        # print(f"Max abs a before clamp: {torch.max(torch.abs(a))}")

        a = torch.clamp(a, -0.9999, 0.9999)  # Prevent numerical issues

        # sin(theta_{k+1} - theta_k) = a
        # cos(theta_{k+1} - theta_k) = sqrt(1 - a^2)
        sin_delta_theta_k_p_1 = a  # [bs, 1]
        cos_delta_theta_k_p_1 = torch.sqrt(1 - a.pow(2))  # [bs, 1]

        # Compute cos(theta_{k+1}) and sin(theta_{k+1})
        cos_theta_k = R_k[:, 0, 0].unsqueeze(1)  # [bs, 1]
        sin_theta_k = R_k[:, 1, 0].unsqueeze(1)  # [bs, 1]
        cos_theta_k_p_1 = (
            cos_theta_k * cos_delta_theta_k_p_1
            - sin_theta_k * sin_delta_theta_k_p_1
        )  # [bs, 1]
        sin_theta_k_p_1 = (
            sin_theta_k * cos_delta_theta_k_p_1
            + cos_theta_k * sin_delta_theta_k_p_1
        )  # [bs, 1]

        # Normalize to ensure cos^2 + sin^2 = 1 (critical for numerical stability)
        norm = torch.sqrt(cos_theta_k_p_1.pow(2) + sin_theta_k_p_1.pow(2))
        cos_theta_k_p_1 = cos_theta_k_p_1 / norm
        sin_theta_k_p_1 = sin_theta_k_p_1 / norm

        return cos_theta_k_p_1, sin_theta_k_p_1


    def step_forward(self, x, enable_grad=True):
        """
        Forward step of the planar position-only PhysORD model.

        Args:
            x: [batch_size, past_history_input_size, state_dim] input state tensor
               state_dim = posedim + feedback_speed_dim + feedback_steer_dim + udim

        Returns:
            x_next: [batch_size, past_history_input_size, posedim + feedback_speed_dim + feedback_steer_dim + udim]
                next state tensor constructed with the latest pose at the end and removing the oldest pose.
        """
        with torch.set_grad_enabled(enable_grad):
            bs, past_history_input_size, state_dim = x.shape

            assert state_dim == self.posedim + self.feedback_speed_dim + self.feedback_steer_dim + self.udim, \
                f"Expected state_dim {self.posedim + self.feedback_speed_dim + self.feedback_steer_dim + self.udim}, got {state_dim}"
            assert past_history_input_size == self.past_history_input, \
                f"Expected past_history_input_size {self.past_history_input}, got {past_history_input_size}"
            
            xdim = self.xdim
            thetadim = self.thetadim
            feedback_speed_dim = self.feedback_speed_dim
            feedback_steer_dim = self.feedback_steer_dim
            udim = self.udim

            # Get external forces and torque from the force model
            force_output_body_frame = self.force_model_helper.evaluate_force_model(x)  # [bs, 3] -> [fx, fy, tau_z]
            fxy_body = force_output_body_frame[:, 0:2]  # [bs, 2] - forces in body frame
            tau_z = force_output_body_frame[:, 2:3]  # [bs, 1]

            # Split the input state
            # theta now contains [cos_theta, sin_theta] with shape [bs, past_history_input, 2]
            xy, theta, feedback_speed, feedback_steer, u = torch.split(
                x,
                [xdim, thetadim, feedback_speed_dim, feedback_steer_dim, udim],
                dim=2,
            )

            # Normalize theta to ensure cos^2 + sin^2 = 1 for all timesteps (safety check)
            theta_norm = torch.sqrt(theta[:, :, 0].pow(2) + theta[:, :, 1].pow(2)).unsqueeze(2)  # [bs, past_history_input, 1]
            theta = theta / theta_norm  # [bs, past_history_input, 2]

            # R_k = rotation matrix from body to world frame at timestep k
            R_k = torch.zeros(bs, 2, 2, device=x.device, dtype=x.dtype)  # [bs, 2, 2]
            cos_theta_k = theta[:, -1, 0]  # [bs] - cos(theta) at timestep k
            sin_theta_k = theta[:, -1, 1]  # [bs] - sin(theta) at timestep k
            R_k[:, 0, 0] = cos_theta_k
            R_k[:, 0, 1] = -sin_theta_k
            R_k[:, 1, 0] = sin_theta_k
            R_k[:, 1, 1] = cos_theta_k

            # R_k_m_1 = rotation matrix at timestep k-1
            R_k_m_1 = torch.zeros(bs, 2, 2, device=x.device, dtype=x.dtype)  # [bs, 2, 2]
            cos_theta_k_m_1 = theta[:, -2, 0]  # [bs]
            sin_theta_k_m_1 = theta[:, -2, 1]  # [bs]
            R_k_m_1[:, 0, 0] = cos_theta_k_m_1
            R_k_m_1[:, 0, 1] = -sin_theta_k_m_1
            R_k_m_1[:, 1, 0] = sin_theta_k_m_1
            R_k_m_1[:, 1, 1] = cos_theta_k_m_1

            # force in world frame
            fxy_world = torch.bmm(R_k, fxy_body.unsqueeze(2)).squeeze(2)  # [bs, 2]

            # Update xy using variational integrator
            xy_k = xy[:, -1, :].unsqueeze(1)  # [bs, 1, 2]
            xy_k_m_1 = xy[:, -2, :].unsqueeze(1)  # [bs, 1, 2]
            xy_k_p_1 = (
                2 * xy_k - xy_k_m_1
                + 0.5 * (self.h ** 2) * fxy_world.unsqueeze(1)  # [bs, 1, 2]
            ) # assuming that predicted by neural net is force / mass.

            # Update orientation using the helper function
            cos_theta_k_p_1, sin_theta_k_p_1 = self.compute_cos_sin_update(R_k, R_k_m_1, tau_z)  # [bs, 1], [bs, 1]

            # Construct next state by appending the new pose and removing the oldest pose
            xy_next = torch.cat((xy[:, 1:, :], xy_k_p_1), dim=1)  # [bs, past_history_input_size, 2]

            # Stack cos and sin for theta_next
            theta_k_p_1 = torch.cat([cos_theta_k_p_1, sin_theta_k_p_1], dim=1).unsqueeze(1)  # [bs, 1, 2]
            theta_next = torch.cat((theta[:, 1:, :], theta_k_p_1), dim=1)  # [bs, past_history_input_size, 2]

            x_next = torch.cat((xy_next, theta_next, feedback_speed, feedback_steer, u), dim=2)  # [bs, past_history_input_size, state_dim]

            return x_next

    def evaluation(self, num_steps, trajectory):
        """
        Evaluate the model through multiple time steps.

        Args:
            num_steps: Number of time steps to simulate
            trajectory: [batch_size, past_history_input_size  + num_steps, state_dim] input trajectory tensor
               state_dim = posedim + feedback_speed_dim + feedback_steer_dim + udim

        Returns:
            traj_out: [batch_size, past_history_input_size + num_steps, state_dim] output trajectory tensor
        """
        x_seq = trajectory[:, :self.past_history_input, :]  # initial past history input
        curx = trajectory[:, :self.past_history_input, :]

        u_inputs = trajectory[:, self.past_history_input:, -self.udim:]  # control inputs for future steps
        for i in range(num_steps):
            # Step forward
            nextx = self.step_forward(curx, enable_grad=False) # size of nextx: [bs, past_history_input_size, state_dim]
            curx = nextx

            # Append the latest control input from the trajectory for the latest time step to the last pose
            # NOTE: Feedback values remain constant through time, as we won't model their dynamics here
            curx[:, -1, -self.udim:] = u_inputs[:, i, :]  # set control input for the latest time step

            # Append to output trajectory
            x_seq = torch.cat((x_seq, curx[:, -1:, :]), dim=1)

        return x_seq


    def forward(self, num_steps, trajectory):
        """
        Forward pass through multiple time steps.

        Args:
            num_steps: Number of time steps to simulate
            trajectory: [batch_size, past_history_input_size  + num_steps, state_dim] input trajectory tensor
               state_dim = posedim + feedback_speed_dim + feedback_steer_dim + udim

        Returns:
            traj_out: [batch_size, past_history_input_size + num_steps, state_dim] output trajectory tensor
        """
        x_seq = trajectory[:, :self.past_history_input, :]  # initial past history input
        curx = trajectory[:, :self.past_history_input, :]

        u_inputs = trajectory[:, self.past_history_input:, -self.udim:]  # control inputs for future steps
        for i in range(num_steps):
            # Step forward
            nextx = self.step_forward(curx, enable_grad=True) # size of nextx: [bs, past_history_input_size, state_dim]
            curx = nextx

            # Append the latest control input from the trajectory for the latest time step to the last pose
            # NOTE: Feedback values remain constant through time, as we won't model their dynamics here
            curx[:, -1, -self.udim:] = u_inputs[:, i, :]  # set control input for the latest time step

            # Append to output trajectory
            x_seq = torch.cat((x_seq, curx[:, -1:, :]), dim=1)

        return x_seq
