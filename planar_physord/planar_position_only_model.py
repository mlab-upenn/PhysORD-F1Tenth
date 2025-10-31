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
            pose_dim=3,
            feedback_speed_dim=1,
            feedback_steer_dim=1,
            hidden_size=128,
        ):
        """
        Args:
            use_feedback: Whether to use feedback measurements in the force model
            udim: Control input dimension
            past_history_input: Number of past history inputs to consider
            pose_dim: Dimension of the pose (default 3 for planar: x, y, θ)
            feedback_speed_dim: Dimension of feedback speed measurement (default 1 for planar)
            feedback_steer_dim: Dimension of feedback steering measurement (default 1 for planar)
            hidden_size: Hidden layer size for the force model MLP
        """
        super(PositionOnlyForceModelHelper, self).__init__()
        self.use_feedback = use_feedback
        self.udim = udim
        self.past_history_input = past_history_input
        self.posedim = pose_dim
        self.feedback_speed_dim = feedback_speed_dim
        self.feedback_steer_dim = feedback_steer_dim

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
        
        # Don't pass absolute pose information, treat the latest pose as origin with zero orientation
        # Reposition + reorient all past poses relative to the latest pose
        initial_xy = x[:, -1, 0:2]  # [bs, 2]
        initial_theta = x[:, -1, 2]  # [bs]

        # Transform to relative coordinates
        x[:, :, 0:2] = x[:, :, 0:2] - initial_xy.unsqueeze(1)  # [bs, past_history_input_size, 2]
        
        # To modify theta, just subtract initial_theta and normalize
        x[:, :, 2] = x[:, :, 2] - initial_theta.unsqueeze(1)  # [bs, past_history_input_size]
        x[:, :, 2] = utils.normalize_theta(x[:, :, 2])  # [bs, past_history_input_size]
        
        if self.use_feedback:
            x_flat = x.reshape(bs, -1)  # flatten past history inputs
        else:
            # Use only pose information and control inputs
            posedim = self.posedim
            x_pose = x[:, :, :posedim]  # [bs, past_history_input_size, posedim]
            u_start_idx = posedim + self.feedback_speed_dim + self.feedback_steer_dim
            u = x[:, :, u_start_idx:]  # [bs, past_history_input_size, udim]
            x_selected = torch.cat((x_pose, u), dim=2)
            x_flat = x_selected.reshape(bs, -1)  # flatten past history inputs

        force_output = self.force_model(x_flat)  # [bs, 3]

        # Bring back x to original coordinates
        x[:, :, 0:2] = x[:, :, 0:2] + initial_xy.unsqueeze(1)  # [bs, past_history_input_size, 2]
        x[:, :, 2] = x[:, :, 2] + initial_theta.unsqueeze(1)  # [bs, past_history_input_size]
        x[:, :, 2] = utils.normalize_theta(x[:, :, 2])  # [bs, past_history_input_size]

        return force_output


class PlanarPositionOnlyPhysORD(torch.nn.Module):
    """
    Planar PhysORD model with position only update for 2D motion with the following constraints:
    - Translation only in x-y plane (no z motion)
    - Rotation only about z-axis (scalar angle θ)
    - No potential energy changes (constant height)
    
    State vector structure: [x, y, θ, feedback_speed, feedback_steer, u_speed, u_steer]
    - Position: [x, y, θ] (2D position + rotation angle)
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
        ):
        super(PlanarPositionOnlyPhysORD, self).__init__()
        self.device = device

        # Dimensional parameters for planar motion
        self.xdim = 2           # 2D position (x, y)
        self.thetadim = 1      # 1D rotation (θ about z-axis)
        self.posedim = self.xdim + self.thetadim  # Total pose dimension = 3
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
        )


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
            xy, theta, feedback_speed, feedback_steer, u = torch.split(
                x,
                [xdim, thetadim, feedback_speed_dim, feedback_steer_dim, udim],
                dim=2,
            )

            # R_k = rotation matrix from body to world frame
            R_k = torch.zeros(bs, 2, 2, device=x.device, dtype=x.dtype)  # [bs, 2, 2]
            cos_theta = torch.cos(theta[:, -1, 0])  # [bs]
            sin_theta = torch.sin(theta[:, -1, 0])  # [bs]
            R_k[:, 0, 0] = cos_theta
            R_k[:, 0, 1] = -sin_theta
            R_k[:, 1, 0] = sin_theta
            R_k[:, 1, 1] = cos_theta

            # force in world frame
            fxy_world = torch.bmm(R_k, fxy_body.unsqueeze(2)).squeeze(2)  # [bs, 2]

            # Update xy using variational integrator
            xy_k = xy[:, -1, :].unsqueeze(1)  # [bs, 1, 2]
            xy_k_m_1 = xy[:, -2, :].unsqueeze(1)  # [bs, 1, 2]
            xy_k_p_1 = (
                2 * xy_k - xy_k_m_1 
                + (self.h ** 2) * fxy_world.unsqueeze(1)  # [bs, 1, 2]
            ) # assuming that predicted by neural net is force / mass.

            # Update theta using variational integrator
            theta_k = theta[:, -1, :].unsqueeze(1)  # [bs, 1, 1]
            theta_k_m_1 = theta[:, -2, :].unsqueeze(1)  # [bs, 1, 1]
            delta_theta = (
                    utils.normalize_theta(theta_k - theta_k_m_1)
                    + (self.h ** 2) * tau_z.unsqueeze(1)  # [bs, 1, 1]
            ) # assuming that predicted by neural net is torque / I.

            theta_k_p_1 = utils.normalize_theta(
                theta_k + delta_theta
            )

            # Construct next state by appending the new pose and removing the oldest pose
            xy_next = torch.cat((xy[:, 1:, :], xy_k_p_1), dim=1)  # [bs, past_history_input_size, 2]
            theta_next = torch.cat((theta[:, 1:, :], theta_k_p_1), dim=1)  # [bs, past_history_input_size, 1]

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
