# Planar PhysORD: 2D Physics-Informed Ordinary Differential Equations for Robotics
# Simplified version of 3D PhysORD for planar motion (x-y plane, z-axis rotation only)

import torch
from planar_nn_models import FixedMass, FixedInertia, ForceMLP

class PlanarPhysORD(torch.nn.Module):
    """
    Planar PhysORD model for 2D motion with the following constraints:
    - Translation only in x-y plane (no z motion)
    - Rotation only about z-axis (scalar angle θ)
    - No potential energy changes (constant height)
    
    State vector structure: [x, y, θ, vx, vy, ωz, s, rpm, ux, uy, uθ]
    - Position: [x, y, θ] (2D position + rotation angle)
    - Velocity: [vx, vy, ωz] (2D linear velocity + angular velocity)
    - Sensor data: [s] (4D sensor readings)
    - RPM data: [rpm] (4D RPM readings)  
    - Control inputs: [ux, uy, uθ] (2D force + torque commands)
    """
    
    def __init__(self, device=None, udim=3, time_step=0.1):
        super(PlanarPhysORD, self).__init__()
        self.device = device
        
        # Dimensional parameters for planar motion
        self.xdim = 2           # 2D position (x, y)
        self.thetadim = 1       # 1D rotation (θ about z-axis)
        self.posedim = self.xdim + self.thetadim  # Total pose dimension = 3
        self.linveldim = 2      # 2D linear velocity (vx, vy)
        self.angveldim = 1      # 1D angular velocity (ωz)
        self.twistdim = self.linveldim + self.angveldim  # Total twist dimension = 3
        self.udim = udim        # Control input dimension
        
        # Mass matrix for planar motion (2x2 diagonal matrix)
        # Represents mass in x and y directions
        eps_m = torch.Tensor([900., 900.])  # [mass_x, mass_y]
        self.M = FixedMass(m_dim=2, eps=eps_m, param_value=0).to(device)
        
        # Inertia matrix for planar motion (scalar moment of inertia about z-axis)
        eps_j = torch.Tensor([1000.])  # [Iz] - only z-axis inertia needed
        self.J = FixedInertia(m_dim=1, eps=eps_j, param_value=0).to(device)
        
        # No potential energy network needed for planar case (constant height)
        
        # External force and torque network
        # Input: [vx, vy, ωz, ux, uy, uθ, v_gap(4)] = 10 dimensions
        # Output: [fx, fy, τz] = 3 dimensions (2D force + z-axis torque)
        self.force_mlp = ForceMLP(10, 64, 3).to(device)
        
        # Physical constants
        self.circumference = 2    # Wheel circumference for RPM conversion
        self.h = time_step        # Integration time step
        
    def step_forward(self, x):
        """
        Forward integration step for planar PhysORD dynamics.
        
        Args:
            x: State tensor [batch_size, state_dim] where state_dim includes:
               [x, y, θ, vx, vy, ωz, s(4), rpm(4), ux, uy, uθ]
               
        Returns:
            Next state tensor with same structure
        """
        with torch.enable_grad():
            bs = x.shape[0]  # batch size
            
            # Split state vector into components
            # posedim=3, twistdim=3, 4 sensor readings, 4 rpm readings, udim control inputs
            pose_k, twist_k, sk, rpmk, uk = torch.split(
                x, [self.posedim, self.twistdim, 4, 4, self.udim], dim=1
            )
            
            # Further split pose and twist
            xyk, thetak = torch.split(pose_k, [self.xdim, self.thetadim], dim=1)  # [x,y], [θ]
            vk, omegak = torch.split(twist_k, [self.linveldim, self.angveldim], dim=1)  # [vx,vy], [ωz]
            
            # Extract mass and inertia matrices
            Mx = self.M.repeat(bs, 1, 1)      # (bs, 2, 2) - 2D mass matrix
            MR = self.J.repeat(bs, 1, 1)      # (bs, 1, 1) - scalar inertia matrix
            Mx_inv = torch.inverse(Mx)        # Inverse mass matrix
            MR_inv = torch.inverse(MR)        # Inverse inertia matrix
            
            # Compute velocity gap for 4 wheels (4-dimensional as in original)
            v_rpm = rpmk / 60 * self.circumference  # Convert RPM to velocity for each wheel
            v_sum = torch.sqrt(torch.sum(vk ** 2, dim=1, keepdim=True))  # Speed magnitude
            v_gap = v_sum - v_rpm  # 4D velocity gap (one for each wheel)
            
            # Prepare input for force/torque neural network
            f_input = torch.cat((vk, omegak, uk, v_gap), dim=1)  # [vx, vy, ωz, ux, uy, uθ, v_gap(4)]
            external_forces = self.force_mlp(f_input)  # Neural network output [fx, fy, τz]
            
            # Split forces and torques
            fX, fR = torch.split(external_forces, [self.linveldim, self.angveldim], dim=1)
            # fX: [fx, fy] - 2D forces
            # fR: [τz] - scalar torque about z-axis
            
            # Semi-implicit integration coefficients
            c = 0.5  # Integration parameter

            # Compute rotation matrices R_t and R_{t+1} for force transformation
            cos_theta_t = torch.cos(thetak.squeeze(-1))  # Current angle
            sin_theta_t = torch.sin(thetak.squeeze(-1))

            # R_t rotation matrix (bs, 2, 2)
            R_t = torch.zeros(bs, 2, 2, device=self.device)
            R_t[:, 0, 0] = cos_theta_t
            R_t[:, 0, 1] = -sin_theta_t
            R_t[:, 1, 0] = sin_theta_t
            R_t[:, 1, 1] = cos_theta_t

            # Compute force contributions for semi-implicit scheme
            fxk_minus = c * self.h * fX          # Force contribution at current timestep
            fxk_plus = (1 - c) * self.h * fX     # Force contribution at next timestep
            fRk_minus = c * self.h * fR          # Torque contribution at current timestep
            fRk_plus = (1 - c) * self.h * fR     # Torque contribution at next timestep

            # Apply rotation matrix R_t to forces at current timestep
            fxk_minus_world = torch.matmul(R_t, fxk_minus.unsqueeze(-1)).squeeze(-1)

            # Convert to proper tensor shapes for matrix operations
            fxk_minus_world = fxk_minus_world.unsqueeze(-1)  # (bs, 2, 1)
            fRk_minus = fRk_minus.unsqueeze(-1)  # (bs, 1, 1)
            fRk_plus = fRk_plus.unsqueeze(-1)    # (bs, 1, 1)
            
            # Compute momenta
            pxk = torch.squeeze(torch.matmul(Mx, vk.unsqueeze(-1)), dim=2)
            pRk = torch.squeeze(torch.matmul(MR, omegak.unsqueeze(-1)), dim=2)
            
            # Position updates (simplified - no potential energy terms)
            # x_{t+1} = x_t + h*vx_t + h²/m * R_t * f_x^-
            # y_{t+1} = y_t + h*vy_t + h²/m * R_t * f_y^-
            xyk_next = xyk + self.h * vk + \
                       self.h * torch.squeeze(torch.matmul(Mx_inv, fxk_minus_world))

            # Rotation update (simplified to scalar angle)
            # θ_{t+1} = θ_t + h*ωz_t + h²/I_z * τz^-
            thetak_next = thetak + self.h * omegak + \
                         self.h * torch.squeeze(torch.matmul(MR_inv, fRk_minus))

            # Compute R_{t+1} rotation matrix for next timestep force transformation
            cos_theta_t_plus_1 = torch.cos(thetak_next.squeeze(-1))  # Next angle
            sin_theta_t_plus_1 = torch.sin(thetak_next.squeeze(-1))

            # R_{t+1} rotation matrix (bs, 2, 2)
            R_t_plus_1 = torch.zeros(bs, 2, 2, device=self.device)
            R_t_plus_1[:, 0, 0] = cos_theta_t_plus_1
            R_t_plus_1[:, 0, 1] = -sin_theta_t_plus_1
            R_t_plus_1[:, 1, 0] = sin_theta_t_plus_1
            R_t_plus_1[:, 1, 1] = cos_theta_t_plus_1

            # Apply rotation matrix R_{t+1} to forces at next timestep
            fxk_plus_world = torch.matmul(R_t_plus_1, fxk_plus.unsqueeze(-1)).squeeze(-1)

            # Combine position components
            pose_k_next = torch.cat((xyk_next, thetak_next), dim=1)

            # Velocity updates (simplified - no potential energy terms)
            # m*v_{t+1} = m*v_t + R_t*f^- + R_{t+1}*f^+
            pxk_next = pxk + torch.squeeze(fxk_minus_world) + fxk_plus_world
            vk_next = torch.squeeze(torch.matmul(Mx_inv, pxk_next.unsqueeze(-1)), dim=-1)
            
            # Angular velocity update
            # I_z*ω_{t+1} = I_z*ω_t + f_θ^- + f_θ^+  
            pRk_next = pRk + torch.squeeze(fRk_minus) + torch.squeeze(fRk_plus)
            omegak_next = torch.squeeze(torch.matmul(MR_inv, pRk_next.unsqueeze(-1)), dim=-1)
            
            # Combine velocity components
            twist_k_next = torch.cat((vk_next, omegak_next), dim=1)
            
            # Return next state (pose, twist, sensors, rpm, control unchanged)
            return torch.cat((pose_k_next, twist_k_next, sk, rpmk, uk), dim=1)
    
    def efficient_evaluation(self, step_num, x, action):
        """
        Efficient trajectory evaluation with coordinate transformation.
        Maintains compatibility with 3D version interface.
        """
        # Store initial position for coordinate transformation
        initial_x, initial_y = x[:, 0].clone(), x[:, 1].clone()
        
        # Transform to relative coordinates
        x[:, 0] = x[:, 0] - initial_x  
        x[:, 1] = x[:, 1] - initial_y
        
        # Initialize trajectory sequence
        xseq = x[None, :, :]
        curx = x
        
        # Forward integration
        for i in range(step_num):
            nextx = self.step_forward(curx)
            # Update control inputs from action sequence  
            curx = torch.cat((nextx[:, :self.posedim + self.twistdim + 8], 
                             action[i+1, :, :]), dim=1)
            xseq = torch.cat((xseq, curx[None, :, :]), dim=0)
        
        # Transform back to absolute coordinates
        for i in range(step_num + 1):
            xseq[i, :, 0] = xseq[i, :, 0] + initial_x
            xseq[i, :, 1] = xseq[i, :, 1] + initial_y
            
        return xseq
    
    def evaluation(self, step_num, traj):
        """
        Standard trajectory evaluation for testing/validation.
        """
        xseq = traj[0, :, :]
        xseq = xseq[None, :, :]
        curx = traj[0, :, :]
        
        for i in range(step_num):
            nextx = self.step_forward(curx)
            # Use control inputs from trajectory
            curx = torch.cat((nextx[:, :self.posedim + self.twistdim + 8], 
                             traj[i+1, :, -self.udim:]), dim=1)
            xseq = torch.cat((xseq, curx[None, :, :]), dim=0)
            
        return xseq
    
    def forward(self, step_num, traj):
        """
        Forward pass for training with automatic control input handling.
        """
        xseq = traj[0, :, :]
        xseq = xseq[None, :, :]
        curx = traj[0, :, :]
        
        for i in range(step_num):
            nextx = self.step_forward(curx)
            curx = nextx
            # Update control inputs from trajectory
            curx[:, -self.udim:] = traj[i+1, :, -self.udim:]
            xseq = torch.cat((xseq, curx[None, :, :]), dim=0)
            
        return xseq