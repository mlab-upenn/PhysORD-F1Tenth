# Simplified neural network models for 2D planar PhysORD
# Based on 3D version but adapted for planar motion constraints

import torch
import numpy as np
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class ForceMLP(nn.Module):
    """
    Neural network for external forces and torque in 2D planar motion.
    Outputs: [fx, fy, tau_z] - 2D force vector + scalar torque about z-axis
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ForceMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

def FixedMass(m_dim, eps, param_value=0):
    """
    Creates a fixed mass matrix for planar motion (2x2 diagonal).
    Simplified from 3D version - creates diagonal matrix directly.
    
    Args:
        m_dim: Mass matrix dimension (2 for planar motion)
        eps: Diagonal values [mass_x, mass_y] 
        param_value: Not used in planar version (kept for compatibility)
    
    Returns:
        M: (1, 2, 2) mass matrix
    """
    # For planar motion, create simple 2x2 diagonal mass matrix
    M = torch.zeros(1, m_dim, m_dim)
    for i in range(m_dim):
        M[0, i, i] = eps[i]
    
    return M

def FixedInertia(m_dim, eps, param_value=0):
    """
    Creates a fixed inertia matrix for planar motion (scalar Iz).
    Simplified from 3D version - returns scalar moment of inertia about z-axis.
    
    Args:
        m_dim: Should be 1 for planar motion (only z-axis rotation)
        eps: Scalar value [Iz] for z-axis moment of inertia
        param_value: Not used in planar version (kept for compatibility)
    
    Returns:
        I: (1, 1, 1) inertia tensor containing only Iz
    """
    # For planar motion, only need scalar inertia about z-axis
    I = torch.zeros(1, m_dim, m_dim)
    I[0, 0, 0] = eps[0]  # Only Iz component
    
    return I