# Planar PhysORD

This directory contains a simplified 2D version of PhysORD (Physics-Informed Ordinary Differential Equations for Robotics in Dynamics) that operates in the planar domain with constraints:

- **2D Translation**: Motion only in x-y plane (no z-axis motion)
- **Single Rotation**: Only rotation about z-axis (θ rotation)
- **No Potential Energy Changes**: Gravitational potential energy remains constant

## Simplified 2D Equations

### State Variables (Planar)
- **Position**: `q = [x, y, θ]` (2D position + 1 rotation angle)
- **Linear Velocity**: `v = [vx, vy]` (2D linear velocity)
- **Angular Velocity**: `ω = ωz` (scalar angular velocity about z-axis)

### Update Equations (Simplified from 3D)

#### Position Update:
```
x_{t+1} = x_t + h*vx_t + h^2/m * R_t * f_x^-
y_{t+1} = y_t + h*vy_t + h^2/m * R_t * f_y^-
θ_{t+1} = θ_t + h*ωz_t + h^2/I_z * tau_z^-  (implicit rotation update for z-axis only)
```

#### Velocity Update:
```
m*v_{t+1} = m*v_t + R_t*f^- + R_{t+1}*f^+
I_z*ω_{t+1} = I_z*ω_t + f_θ^-  + f_θ^+
```

Where:
- `h`: time step
- `α ∈ [0,1]`: integration parameter
- `m`: 2x2 mass matrix (diagonal for planar motion)
- `I_z`: scalar moment of inertia about z-axis
- `f^-`, `f^+`: external forces (neural network)

### Simplifications from 3D

1. **Mass Matrix**: Simplified from 3x3 to 2x2 diagonal matrix
2. **Inertia Matrix**: Reduced from 3x3 to scalar `I_z`
3. **Rotation**: Only z-axis rotation (scalar instead of 3D rotation matrix)
4. **Forces**: 2D force vector + 1D torque instead of 6D wrench
5. **No Gravity**: Potential energy changes eliminated (constant z-height)

### Neural Networks

**Force Network**: `f_net(v, ω, u, gap) → [fx, fy, τz]`

The planar version significantly reduces computational complexity while maintaining the physics-informed structure of the original 3D PhysORD model.