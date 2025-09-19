import sympy
import numpy as np

def S(v):
    """
    Convert a 3D column vector to a skew-symmetric matrix.
    Args:
        v: A tensor of shape (3,) or (1, 3)
    Returns:
        A sympy Matrix of shape (3, 3) representing the skew-symmetric matrix.
    """
    skew = sympy.Matrix([[0, -v[2, 0], v[1, 0]],
                         [v[2, 0], 0, -v[0, 0]],
                         [-v[1, 0], v[0, 0], 0]])
    return skew

def trace(A):
    """
    Compute the trace of a square matrix.
    Args:
        A: A sympy Matrix of shape (n, n)
    Returns:
        A sympy expression representing the trace of the matrix.
    """
    if not isinstance(A, sympy.Matrix):
        raise ValueError("Input must be a sympy Matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square")

    tr = sum(A[i, i] for i in range(A.shape[0]))
    return tr

# Define symbolic variables for inertia matrix components
J11, J12, J13 = sympy.symbols('J11 J12 J13')
J21, J22, J23 = sympy.symbols('J21 J22 J23')
J31, J32, J33 = sympy.symbols('J31 J32 J33')
J = sympy.Matrix([[J11, J12, J13],
                  [J21, J22, J23],
                  [J31, J32, J33]])
for i in range(3):
    for j in range(i+1, 3):
        J[j, i] = J[i, j]

print("Symbolic Inertia Matrix J:")
sympy.pprint(J)
print("\n")

Jd = 0.5*trace(J)*sympy.eye(3) - J
print("Symbolic Jd (0.5*trace(J)*I - J):")
sympy.pprint(Jd)
print("\n")

h = sympy.symbols('h') # for time step
wz = sympy.symbols('wz') # angular velocity about z-axis
w = sympy.Matrix([0, 0, wz]) # angular velocity vector
tau_z = sympy.symbols('tau_z') # torque about z-axis
tau = sympy.Matrix([0, 0, tau_z]) # torque vector
theta = sympy.symbols('theta') # rotation angle about z-axis
Z = sympy.Matrix([[sympy.cos(theta), -sympy.sin(theta), 0],
                  [sympy.sin(theta), sympy.cos(theta), 0],
                  [0, 0, 1]])


# LHS
lhs_eqn = h*S(J @ w) + h**2 * S(tau)
print("LHS of the equation (h*S(J*w) + h^2*S(tau)):")
sympy.pprint(lhs_eqn)
print("\n")

# RHS
rhs_eqn = Z @ Jd - Jd @ Z.T
print("RHS of the equation (Z*J - J*Z^T):")
sympy.pprint(rhs_eqn)
print("\n")

# Full equation
full_eqn = sympy.Eq(lhs_eqn, rhs_eqn)
print("Full equation (LHS = RHS):")
sympy.pprint(full_eqn)
print("\n")

# Simplify to find theta
simplified_eqn = sympy.simplify(lhs_eqn - rhs_eqn)
print("Simplified equation:")
sympy.pprint(simplified_eqn)
print("\n")

# Solve for theta
theta_solution = sympy.solve(simplified_eqn, theta)
print("Solution for theta:")
sympy.pprint(theta_solution)
print("\n")
