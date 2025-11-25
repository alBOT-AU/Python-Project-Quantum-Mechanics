import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate
#-----------------------------------------

# Gitter
x_lattice = np.linspace(-15, 15, 200)

# Konstruer T

mass_electron = 1
delta_x_lattice = (x_lattice[-1] - x_lattice[0]) / len(x_lattice)


def construct_T(matrix_length: int, delta_x: float, mass: float) -> np.ndarray:
    """
    Constructs the T matrix, which consists of a tri-diagonal matrix. All other points than the diagonal, 
    super- and sub-diagonal have the value 0, so only those diagonals are represented as a matrix.

    parameters:
        matrix_length: length of the matrix
        delta_x: length between x-values
        mass: mass of the particle
    """
    T_matrix = np.zeros((3, 200))

    T_sub_diag = np.zeros((1,matrix_length - 1)) + 1
    T_super_diag = np.zeros((1,matrix_length - 1)) + 1
    T_diag = np.zeros((1,matrix_length)) -2

    T_matrix[0,1:] = T_sub_diag
    T_matrix[1,:] = T_diag
    T_matrix[2,:-1] = T_super_diag

    T_factor = 1 / (2*mass*delta_x**2)

    T = T_factor * T_matrix
    return T

# Konstruer V
def V_factor(x: np.ndarray, omega: float) -> float:
    """
    Calculates the V-factor for the simple harmonic oscillator.

    parameters:
        x: the x-values to calculate the V-factor for
        omega: value for the angular frequency
    """
  omega_square = omega**2
  x_square = x**2
  return omega_square * x_square / 2

V_diag = V_factor(x_lattice, 2)
V =  np.zeros((3, 200))
V[1,:] = V_diag

# Konstruer H
H = construct_T(200, delta_x_lattice, mass_electron) + V

#---------------------------------------------------------------------------
# Eigenvalues and Eigenvectors
eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(H[1,:], H[0, 1:])  # Skal kun have diagonal + superdiagonal 


