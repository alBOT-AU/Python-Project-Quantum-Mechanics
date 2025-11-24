import numpy as np
import scipy.linalg
-----------------------------------------

# Gitter
x_lattice = np.linspace(-15, 15, 200)

# Konstruer T
T_matrix = np.zeros((3, 200))  # 3 for diagonalerne, 200 for størrelsen af matricen

T_sub_diag = np.zeros((1, 199)) + 1 
T_super_diag = np.zeros((1, 199)) + 1  
T_diag = np.zeros((1, 200)) - 2  

T_matrix[0,1:] = T_sub_diag  # Super diag har ikke noget i første kolonne 
T_matrix[1,:] = T_diag
T_matrix[2,:-1] = T_super_diag  # Sub diag har ikke noget i sidste kolonne 

mass_electron = 1 # Definerer enhedsløst
delta_x_lattice = (x_lattice[-1] - x_lattice[0]) / len(x_lattice)
T_factor = -1 / (2*mass_electron*delta_x_lattice**2)

T = T_factor * T_matrix

# Konstruer V
def V_factor(x, omega):
  omega_square = omega**2
  x_square = x**2
  return omega_square * x_square / 2

V_diag = V_factor(x_lattice, 2)
V =  np.zeros((3, 200))
V[1,:] = V_diag

# Konstruer H
H = T + V

---------------------------------------------------------------------------
# Eigenvalues and Eigenvectors
eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(H[1,:], H[0, 1:])  # Skal kun have diagonal + superdiagonal 


