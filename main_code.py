import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate
#-----------------------------------------
### Egenværdier og egenvektorer for et kvantesystem

# Gitter
points = 200
x_lattice = np.linspace(-15, 15, points)
t_grid = np.linspace(0, 100, 100)

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
    T_matrix = np.zeros((3, points))

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

def construct_V(x, t, omega):
    omega_square = omega**2
    x_square = x**2
    V_diag = omega_square * x_square / 2
    V =  np.zeros((3, points))
    V[1,:] = V_diag
    return V

# Konstruer H
H = construct_T(points, delta_x_lattice, mass_electron) + V

#---------------------------------------------------------------------------
# Eigenvalues and Eigenvectors
eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(H[1,:], H[0, 1:])  # Skal kun have diagonal + superdiagonal

def plot_normalized_eigenfunction(eigenvectors: np.ndarray, x_lattice: np.ndarray)->None:
    fig, ax = plt.subplots(10, 1)
    for i, axes in enumerate(ax):
        normalizing_factor = (scipy.integrate.simpson(abs(eigenvectors[i])**2, x_lattice))
        axes.plot(x_lattice, abs(eigenvectors[i])**2*normalizing_factor)


#---------------------------------------------------------------------------
### Tidspropagation

def banded_mv(A, x):
    y = A[1,:] * x
    y[:-1] += A[0,1:] * x[1:]
    y[1:]  += A[2,:-1] * x[:-1]
    return y

def crank_nicholson_matrix(matrix_length: int, x, delta_t, t: float, n: int = 0) -> np.ndarray:
    """

    """
    V = construct_V(x_lattice, t, omega)
    T = construct_T(points, delta_x_lattice, mass_electron)
    H = T + V
    second_term = (-1)**n * ((1j * delta_t) / 2) * H
    second_term[1,:] += 1
    return second_term

psis = [eigvecs[0]]
dt = 0.1
omega = 2
for i in range(10000):
    right_vector = banded_mv(crank_nicholson_matrix(points, dt, 0, 1), psis[-1])
    left_matrix = crank_nicholson_matrix(points, x_lattice, dt, 0)
    new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
    psis.append(new_psi)

psis = np.array(psis)

def plot_normalized_eigenfunction(psis: np.ndarray, x_lattice: np.ndarray)->None:
    fig, ax = plt.subplots(10, 1)
    for i, axes in enumerate(ax):
        normalizing_factor = 1 / (scipy.integrate.simpson(abs(eigvecs[:, i])**2, x_lattice))
        axes.plot(x_lattice, abs(eigvecs[:, i])**2*normalizing_factor)

plot_normalized_eigenfunction(psis, x_lattice)
