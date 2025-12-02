import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate

points = 2000
x_lattice = np.linspace(-15, 15, points)
t_grid = np.linspace(0, 100, 100)
mass_electron = 1
delta_x_lattice = (x_lattice[-1] - x_lattice[0]) / len(x_lattice)
omega = 2

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

def construct_V(x: np.ndarray, omega: float):
    """
    Constructs the V matrix, which consists of a tri-diagonal matrix.

    parameters:
        x: x-values
    """
    omega_square = omega**2
    x_square = x**2
    V_diag = omega_square * x_square / 2
    V =  np.zeros((3, points))
    V[1,:] = V_diag
    return V

H = construct_T(points, delta_x_lattice, mass_electron) + construct_V(x_lattice, 2)

eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(H[1,:], H[0, 1:])

def plot_normalized_eigenfunction(x_lattice: np.ndarray)->None:
    fig, ax = plt.subplots(10, 1)
    for i, axes in enumerate(ax):
        normalizing_factor = 1 / (scipy.integrate.simpson(abs(eigvecs[:, i])**2, x_lattice))
        axes.plot(x_lattice, abs(eigvecs[:, i])**2*normalizing_factor)

plot_normalized_eigenfunction(x_lattice)
