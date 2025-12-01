import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate
#-----------------------------------------
### Egenværdier og egenvektorer for et kvantesystem

points = 200
x_lattice = np.linspace(-15, 15, points)
t_grid = np.linspace(0, 100, 100)

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
    T_matrix = np.zeros((3, 3000))

    T_sub_diag = np.zeros((1,matrix_length - 1)) + 1
    T_super_diag = np.zeros((1,matrix_length - 1)) + 1
    T_diag = np.zeros((1,matrix_length)) -2

    T_matrix[0,1:] = T_sub_diag
    T_matrix[1,:] = T_diag
    T_matrix[2,:-1] = T_super_diag

    T_factor = 1 / (2*mass*delta_x**2)

    T = T_factor * T_matrix
    return T

# Konstruer H
H = construct_T(points, delta_x_lattice, mass_electron) + V

#---------------------------------------------------------------------------
# Eigenvalues and Eigenvectors
eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(H[1,:], H[0, 1:])

def plot_normalized_eigenfunction(eigenvectors: np.ndarray, x_lattice: np.ndarray)->None:
    fig, ax = plt.subplots(10, 1)
    for i, axes in enumerate(ax):
        normalizing_factor = (scipy.integrate.simpson(abs(eigenvectors[i])**2, x_lattice))
        axes.plot(x_lattice, abs(eigenvectors[i])**2*normalizing_factor)

def banded_mv(A, x):
    y = A[1,:] * x
    y[:-1] += A[0,1:] * x[1:]
    y[1:]  += A[2,:-1] * x[:-1]
    return y

def potential(x: np.ndarray, V_0: float = 2, standard_deviation: float = 10, x_0: float = 0) -> np.ndarray:
    """
    Calculates the potential in form of a slim Gauss for a wave to collide with.

    paramenters:
        x: the values of which the potential is calculated
        V_0: the initial value of the potential
        standard_deviation: how much the function deviates
        x_0: displacement of the top point along the x-axis in the positive direction
    """
    if standard_deviation == 0:
        raise Exception("The standard deviation cannot be 0")
    return V_0 * np.exp(-1 * ((x - x_0)**2) / 4 * standard_deviation**2)

def crank_nicholson_matrix(matrix_length: int, x, delta_t, t: float, n: int = 0) -> np.ndarray:
    """

    """
    delta_x_lattice = x[1] - x[0]
    V = potential(x)
    T = construct_T(3000, delta_x_lattice, mass_electron)
    H = T + V
    second_term = (-1)**n * ((1j * delta_t) / 2) * H
    second_term[1,:] += 1
    return second_term

spread = 5
variance = spread ** 2
center_speed = np.sqrt(8)
center_position = 0
x_lattice = np.linspace(-100, 100, 3000)
dt = 0.01
wave_0 = construct_wave(x_lattice, variance, center_speed, center_position)

fig, ax = plt.subplots(3, 1)
ax[0].plot(abs(construct_wave(x_lattice, variance, center_speed, center_position)**2))

psis = [wave_0]
for i in range(10000):
    right_vector = banded_mv(crank_nicholson_matrix(3000, x_lattice, dt, 0, 1), psis[-1])
    left_matrix = crank_nicholson_matrix(3000, x_lattice, dt, 0)
    new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
    psis.append(new_psi)
ax[1].plot(x_lattice, abs(psis[-1])**2)
ax[1].plot(x_lattice, potential(x_lattice)/(max(potential(x_lattice)*20)))
