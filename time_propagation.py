import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib.animation as anim
from matplotlib import rc
plt.rc("animation", html="jshtml")

def banded_mv(A, x):
    y = A[1,:] * x
    y[:-1] += A[0,1:] * x[1:]
    y[1:]  += A[2,:-1] * x[:-1]
    return y

def potential(x: np.ndarray, V_0: float = 10, standard_deviation: float = 1, x_0: float = 40) -> np.ndarray:
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
    return V_0 * np.exp(-1 * ((x - x_0)**2) / (4 * standard_deviation**2))

def construct_T(matrix_length: int, delta_x: float, mass: float) -> np.ndarray:
    """
    Constructs the T matrix, which consists of a tri-diagonal matrix. All other points than the diagonal,
    super- and sub-diagonal have the value 0, so only those diagonals are represented as a matrix.

    parameters:
        matrix_length: length of the matrix
        delta_x: length between x-values
        mass: mass of the particle
    """
    T_matrix = np.zeros((3, matrix_length))

    T_sub_diag = np.zeros((1,matrix_length - 1)) + 1
    T_super_diag = np.zeros((1,matrix_length - 1)) + 1
    T_diag = np.zeros((1,matrix_length)) -2

    T_matrix[0,1:] = T_sub_diag
    T_matrix[1,:] = T_diag
    T_matrix[2,:-1] = T_super_diag

    T_factor = - 1 / (2*mass*delta_x**2)

    T = T_factor * T_matrix
    return T

def crank_nicholson_matrix(matrix_length: int, x, delta_t, t: float, n: int = 0) -> np.ndarray:
    """

    """
    delta_x_lattice = x[1] - x[0]
    V = potential(x)
    T = construct_T(len(x), delta_x_lattice, mass_electron)
    H = T
    H[1,:] += V
    second_term = (-1)**n * ((1j * delta_t) / 2) * H
    second_term[1,:] += 1
    return second_term

def construct_wave(position, variance, central_speed, center_position):
    factor_1 = 1 / (2*np.pi*variance)**(1/4)
    factor_2 = np.exp(-(position - center_position)**2 / (4 * variance))
    factor_3 = np.exp(1j * position * central_speed)
    wave = factor_1 * factor_2 * factor_3
    return wave

mass_electron = 1
points = 3000
spread = 4
variance = spread ** 2
center_speed = np.sqrt(21)
center_position = 0
x_lattice = np.linspace(-150, 150, points)
dt = 0.1
wave_0 = construct_wave(x_lattice, variance, center_speed, center_position)

#fig, ax = plt.subplots(3, 1)
#ax[0].plot(abs(construct_wave(x_lattice, variance, center_speed, center_position)**2))

psis = [wave_0]
for i in range(1000):
    right_vector = banded_mv(crank_nicholson_matrix(points, x_lattice, dt, 0, 1), psis[-1])
    left_matrix = crank_nicholson_matrix(points, x_lattice, dt, 0)
    new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
    psis.append(new_psi)
#ax[1].plot(x_lattice, abs(psis[-1])**2)
#ax[1].plot(x_lattice, potential(x_lattice)/(max(potential(x_lattice)*20)))

def ani_func(index):
    normalizing_factor = 1 / (scipy.integrate.simpson(abs(psis[index*10])**2, x_lattice))
    plot_func = normalizing_factor * abs((psis[index*10]))**2
    line.set_ydata(plot_func)
    return line,

fig1, ani_ax = plt.subplots()
potential_plot_normalize = 1 / max(potential(x_lattice)) * max(abs(psis[0])**2) * 1.2
potential_plot = potential(x_lattice) * potential_plot_normalize
ani_ax.plot(x_lattice, potential_plot, color="C3")
line, = ani_ax.plot(x_lattice, abs(psis[0])**2)

ani = anim.FuncAnimation(fig1, ani_func, frames=100, blit=False)
ani
