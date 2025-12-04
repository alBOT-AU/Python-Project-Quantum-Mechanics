import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.integrate
from collections.abc import Callable
plt.rc("animation", html="jshtml")

class Particle_Collision():
    def __init__(
            self, wave: type[Callable], barrier: Callable, matrix_length: int = 3000,
            matrix_bounds: float = 150, mass: float = 1, dt: float = 0.1
    ):
        """
        Simulates a collision between a particle and a barrier, and shows quantum tunneling for certain energies of the particle and barrier.

        parameters:
            wave: wave function
            barrier: barrier function
            matrix_length: length of the matrix
            matrix_bounds: bounds of the matrix
            mass: mass of the particle
            dt: change in time between steps
        """
        self.matrix_length = matrix_length
        self.matrix_bounds = matrix_bounds
        self.mass = mass
        self.dt = dt
        self.x_lattice = np.linspace(-matrix_bounds, matrix_bounds, matrix_length)
        self.delta_x = (self.x_lattice[-1] - self.x_lattice[0]) / self.matrix_length
        self.wave = wave(self.x_lattice)
        self.barrier = barrier(self.x_lattice)

    def double_deriv(self) -> np.ndarray:
        """
        Constructs the T matrix, which consists of a tri-diagonal matrix. All other points than the diagonal,
        super- and sub-diagonal have the value 0, so only those diagonals are represented as a matrix.
        """
        T_matrix = np.zeros((3, self.matrix_length))

        T_sub_diag = np.zeros((1, self.matrix_length - 1)) + 1
        T_super_diag = np.zeros((1, self.matrix_length - 1)) + 1
        T_diag = np.zeros((1, self.matrix_length)) -2

        T_matrix[0,1:] = T_sub_diag
        T_matrix[1,:] = T_diag
        T_matrix[2,:-1] = T_super_diag

        T_factor = - 1 / (2* self.mass * self.delta_x**2)

        T = T_factor * T_matrix
        return T

    def crank_nicholson_matrix(self, n) -> np.ndarray:
        """
        Creates the Crank-Nicholson matrix, a numerical way to solve the Schrödinger equation.

        parameters:
            n: even values (zero included) gives a positive sign, uneven values gives a negative sign
        """
        V = self.barrier
        T = Particle_Collision.double_deriv(self)
        H = T
        H[1,:] += V
        second_term = (-1)**n * ((1j * self.dt) / 2) * H
        second_term[1,:] += 1
        return second_term

    def animate_collision(self, frame_space: int = 10, animation_points: int = 1000):
        """
        Animates the collision between the particle and the barrier.
        
        parameters:
            frame_space: chooses every n'th point to show in animation, where n is the value assigned
            animation_points: amount of points to animate
        """
        def banded_mv(A, x):
            y = A[1,:] * x
            y[:-1] += A[0,1:] * x[1:]
            y[1:]  += A[2,:-1] * x[:-1]
            return y

        fig1, ani_ax = plt.subplots()
        psis = [self.wave]

        potential_plot_normalize = 1 / max(self.barrier) * max(abs(psis[0])**2) * 1.2
        potential_plot = self.barrier * potential_plot_normalize
        ani_ax.plot(self.x_lattice, potential_plot, color="C3")
        line, = ani_ax.plot(self.x_lattice, abs(psis[0])**2)

        for i in range(animation_points):
            right_vector = banded_mv(
                Particle_Collision.crank_nicholson_matrix(self, i), psis[-1]
            )
            left_matrix = (
                Particle_Collision.crank_nicholson_matrix(self, 0)
            )
            new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
            psis.append(new_psi)

        def ani_func(index):
            normalizing_factor = 1 / (scipy.integrate.simpson(abs(psis[index * frame_space])**2, self.x_lattice))
            plot_func = normalizing_factor * abs((psis[index * frame_space]))**2
            line.set_ydata(plot_func)
            return line,

        number_frames = animation_points // frame_space
        ani = anim.FuncAnimation(fig1, ani_func, frames = number_frames, blit=False)
        return ani

def gaussian_potential(x_lattice: np.ndarray, height: float = 10, standard_deviation: float = 1, start_position: float = 40) -> np.ndarray:
        """
        Calculates the potential in form of a slim Gauss for a wave to collide with.

        paramenters:
            x_lattice: the values of which the potential is calculated
            height: height of the potential
            standard_deviation: how much the function deviates
            start_position: displacement of the top point along the x-axis in the positive direction
        """
        if standard_deviation == 0:
            raise Exception("The standard deviation cannot be 0")
        return height * np.exp(-1 * ((x_lattice - start_position)**2) / (4 * standard_deviation**2))

def gaussian_wave(x_lattice: np.ndarray, center_position: float = 0, center_velocity: float = 4, variance: float = 2):
    """
    Creates a wave with the form of a gaussian function.
    
    parameters:
        x_lattice: the values of which the wave is calculated
        center_position: position for the center of the wave
        center_velocity: velocity for the center of the wave
        variance: the variance of the gaussian function
    """
    factor_1 = 1 / (2 * np.pi * variance)**(1/4)
    factor_2 = np.exp(-(x_lattice - center_position)**2 / (4 * variance))
    factor_3 = np.exp(1j * x_lattice * center_velocity)
    wave = factor_1 * factor_2 * factor_3
    return wave

specific_collision = Particle_Collision(gaussian_wave, gaussian_potential)
specific_collision.animate_collision(10, 1000)
