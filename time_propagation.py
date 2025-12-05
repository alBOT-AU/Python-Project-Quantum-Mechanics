import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.integrate
from collections.abc import Callable
plt.rc("animation", html="jshtml")

class ParticleCollision:
    def __init__(
            self, wave: Callable, barrier: Callable, matrix_length: int = 3000,
            matrix_bounds: float = 150, mass: float = 1, dt: float = 0.1
    ):
        """
        Simulates a collision between a particle and a barrier, and shows quantum tunneling for
        certain energies of the particle and barrier.

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

    def __str__(self):
        return (f"Parameters:\n Matrix length = {self.matrix_length} \n Matrix bounds = {self.matrix_bounds}"
                f"\n Mass = {self.mass} \n dt = {self.dt}")

    def banded_mv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        A tool to solve the matrix-vector multiplication A * y = x for y.

        parameters:
            A: matrix
            x: vector
        """
        y = A[1,:] * x
        y[:-1] += A[0,1:] * x[1:]
        y[1:]  += A[2,:-1] * x[:-1]
        return y

    def double_deriv(self) -> np.ndarray:
        """
        Constructs the T matrix, which consists of a tri-diagonal matrix. All other points than the diagonal,
        super- and sub-diagonal have the value 0, so only those diagonals are represented as a matrix.
        """
        kinetic_matrix = np.zeros((3, self.matrix_length))

        kinetic_sub_diag = np.zeros((1, self.matrix_length - 1)) + 1
        kinetic_super_diag = np.zeros((1, self.matrix_length - 1)) + 1
        kinetic_diag = np.zeros((1, self.matrix_length)) -2

        kinetic_matrix[0,1:] = kinetic_sub_diag
        kinetic_matrix[1,:] = kinetic_diag
        kinetic_matrix[2,:-1] = kinetic_super_diag

        kinetic_factor = - 1 / (2* self.mass * self.delta_x**2)

        return kinetic_factor * kinetic_matrix

    def crank_nicholson_matrix(self, n) -> np.ndarray:
        """
        Creates the Crank-Nicholson matrix, a numerical way to solve the Schrödinger equation.

        parameters:
            n: even values (zero included) gives a positive sign, uneven values gives a negative sign
        """
        potential = self.barrier
        kinetic = ParticleCollision.double_deriv(self)
        hamiltonian = kinetic
        hamiltonian[1,:] += potential.copy()
        second_term = (-1)**n * ((1j * self.dt) / 2) * hamiltonian
        second_term[1,:] += 1
        return second_term

    def animate_collision(self, frame_space: int = 10, animation_points: int = 1000):
        """
        Animates the collision between the particle and the barrier.

        parameters:
            frame_space: chooses every n'th point to show in animation, where n is the value assigned
            animation_points: amount of points to animate
        """
        fig, ani_ax = plt.subplots()
        ani_ax.grid()
        ani_ax.set_xlabel("Position")
        ani_ax.set_ylabel("Probability")
        ani_ax.set_title("Normalized wave function and linear relation to barrier")
        psis = [self.wave]

        potential_plot_normalize = 1 / max(self.barrier) * max(abs(psis[0])**2) * 1.2
        potential_plot = self.barrier * potential_plot_normalize
        ani_ax.plot(self.x_lattice, potential_plot, color="C3", label = "Barrier")
        line, = ani_ax.plot(self.x_lattice, abs(psis[0])**2, label="Normalized wave function", color = "C0")
        ani_ax.legend()

        for i in range(animation_points):
            right_vector = ParticleCollision.banded_mv(
                ParticleCollision.crank_nicholson_matrix(self, i), psis[-1]
            )
            left_matrix = (
                ParticleCollision.crank_nicholson_matrix(self, 0)
            )
            new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
            psis.append(new_psi)

        def ani_func(index):
            """
            A part to help animate the collision.
            """
            normalizing_factor = 1 / (scipy.integrate.simpson(abs(psis[index * frame_space])**2, self.x_lattice))
            plot_func = normalizing_factor * abs((psis[index * frame_space]))**2
            line.set_ydata(plot_func)
            return line,

        number_frames = animation_points // frame_space
        ani = anim.FuncAnimation(fig, ani_func, frames = number_frames, blit=False)
        return ani

    def tunneling_chance(self) -> float:
        """
        Calculates the chance of quantum tunneling. Assumes that the barrier is at a certain position.
        """
        psis = [self.wave]
        for i in range(1000):
            right_vector = ParticleCollision.banded_mv(ParticleCollision.crank_nicholson_matrix(self, n = 1), psis[-1])
            left_matrix = ParticleCollision.crank_nicholson_matrix(self, n = 0)
            new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
            psis.append(new_psi)
        relative_tunneling_chance = scipy.integrate.simpson(abs(psis[1000][1620:])**2, self.x_lattice[1620:])
        real_tunneling_chance = relative_tunneling_chance / scipy.integrate.simpson(abs(psis[1000])**2, self.x_lattice)
        return real_tunneling_chance

def gaussian_potential(
        x_lattice: np.ndarray, height: float = 10, standard_deviation: float = 1, start_position: float = 30
) -> np.ndarray:
        """
        Calculates the potential in form of a slim Gauss for a wave to collide with.

        parameters:
            x_lattice: the values of which the potential is calculated
            height: height of the potential
            standard_deviation: how much the function deviates
            start_position: displacement of the top point along the x-axis in the positive direction
        """
        if standard_deviation == 0:
            raise Exception("The standard deviation cannot be 0")
        return height * np.exp(-1 * ((x_lattice - start_position)**2) / (4 * standard_deviation**2))

def gaussian_wave(
        x_lattice: np.ndarray, center_position: float = 0, center_velocity: float = 4, variance: float = 2
):
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

speed_chances = []
for speed in range(21, 71):
    specific_gaussian_wave = lambda x_lattice: gaussian_wave(x_lattice, center_velocity=speed/10)
    wave_0 = ParticleCollision(specific_gaussian_wave, gaussian_potential, matrix_length = 3000, matrix_bounds = 500, dt = 0.1)
    speed_chances.append(wave_0.tunneling_chance())

plot_over_tunneling_chances, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("Linear energy relation of most probable energy")
ax.set_ylabel("Chance of tunneling")
ax.plot((np.linspace(2, 7, 50))**2/2, speed_chances)
ax.set_title("Chance of tunneling for different gaussian wave functions")

specific_collision = ParticleCollision(gaussian_wave, gaussian_potential, dt = 0.1)
print(specific_collision)
specific_collision.animate_collision(5, 1000)
