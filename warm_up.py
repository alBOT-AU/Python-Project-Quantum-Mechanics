import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate

class Eigenvalues_Harmonic_Oscillator():
    def __init__(self, matrix_length: int = 2000, boundaries: float = 15, angular_frequency: float = 2, mass: float = 1):
        """
        Simulates the harmonic oscillation of a particle

        parameters:
            matrix_length: amount of points in th matrix
            boundaries: the positive and negative boundaries along the x-axis
            angular_frequency: angular frequency of the particle
            mass: mass of the particle
        """
        self.matrix_length = matrix_length
        self.angular_frequency = angular_frequency
        self.mass = mass
        self.x_lattice = np.linspace(-boundaries, boundaries, self.matrix_length)
        self.delta_x = (self.x_lattice[-1] - self.x_lattice[0]) / self.matrix_length

    def construct_T(self) -> np.ndarray:
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

    def construct_V(self):
        """
        Constructs the V matrix, which consists of a tri-diagonal matrix. All other points than the diagonal,
        super- and sub-diagonal have the value 0, so only those diagonals are represented as a matrix.
        """
        omega_square = self.angular_frequency**2
        x_square = self.x_lattice**2
        V_diag = omega_square * x_square / 2
        V =  np.zeros((3, self.matrix_length))
        V[1,:] = V_diag
        return V

    def get_normalized_eigenfunction(self, n: int)->None:
        """
        Calculates the normalized eigenfunction for the n'th eigenvalue

        parameters:
            n: eigenfunction number
        """
        H = Eigenvalues_Harmonic_Oscillator.construct_T(self) + Eigenvalues_Harmonic_Oscillator.construct_V(self)

        eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(H[1,:], H[0, 1:])
        normalizing_factor = 1 / (scipy.integrate.simpson(abs(eigvecs[:, n])**2, self.x_lattice))
        return abs(eigvecs[:, n])**2*normalizing_factor

fig, ax = plt.subplots(10, 1)
specific_case = Eigenvalues_Harmonic_Oscillator()
for i in range(10):
    ax[i].plot(specific_case.x_lattice, specific_case.get_normalized_eigenfunction(i))
