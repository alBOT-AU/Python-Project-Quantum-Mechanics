import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.integrate

class EigenvaluesHarmonicOscillator():
    def __init__(self, matrix_length: int = 2000, boundaries: float = 15, 
                 angular_frequency: float = 2, mass: float = 1):
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

    def double_deriv(self) -> np.ndarray:
        """
        Constructs the T matrix, which consists of a tri-diagonal matrix. 
        All other points than the diagonal, super- and sub-diagonal have 
        the value 0, so only those diagonals are represented as a matrix.
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

    def potential(self):
        """
        Constructs the V matrix, which consists of a tri-diagonal matrix. 
        All other points than the diagonal, super- and sub-diagonal have 
        the value 0, so only those diagonals are represented as a matrix.
        """
        omega_square = self.angular_frequency**2
        x_square = self.x_lattice**2
        potential_diag = omega_square * x_square / 2
        potential =  np.zeros((3, self.matrix_length))
        potential[1,:] = potential_diag
        return potential

    def get_normalized_eigenfunction(self, n: int)->None:
        """
        Calculates the normalized eigenfunction for the n'th eigenvalue

        parameters:
            n: eigenfunction number
        """
        double_deriv = EigenvaluesHarmonicOscillator.double_deriv(self)
        potential = EigenvaluesHarmonicOscillator.potential(self)
        hamiltonian = double_deriv + potential
        
        eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(hamiltonian[1,:], hamiltonian[0, 1:])
        normalizing_factor = 1 / (scipy.integrate.simpson(abs(eigvecs[:, n])**2, self.x_lattice))
        return abs(eigvecs[:, n])**2*normalizing_factor, eigvals

fig, ax = plt.subplots(10, 1, sharex=True, figsize=(7,10))
fig.tight_layout()
specific_case = EigenvaluesHarmonicOscillator()
ax[9].set_xlabel("position")
ax[4].set_ylabel(r"|$\psi(x)^2$|")
ax[0].set_title("Plots of different eigenvectors for the harmonic oscillator")
for i in range(10):
    normalized_eigenfunction, eigenvalues = specific_case.get_normalized_eigenfunction(i)
    ax[i].plot(specific_case.x_lattice, normalized_eigenfunction)
    ax[i].set_yticks([0,0.5,1])
print(eigenvalues[:10])
plt.show()
