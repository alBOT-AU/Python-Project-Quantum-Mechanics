import numpy as np

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
T_factor = 1 / (2*mass_electron*delta_x_lattice**2)
