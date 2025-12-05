speed_chances = []
for speed in range(21, 71):
    wave_0 = construct_wave(x_lattice, variance, speed/10, center_position)
    psis = [wave_0]
    for i in range(1000):
        right_vector = banded_mv(crank_nicholson_matrix(points, x_lattice, dt, 0, 1), psis[-1])
        left_matrix = crank_nicholson_matrix(points, x_lattice, dt, 0)
        new_psi = scipy.linalg.solve_banded((1, 1), left_matrix, right_vector)
        psis.append(new_psi)
    relative_tunneling_chance = scipy.integrate.simpson(abs(psis[1000][1620:])**2, x_lattice[1620:])
    real_tunneling_chance = relative_tunneling_chance / scipy.integrate.simpson(abs(psis[1000])**2, x_lattice)
    print(f"Iteration number: {speed}\nTunneling chance: {real_tunneling_chance}")
    speed_chances.append(real_tunneling_chance)


fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("Linear energy relation of most probable energy")
ax.set_ylabel("Chance of tunneling")
ax.plot((np.linspace(2, 7, 50))**2/2, speed_chances)
ax.set_title("Chance of tunneling for different gaussian wave functions")
plt.savefig("Plot")
