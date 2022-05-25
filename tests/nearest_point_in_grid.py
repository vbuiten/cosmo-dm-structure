'''File for checking the potential computation in 2D.'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh

grid = Grid(50, 2)
particles = ParticleSet(50, 2, 64)
particles.uniformRandomPositions()
particles.zeroMomenta()
pm = ParticleMesh(grid, particles)
pm.densityFromParticles()

fig, ax = plt.subplots(dpi=240)
ax.set_aspect("equal")
im = ax.pcolormesh(grid.x_mids, grid.y_mids, grid.densities)
cbar = fig.colorbar(im, ax=ax, label=r"$\rho$")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.show()

# solve Poisson's equation to find the potential
potential = pm.grid.potential(Om0=0.3, scale_factor=1.)

fig2, ax2 = plt.subplots(dpi=240)
ax2.set_aspect("equal")
im2 = ax2.pcolormesh(grid.x_mids, grid.y_mids, potential.real, cmap="bwr")
cbar2 = fig2.colorbar(im2, ax=ax2, label=r"$\phi (\mathbf{x})$")
ax2.set_xlabel(r"$k_x$")
ax2.set_ylabel(r"$k_y$")
fig2.show()