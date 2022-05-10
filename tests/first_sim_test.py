import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator

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
ax.set_title("Initial Densities")
fig.show()

# evolve the system
sim = Simulator(pm, 0.01, 0.0001)
sim.evolve(0.1)

fig2, ax2 = plt.subplots(dpi=240)
ax2.set_aspect("equal")
im2 = ax2.pcolormesh(grid.x_mids, grid.y_mids, grid.densities)
cbar2 = fig2.colorbar(im2, ax=ax2, label=r"$\rho$")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(r"Density Field at $a = 0.1$")
fig2.show()