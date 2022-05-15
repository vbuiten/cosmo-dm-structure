import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator
import numpy as np

grid = Grid(50, 2)
particles = ParticleSet(50, 2, 500)
particles.uniformRandomPositions()
#particles.zeroMomenta()
particles.momenta = np.random.normal(0., 1., size=particles.positions.shape)

pm = ParticleMesh(grid, particles)
pm.densityFromParticles()

fig, ax = plt.subplots(dpi=240)
ax.set_aspect("equal")
im = ax.pcolormesh(grid.x_mids, grid.y_mids, grid.densities)
ax.plot(particles.positions[:,0], particles.positions[:,1], c="black", marker=".",
        alpha=0.7, ls="")
cbar = fig.colorbar(im, ax=ax, label=r"$\rho$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Initial Densities")
fig.show()

# evolve the system
sim = Simulator(pm, 0.001, 0.0001, Om0=.3, Ode0=0.7, Ok0=0.)
sim.evolve(1.0)

fig2, ax2 = plt.subplots(dpi=240)
ax2.set_aspect("equal")
im2 = ax2.pcolormesh(grid.x_mids, grid.y_mids, grid.densities)
ax2.plot(particles.positions[:,0], particles.positions[:,1], c="black", marker=".",
        alpha=0.7, ls="")
cbar2 = fig2.colorbar(im2, ax=ax2, label=r"$\rho$")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(r"Density Field at $a = 0.3$")
fig2.show()