import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator
import numpy as np

grid = Grid(10, 3)
particles = ParticleSet(10, 3, 1000)
particles.uniformRandomPositions()
particles.zeroMomenta()

pm = ParticleMesh(grid, particles, method="CIC")
pm.densityFromParticles()

fig = plt.figure(dpi=240)
ax = fig.add_subplot(projection="3d")
ax.plot(particles.positions[:,0], particles.positions[:,1], particles.positions[:,2], c="black", marker=".",
        alpha=0.7, ls="")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Initial Overdensities at $a = 0.00001$")
fig.show()

# evolve the system
sim = Simulator(pm, 0.00001, 0.00001, Om0=.3, Ode0=.7, Ok0=0.)
sim.evolve(.5)

fig2 = plt.figure(dpi=240)
ax2 = fig2.add_subplot(projection="3d")
ax2.plot(particles.positions[:,0], particles.positions[:,1], particles.positions[:,2], c="black", marker=".",
        alpha=0.7, ls="")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(r"Overdensity Field at $a = 0.5$")
fig2.show()