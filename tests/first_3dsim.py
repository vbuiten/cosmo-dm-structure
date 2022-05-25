import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator
import numpy as np

n_particles = 8**3
size = 20

folder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
savefile = "{}size{}_N{}_test.hdf5".format(folder, size, n_particles)

grid = Grid(size, 3)
particles = ParticleSet(size, 3, n_particles)
particles.uniformRandomPositions()
particles.zeroMomenta()

pm = ParticleMesh(grid, particles, method="CIC")
pm.densityFromParticles()

a_start = 0.01

fig = plt.figure(dpi=240)
ax = fig.add_subplot(projection="3d")
ax.plot(particles.positions[:,0], particles.positions[:,1], particles.positions[:,2], c="black", marker=".",
        alpha=0.7, ls="")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Initial Overdensities at $a = $ {}".format(a_start))
fig.show()

a_step = 0.0001
a_end = 0.2

# evolve the system
sim = Simulator(pm, a_start, a_step, Om0=.3, Ode0=.7, Ok0=0.)
sim.evolve(a_end, savefile=savefile, save_err_thresh=1)

fig2 = plt.figure(dpi=240)
ax2 = fig2.add_subplot(projection="3d")
ax2.plot(particles.positions[:,0], particles.positions[:,1], particles.positions[:,2], c="black", marker=".",
        alpha=0.7, ls="")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(r"Overdensity Field at $a = $ {}".format(a_end))
fig2.show()