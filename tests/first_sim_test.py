'''File for testing 2D simulation code and the density assignment.'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from matplotlib.colors import TwoSlopeNorm
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator
import numpy as np

'''
datafolder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
filename = datafolder+"size10_N1000_test.hdf5"
'''
datafolder = r"C:\Users\victo\Documents\Uni\COP\open_project\sim_data"
filename = datafolder + r"\size20_N28x28_2d.hdf5"

grid = Grid(20, 2)
particles = ParticleSet(20, 2, 32**2)
#particles.positions = np.array([[20,20], [40,25]])
particles.uniformRandomPositions()
#particles.positions = np.random.normal((25,25), (10,10), size=(128,2))
particles.zeroMomenta()
#particles.momenta = np.random.normal(0., 1., size=particles.positions.shape)

pm = ParticleMesh(grid, particles, method="CIC")
pm.densityFromParticles()
a_start = 0.001

fig, ax = plt.subplots(dpi=240)
ax.set_aspect("equal")
im = ax.pcolormesh(grid.x_mids, grid.y_mids, grid.overdensities, cmap="RdBu",
                   norm=TwoSlopeNorm(vcenter=0.))
ax.plot(particles.positions[:,0], particles.positions[:,1], c="black", marker="o",
        alpha=0.3, ls="")
cbar = fig.colorbar(im, ax=ax, label=r"$\delta$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Initial Overdensities at $a ="+str(a_start)+"$")
fig.show()

# evolve the system
a_step = 0.0001
a_end = .2
sim = Simulator(pm, a_start, a_step, Om0=.3, Ode0=.7, Ok0=0.)
sim.evolve(a_end, savefile=filename, save_step=0.001)

fig2, ax2 = plt.subplots(dpi=240)
ax2.set_aspect("equal")
im2 = ax2.pcolormesh(grid.x_mids, grid.y_mids, grid.overdensities, cmap="RdBu",
                     norm=TwoSlopeNorm(vcenter=0.))
ax2.plot(particles.positions[:,0], particles.positions[:,1], c="black", marker="o",
        alpha=0.3, ls="", markersize=5)
cbar2 = fig2.colorbar(im2, ax=ax2, label=r"$\delta$")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(r"Overdensity Field at $a ="+str(a_end)+"$")
fig2.show()