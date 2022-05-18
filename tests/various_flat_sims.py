import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator

folder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
linear_size = 800
n_particles = 256
dim = 3

grid = Grid(linear_size, dim)
particles = ParticleSet(linear_size, dim, n_particles)
particles.uniformRandomPositions()
particles.zeroMomenta()

pm = ParticleMesh(grid, particles, method="CIC")
pm.densityFromParticles()

z_start = 1000
a_start = 1e-3
a_step = 1e-4
a_end = 1.0

matter_densities = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
lambda_densities = 1. - matter_densities
Ok0 = 0.

for Om0, Ode0 in zip(matter_densities, lambda_densities):

    savefile = folder + "flat_size800_N256_Omega_m_" + str(Om0) + ".hdf5"

    sim = Simulator(pm, a_start, a_step, Om0, Ode0, Ok0)
    sim.evolve(a_end, savefile)

    print ("Finished for $\Omega_m = $ {}".format(Om0))