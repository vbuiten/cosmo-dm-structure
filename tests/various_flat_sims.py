import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "serif"
from framework.mesh import Grid
from framework.particles import ParticleSet
from framework.particle_mesh import ParticleMesh
from simulation.simulator import Simulator

'''
folder = "/net/vdesk/data2/buiten/COP/cosmo_sims_data/"
'''

folder = r"C:\Users\victo\Documents\Uni\COP\open_project\sim_data\\"
linear_size = 20
n_particles = 8**3
dim = 3

grid = Grid(linear_size, dim)
particles = ParticleSet(linear_size, dim, n_particles)
particles.uniformRandomPositions()
particles.zeroMomenta()

pm = ParticleMesh(grid, particles, method="CIC")
pm.densityFromParticles()

a_start = 1e-2
a_step = 1e-3
a_step_save = 1e-3
a_end = 1.0

matter_densities = np.array([0.3, 0.6, 0.9])
lambda_densities = 1. - matter_densities
Ok0 = 0.

for Om0, Ode0 in zip(matter_densities, lambda_densities):

    savefile = "{}flat_size{}_N{}_Omega_m_{}.hdf5".format(folder, linear_size, n_particles, Om0)

    print ("Starting simulation for $\Omega_m =$ {}".format(Om0))
    sim = Simulator(pm, a_start, a_step, Om0, Ode0, Ok0)
    sim.evolve(a_end, savefile, save_step=a_step_save, save_err_thresh=1e-3)

    print ("Finished for $\Omega_m = $ {}".format(Om0))